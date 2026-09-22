"""Offline tests for the NOWEB store history backfill (Phase 6 admin endpoint).

Covers the shared payload -> Message mapping (incl. fromMe=true bot-sent rows),
the store fetch -> dedup -> rebuild pipeline, and the /admin/store-backfill route
delegation. Pure/stdlib + mocks - no DB, no network, no API keys.
"""

import sys
import unittest
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import MagicMock

sys.path.insert(0, "/home/nate/promptcraft_chatbot/unipods-bot/backend")

from app.ingest import waha_store_backfill as backfill_mod  # noqa: E402
from app.ingest.waha_mapper import build_message  # noqa: E402
from app.models import Chunk, Message, Session  # noqa: E402

CHAT_A = "120363411777113932@g.us"
BOT_JID = "251947711181@c.us"

# Real-shape NOWEB store message objects (GET /{session}/chats/{chatId}/messages).
INBOUND_ID = "false_120363411777113932@g.us_ACA9D3CF3FDCC1E0CEFFAE7180413543"
BOT_OUT_ID = "true_120363411777113932@g.us_3EB0965B123D06E0B70741F0C0A5B3C4"

INBOUND = {
    "id": INBOUND_ID,
    "timestamp": 1789970115,
    "from": CHAT_A,
    "fromMe": False,
    "body": "Does the store hold this?",
    "hasMedia": False,
    "ack": 3,
    "ackName": "READ",
    "replyTo": None,
    "_data": {"pushName": "Alice"},
}

BOT_OUT = {
    "id": BOT_OUT_ID,
    "timestamp": 1790002008,
    "from": CHAT_A,
    "fromMe": True,
    "body": "Yes - asking you live now.",
    "hasMedia": False,
    "ack": 1,
    "ackName": "SERVER",
    "replyTo": None,
    "_data": {},
}


class MapperTests(unittest.TestCase):
    """build_message: store payloads map to rows just like webhook payloads."""

    def test_inbound_group_message(self):
        row = build_message(INBOUND, session_name="default", raw_payload=INBOUND)
        self.assertEqual(row.waha_message_id, INBOUND_ID)
        self.assertEqual(row.chat_id, CHAT_A)
        self.assertTrue(row.is_group)
        self.assertFalse(row.from_me)
        # store messages carry no `participant`, so sender falls back to `from`
        self.assertEqual(row.sender_id, CHAT_A)
        self.assertEqual(row.sender_name, "Alice")
        self.assertEqual(row.body, "Does the store hold this?")
        self.assertEqual(row.msg_type, "text")
        self.assertEqual(row.timestamp, datetime.fromtimestamp(1789970115, tz=timezone.utc))
        self.assertIs(row.raw_payload, INBOUND)  # stored verbatim, wrapper is caller's job

    def test_from_me_attributes_bot_sender(self):
        row = build_message(BOT_OUT, me={"id": BOT_JID, "pushName": "UniPods"})
        self.assertTrue(row.from_me)
        self.assertEqual(row.chat_id, CHAT_A)
        self.assertTrue(row.is_group)
        self.assertEqual(row.sender_id, BOT_JID)
        self.assertEqual(row.sender_name, "UniPods")
        self.assertEqual(row.timestamp, datetime.fromtimestamp(1790002008, tz=timezone.utc))

    def test_normalizes_s_whatsapp_jids(self):
        payload = dict(INBOUND, **{"from": "120363411777113932@s.whatsapp.net"})
        row = build_message(payload)
        self.assertEqual(row.chat_id, "120363411777113932@c.us")

    def test_media_mime_drives_msg_type(self):
        payload = dict(BOT_OUT, body=None, media={"mimetype": "image/jpeg"})
        row = build_message(payload, me={"id": BOT_JID})
        self.assertEqual(row.msg_type, "image")
        self.assertEqual(row.media_mime, "image/jpeg")


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class _FakeBackfillDB:
    """Minimal Session stand-in: serves existing waha ids, collects added rows."""

    def __init__(self, existing_ids=()):
        self.existing_ids = list(existing_ids)
        self.added = []

    def execute(self, stmt):
        return _Rows([(wid,) for wid in self.existing_ids])

    def flush(self):
        # Simulate the real SQLAlchemy flush: pending adds become queryable rows.
        for obj in self.added:
            if obj.waha_message_id and obj.waha_message_id not in self.existing_ids:
                self.existing_ids.append(obj.waha_message_id)

    def add(self, obj):
        self.added.append(obj)


def _run_backfill(fake_db, fetch_result, *, bot_id=BOT_JID):
    with mock.patch.object(
        backfill_mod, "fetch_store_messages", return_value=fetch_result
    ), mock.patch.object(
        backfill_mod,
        "rebuild_sessions_and_chunks",
        return_value={"sessions_built": 1, "chunks_written": 3},
    ) as mock_rebuild, mock.patch.object(
        backfill_mod.settings, "bot_whatsapp_id", bot_id
    ):
        result = backfill_mod.backfill_chat_from_store(fake_db, "default", CHAT_A)
    return result, mock_rebuild


class BackfillTests(unittest.TestCase):
    """backfill_chat_from_store: fetch -> dedup (@lid aware) -> insert -> rebuild."""

    def test_inserts_bot_message_and_skips_lid_suffixed_live_capture(self):
        # The inbound message was already captured live by the webhook, which
        # persisted its id WITH the group participant's @lid suffix. The store id
        # (no suffix) must still be recognized as a duplicate.
        existing = [INBOUND_ID + "_126474740867226@lid"]
        fake_db = _FakeBackfillDB(existing)
        result, mock_rebuild = _run_backfill(fake_db, [INBOUND, BOT_OUT])

        self.assertEqual(result["fetched"], 2)
        self.assertEqual(result["inserted"], 1)
        self.assertEqual(result["already_existed"], 1)
        self.assertEqual(result["sessions_built"], 1)
        self.assertEqual(result["chunks_written"], 3)

        self.assertEqual(len(fake_db.added), 1)
        row = fake_db.added[0]
        self.assertEqual(row.waha_message_id, BOT_OUT_ID)
        self.assertTrue(row.from_me)
        self.assertEqual(row.sender_id, BOT_JID)
        mock_rebuild.assert_called_once_with(fake_db, CHAT_A)

    def test_rerun_is_idempotent(self):
        fake_db = _FakeBackfillDB()
        first, _ = _run_backfill(fake_db, [INBOUND, BOT_OUT])
        self.assertEqual(first["inserted"], 2)

        # Second run against the same DB (flush already persisted the ids, and the
        # route committed) inserts nothing.
        second, mock_rebuild = _run_backfill(fake_db, [INBOUND, BOT_OUT])

        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["already_existed"], 2)
        self.assertEqual(len(fake_db.added), 2)
        # the store still returned rows, so the rebuild re-ran as an index-repair
        # pass - it must see ALL rows (already committed + none new), so the
        # rebuilt index is complete even after a no-op backfill.
        mock_rebuild.assert_called_once_with(fake_db, CHAT_A)

    def test_empty_store_is_a_safe_noop(self):
        fake_db = _FakeBackfillDB()
        result, mock_rebuild = _run_backfill(fake_db, [])

        self.assertEqual(result["fetched"], 0)
        self.assertEqual(result["inserted"], 0)
        self.assertEqual(result["already_existed"], 0)
        self.assertEqual(fake_db.added, [])
        mock_rebuild.assert_not_called()

    def test_from_me_without_bot_id_attr_falls_back_to_group_jid(self):
        fake_db = _FakeBackfillDB()
        result, _ = _run_backfill(fake_db, [BOT_OUT], bot_id="")

        self.assertEqual(result["inserted"], 1)
        row = fake_db.added[0]
        self.assertTrue(row.from_me)
        self.assertEqual(row.sender_id, CHAT_A)


class _ScalarMessageRows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _ExecResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _ScalarMessageRows(self._rows)

    def all(self):
        return self._rows


class _DeleteRowCount:
    rowcount = 0


class _FakeRebuildBackfillDB:
    """Drives the REAL rebuild path under backfill (autoflush=False semantics).

    ``add`` keeps Messages pending until ``flush``; ``execute`` serves the
    waha-id dedup SELECT from committed rows and evaluates the
    ``from_me IS false`` filter on the rebuild's message SELECT exactly the way
    the real DB would - so removing the filter from matching_message_dicts makes
    bot rows visible again and the assertion below fails loudly.
    """

    def __init__(self, committed=()):
        self.committed = list(committed)
        self.pending = []
        self.added_sessions = []
        self.added_chunks = []

    def add(self, obj):
        if isinstance(obj, Message):
            self.pending.append(obj)
        elif isinstance(obj, Session):
            self.added_sessions.append(obj)
        elif isinstance(obj, Chunk):
            self.added_chunks.append(obj)

    def flush(self):
        self.committed.extend(self.pending)
        self.pending = []

    def execute(self, stmt):
        s = str(stmt)
        if "DELETE FROM" in s:
            return _DeleteRowCount()
        if "IS NOT NULL" in s:  # backfill dedup: existing waha ids
            return _Rows([(m.waha_message_id,) for m in self.committed if m.waha_message_id])
        if "FROM messages" in s:  # rebuild selection
            if "from_me" in s and "false" in s:
                return _ExecResult([m for m in self.committed if not m.from_me])
            return _ExecResult(self.committed)
        return _ExecResult([])


class FromMeExclusionBackfillTests(unittest.TestCase):
    """Store backfill entry point: from_me rows stored but never sessionized."""

    def _ts(self, hh, mm):
        return int(datetime(2026, 9, 21, hh, mm, tzinfo=timezone.utc).timestamp())

    def test_from_me_rows_excluded_from_chunks_through_backfill(self):
        from app.ingest import rebuild as rebuild_mod

        human_seed = Message(
            waha_message_id="seed_h", session_name="default", chat_id=CHAT_A,
            chat_name="Test Group", sender_name="Alice", from_me=False,
            msg_type="text", body="pre-existing human note",
            timestamp=datetime.fromtimestamp(self._ts(9, 59), tz=timezone.utc),
            raw_payload={"source": "test_seed"},
        )
        bot_seed = Message(
            waha_message_id="seed_b", session_name="default", chat_id=CHAT_A,
            chat_name="Test Group", sender_name="UniPods", from_me=True,
            msg_type="text", body="pre-existing bot reply",
            timestamp=datetime.fromtimestamp(self._ts(10, 0), tz=timezone.utc),
            raw_payload={"source": "test_seed"},
        )
        fake = _FakeRebuildBackfillDB([human_seed, bot_seed])

        store_human = dict(
            INBOUND, id="store_h", body="Does the store hold this?",
            timestamp=self._ts(10, 1),
        )
        store_bot = dict(
            BOT_OUT, id="store_b", body="Yes - asking you live now.",
            timestamp=self._ts(10, 2),
        )
        with mock.patch.object(
            backfill_mod, "fetch_store_messages", return_value=[store_human, store_bot]
        ), mock.patch.object(backfill_mod.settings, "bot_whatsapp_id", BOT_JID):
            result = backfill_mod.backfill_chat_from_store(fake, "default", CHAT_A)

        self.assertEqual(result["inserted"], 2)
        self.assertEqual(result["sessions_built"], 1)
        self.assertEqual(result["chunks_written"], 1)

        content = fake.added_chunks[0].content
        self.assertIn("pre-existing human note", content)
        self.assertIn("Does the store hold this?", content)
        self.assertNotIn("Yes - asking you live now.", content)
        self.assertNotIn("pre-existing bot reply", content)
        # from_me rows stay stored (never deleted) - just not in chunks.
        self.assertTrue(
            any(m.from_me for m in fake.committed),
        )
        self.assertIs(
            rebuild_mod.rebuild_sessions_and_chunks,
            backfill_mod.rebuild_sessions_and_chunks,
        )


class RouteTests(unittest.TestCase):
    """POST /admin/store-backfill/{chat_id} delegates + commits."""

    def test_route_delegates_and_commits(self):
        import app.routes_admin as routes

        stats = {
            "fetched": 2,
            "inserted": 1,
            "already_existed": 1,
            "sessions_built": 1,
            "chunks_written": 3,
        }
        with mock.patch.object(
            routes, "backfill_chat_from_store", return_value=stats
        ) as mock_backfill:
            db = MagicMock()
            resp = routes.store_backfill(chat_id=CHAT_A, db=db, _=None)

        mock_backfill.assert_called_once_with(db, routes.settings.waha_session, CHAT_A)
        db.commit.assert_called_once()
        self.assertEqual(resp.fetched, 2)
        self.assertEqual(resp.inserted, 1)
        self.assertEqual(resp.chat_id, CHAT_A)
        self.assertEqual(resp.session_name, routes.settings.waha_session)

    def test_route_rolls_back_on_error(self):
        import app.routes_admin as routes

        with mock.patch.object(
            routes, "backfill_chat_from_store", side_effect=RuntimeError("boom")
        ):
            db = MagicMock()
            with self.assertRaises(Exception):
                routes.store_backfill(chat_id=CHAT_A, db=db, _=None)
        db.rollback.assert_called_once()
        db.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()