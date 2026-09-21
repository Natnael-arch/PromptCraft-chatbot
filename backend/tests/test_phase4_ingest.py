"""Offline tests for the Phase 4 live-ingest pipeline (auto-index webhook messages).

Pure/stdlib + mocks - no DB, no network, no API keys. The rebuild pipeline is
tested with the REAL sessionizer and the deterministic mock embedder (the debug
default), while live_ingest.schedule_incremental_ingest is driven through a tiny
stateful fake DB + a controllable debounce clock.
"""

import sys
import unittest
import uuid
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import MagicMock

sys.path.insert(0, "/home/nate/promptcraft_chatbot/unipods-bot/backend")

from app.ingest import live_ingest, rebuild  # noqa: E402
from app.models import ChatIngestState, Chunk, Session as ChatSession  # noqa: E402
from app.reply.rate_limit import InMemoryCooldown  # noqa: E402

CHAT_A = "111222333@g.us"
CHAT_B = "444555666@g.us"
MSG_1 = uuid.uuid4()
MSG_2 = uuid.uuid4()

DEBOUNCE_WINDOW = 45.0

MESSAGE_DICTS = [
    {
        "id": str(MSG_1),
        "sender_name": "Alice",
        "body": "shared secret plan: ship friday",
        "msg_type": "text",
        "timestamp": datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc),
        "content": True,
    },
    {
        "id": str(MSG_2),
        "sender_name": "Bob",
        "body": "confirm at noon",
        "msg_type": "text",
        "timestamp": datetime(2026, 9, 21, 10, 1, tzinfo=timezone.utc),
        "content": True,
    },
]


def _fake_clock(state: list):
    return lambda: state[0]


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeLiveDB:
    """Minimal stateful stand-in for a SessionLocal on one chat's ingest flow."""

    def __init__(self):
        self.newest = None
        self.cursor_row = None
        self.adds = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def execute(self, stmt):
        s = str(stmt)
        if "chat_ingest_state" in s:
            return _ScalarResult(self.cursor_row)
        if "created_at" in s and "ORDER BY" in s.upper():
            return _ScalarResult(self.newest)
        return MagicMock()

    def add(self, obj):
        if isinstance(obj, ChatIngestState):
            self.cursor_row = obj
        self.adds.append(obj)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class RebuildPipelineTests(unittest.TestCase):
    """rebuild_sessions_and_chunks runs the REAL sessionizer + mock embedder."""

    def test_rebuild_includes_both_messages_in_chunk_output(self):
        db = MagicMock()
        with mock.patch.object(
            rebuild, "matching_message_dicts", return_value=MESSAGE_DICTS
        ):
            stats = rebuild.rebuild_sessions_and_chunks(db, CHAT_A)

        self.assertEqual(stats["sessions_built"], 1)
        self.assertEqual(stats["chunks_written"], 1)

        rows = [c.args[0] for c in db.add.call_args_list]
        session_rows = [r for r in rows if isinstance(r, ChatSession)]
        chunk_rows = [r for r in rows if isinstance(r, Chunk)]
        self.assertEqual(len(session_rows), 1)
        self.assertEqual(session_rows[0].message_ids, [str(MSG_1), str(MSG_2)])

        self.assertEqual(len(chunk_rows), 1)
        content = chunk_rows[0].content
        self.assertIn("shared secret plan: ship friday", content)
        self.assertIn("confirm at noon", content)

    def test_rebuild_returns_zero_counts_for_empty_chat(self):
        db = MagicMock()
        with mock.patch.object(rebuild, "matching_message_dicts", return_value=[]):
            stats = rebuild.rebuild_sessions_and_chunks(db, CHAT_A)
        self.assertEqual(stats, {"sessions_built": 0, "chunks_written": 0})

    def test_rebuild_never_commits(self):
        # the caller owns the transaction boundary
        db = MagicMock()
        with mock.patch.object(
            rebuild, "matching_message_dicts", return_value=MESSAGE_DICTS
        ):
            rebuild.rebuild_sessions_and_chunks(db, CHAT_A)
        db.commit.assert_not_called()
        db.rollback.assert_not_called()


class IngestRouteTests(unittest.TestCase):
    """POST /ingest/export delegates to the shared rebuild (single source)."""

    class _FakeFile:
        class _Buff:
            def read(self, n):
                return b"2026-09-21, 10:00 - Alice: hello"

        filename = "export.txt"
        file = _Buff()

    def test_export_route_uses_shared_rebuild(self):
        import app.routes_ingest as routes

        from types import SimpleNamespace

        parsed = SimpleNamespace(
            records=[SimpleNamespace()], source="export.txt", chat_name="Test Group"
        )
        with mock.patch.object(routes, "parse_export", return_value=parsed), \
             mock.patch.object(routes, "_insert_messages", return_value=(1, {"text": 1})), \
             mock.patch.object(
                 routes, "rebuild_sessions_and_chunks",
                 return_value={"sessions_built": 1, "chunks_written": 3},
             ) as mock_rebuild:
            db = MagicMock()
            resp = routes.ingest_export(
                file=self._FakeFile(),
                chat_id=CHAT_A,
                db=db,
            )

        mock_rebuild.assert_called_once_with(db, CHAT_A)
        self.assertEqual(resp.messages_imported, 1)
        self.assertEqual(resp.sessions_built, 1)
        self.assertEqual(resp.chunks_written, 3)
        db.commit.assert_called_once()


class ScheduleIngestTests(unittest.TestCase):
    """schedule_incremental_ingest: debounce, cursor skip, per-chat isolation."""

    def setUp(self):
        self.clock = [1000.0]
        live_ingest._ingest_cooldown = InMemoryCooldown(
            DEBOUNCE_WINDOW, now=_fake_clock(self.clock)
        )
        live_ingest._reset()  # clear module-level per-chat locks
        self.rebuild_patcher = mock.patch.object(
            live_ingest,
            "rebuild_sessions_and_chunks",
            return_value={"sessions_built": 1, "chunks_written": 1},
        )
        self.rebuild_mock = self.rebuild_patcher.start()
        self.addCleanup(self.rebuild_patcher.stop)
        # restore the real cooldown after each test
        self.addCleanup(self._restore_cooldown)

    def _restore_cooldown(self):
        live_ingest._ingest_cooldown = InMemoryCooldown(
            live_ingest.settings.ingest_debounce_seconds
        )
        live_ingest._reset()

    def _run(self, fake_db):
        # keep the SessionLocal patch active for the whole test (not just this call)
        patcher = mock.patch.object(live_ingest, "SessionLocal", return_value=fake_db)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake_db

    def test_two_messages_within_window_trigger_one_rebuild(self):
        # Both messages committed before the task runs; the debounce collapses
        # the two triggers into a single rebuild that sees both.
        db = _FakeLiveDB()
        db.newest = MSG_2

        self._run(db)
        live_ingest.schedule_incremental_ingest(CHAT_A)
        live_ingest.schedule_incremental_ingest(CHAT_A)

        self.assertEqual(self.rebuild_mock.call_count, 1)
        self.rebuild_mock.assert_called_once_with(db, CHAT_A)
        self.assertEqual(db.commits, 1)
        # both messages committed before the single rebuild -> cursor is the newest
        self.assertEqual(db.cursor_row.last_message_id, MSG_2)

    def test_skipped_when_no_messages_newer_than_cursor(self):
        db = _FakeLiveDB()
        db.newest = MSG_1

        self._run(db)
        live_ingest.schedule_incremental_ingest(CHAT_A)  # first rebuild, cursor set
        self.assertEqual(self.rebuild_mock.call_count, 1)

        # after the debounce window elapses, and with NO new message, the cursor
        # still equals the newest -> the rebuild is skipped entirely.
        self.clock[0] += DEBOUNCE_WINDOW + 1
        live_ingest.schedule_incremental_ingest(CHAT_A)

        self.assertEqual(self.rebuild_mock.call_count, 1)
        self.assertEqual(db.commits, 1)
        self.assertEqual(db.cursor_row.last_message_id, MSG_1)

    def test_second_rebuild_after_window_with_new_message_advances_cursor(self):
        db = _FakeLiveDB()
        db.newest = MSG_1

        self._run(db)
        live_ingest.schedule_incremental_ingest(CHAT_A)
        self.assertEqual(self.rebuild_mock.call_count, 1)
        self.assertEqual(db.cursor_row.last_message_id, MSG_1)

        self.clock[0] += DEBOUNCE_WINDOW + 1
        db.newest = MSG_2  # a new message arrived after the window
        live_ingest.schedule_incremental_ingest(CHAT_A)

        self.assertEqual(self.rebuild_mock.call_count, 2)
        self.assertEqual(db.commits, 2)
        self.assertEqual(db.cursor_row.last_message_id, MSG_2)

    def test_other_chat_not_triggered_or_blocked(self):
        db = _FakeLiveDB()
        db.newest = None  # CHAT_B has no captured messages

        self._run(db)
        live_ingest.schedule_incremental_ingest(CHAT_B)  # no-op (no messages)
        live_ingest.schedule_incremental_ingest(CHAT_B)  # no-op again

        self.assertEqual(self.rebuild_mock.call_count, 0)

        # CHAT_A's cooldown key is independent: the CHAT_B no-ops above must not
        # have debounced it.
        db.newest = MSG_1
        live_ingest.schedule_incremental_ingest(CHAT_A)
        self.assertEqual(self.rebuild_mock.call_count, 1)
        self.rebuild_mock.assert_called_once_with(db, CHAT_A)

    def test_disabled_by_kill_switch(self):
        db = _FakeLiveDB()
        db.newest = MSG_1
        self._run(db)
        with mock.patch.object(live_ingest.settings, "ingest_auto_enabled", False):
            live_ingest.schedule_incremental_ingest(CHAT_A)
        self.assertEqual(self.rebuild_mock.call_count, 0)
        self.assertEqual(db.commits, 0)

    def test_empty_chat_id_is_safe_noop(self):
        db = _FakeLiveDB()
        db.newest = MSG_1
        self._run(db)
        live_ingest.schedule_incremental_ingest("")
        self.assertEqual(self.rebuild_mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()