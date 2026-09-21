"""Offline tests for Phase 5/6: per-chat /unhinged_on|off slash commands.

The slash-command detection is pure (no DB); the per-chat settings + gating are
exercised through a real in-memory SQLite database holding just the tables they
touch (``chat_settings``, ``trusted_senders``) - neither has a pgvector column,
so they create cleanly under SQLite, unlike the full schema. The WAHA send step
is always mocked; delivery itself is proven live against a real paired chat.
"""

import sys
import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, "/home/nate/promptcraft_chatbot/unipods-bot/backend")

from app.config import settings  # noqa: E402
from app.models import ChatSetting, TrustedSender  # noqa: E402
from app.reply.chat_settings import get_unhinged_enabled, upsert_unhinged  # noqa: E402
from app.reply import reply_worker  # noqa: E402
from app.reply.trigger_detector import (  # noqa: E402
    detect_command,
    resolve_chat_id,
    resolve_sender_id,
)

CONTACT = "15551234567@c.us"
GROUP = "1234567890@g.us"
TRUSTED_JID = "15551230001@c.us"
CASUAL_JID = "15551230002@c.us"


def _dm_payload(body, *, from_me=False, contact=CONTACT):
    return {
        "fromMe": from_me,
        "from": contact,
        "to": None,
        "participant": None,
        "body": body,
        "_data": {},
    }


def _group_payload(body, *, participant=CASUAL_JID, mention_jids=None, from_me=False):
    return {
        "fromMe": from_me,
        "from": GROUP,
        "to": GROUP,
        "participant": participant,
        "body": body,
        "_data": {
            "message": {
                "extendedTextMessage": {
                    "text": body,
                    "contextInfo": {"mentionedJid": list(mention_jids or [])},
                }
            }
        },
    }


def _sqlite_session():
    """Fresh in-memory SQLite with only the tables this feature touches."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ChatSetting.__table__.create(engine)
    TrustedSender.__table__.create(engine)
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# Step 2: pure command detection + shared payload helpers
# ---------------------------------------------------------------------------

class SlashCommandDetectionTests(unittest.TestCase):
    def test_unhinged_on(self):
        self.assertEqual(detect_command(_dm_payload("/unhinged_on")), "unhinged_on")

    def test_unhinged_off(self):
        self.assertEqual(detect_command(_dm_payload("/unhinged_off")), "unhinged_off")

    def test_case_insensitive(self):
        self.assertEqual(detect_command(_dm_payload("/Unhinged_ON")), "unhinged_on")
        self.assertEqual(detect_command(_dm_payload("/uNhInGeD_OfF")), "unhinged_off")

    def test_surrounding_whitespace(self):
        self.assertEqual(detect_command(_dm_payload("  /unhinged_on  ")), "unhinged_on")
        self.assertEqual(detect_command(_dm_payload("\t/unhinged_off\n")), "unhinged_off")

    def test_unknown_or_malformed_is_none(self):
        for body in ["/other", "unhinged_on", "unhinged_off", "/unhinged_on x",
                     "/unhinged off", "please /unhinged_on", "", None]:
            self.assertIsNone(detect_command(_dm_payload(body)), body)

    def test_own_message_ignored(self):
        self.assertIsNone(detect_command(_dm_payload("/unhinged_on", from_me=True)))

    def test_shared_payload_helpers(self):
        self.assertEqual(resolve_chat_id(_group_payload("hi")), GROUP)
        self.assertEqual(resolve_chat_id(_dm_payload("hi")), CONTACT)
        # group author lives in participant, normalized @s.whatsapp.net -> @c.us
        self.assertEqual(
            resolve_sender_id(_group_payload("hi", participant="15551230001@s.whatsapp.net")),
            TRUSTED_JID,
        )
        # 1:1 author is the `from` JID (no participant)
        self.assertEqual(resolve_sender_id(_dm_payload("hi")), CONTACT)


# ---------------------------------------------------------------------------
# Step 1: per-chat settings lookup (real SQLite)
# ---------------------------------------------------------------------------

class UnhingedSettingsTests(unittest.TestCase):
    def setUp(self):
        self.sm = _sqlite_session()
        self._mode = settings.banter_mode_enabled
        self.addCleanup(setattr, settings, "banter_mode_enabled", self._mode)

    def test_no_row_falls_back_to_global_default(self):
        settings.banter_mode_enabled = True
        with self.sm() as db:
            self.assertTrue(get_unhinged_enabled(db, GROUP))
        settings.banter_mode_enabled = False
        with self.sm() as db:
            self.assertFalse(get_unhinged_enabled(db, GROUP))

    def test_override_wins_over_global_default(self):
        with self.sm() as db:
            upsert_unhinged(db, GROUP, False)
        settings.banter_mode_enabled = True  # global on
        with self.sm() as db:
            self.assertFalse(get_unhinged_enabled(db, GROUP))  # chat override off

    def test_read_does_not_create_a_row(self):
        with self.sm() as db:
            self.assertTrue(get_unhinged_enabled(db, GROUP))
            rows = db.query(ChatSetting).filter(ChatSetting.chat_id == GROUP).all()
        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# Step 3: command routing + confirmation (SQLite-backed worker, mocked send)
# ---------------------------------------------------------------------------

class CommandRoutingTests(unittest.TestCase):
    def setUp(self):
        self.sm = _sqlite_session()
        self._mode = settings.banter_mode_enabled
        # The cooldown is module-level and shared across the whole run; earlier
        # tests claim chats, so start each test clean (mirrors test_phase4).
        reply_worker._cooldown.clear()
        self.addCleanup(setattr, settings, "banter_mode_enabled", self._mode)

    def _chat_row(self, chat_id):
        with self.sm() as db:
            return db.query(ChatSetting).filter(ChatSetting.chat_id == chat_id).first()

    @mock.patch("app.reply.reply_worker.send_text")
    def test_unhinged_on_persists_true_and_confirms(self, send):
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(GROUP, _group_payload("/unhinged_on"))  # no @mention
        row = self._chat_row(GROUP)
        self.assertIsNotNone(row)
        self.assertTrue(row.unhinged_enabled)
        send.assert_called_once_with(GROUP, reply_worker.UNHINGED_ON_REPLY)

    @mock.patch("app.reply.reply_worker.send_text")
    def test_unhinged_off_persists_false_and_confirms(self, send):
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(GROUP, _group_payload("/unhinged_off"))
        row = self._chat_row(GROUP)
        self.assertIsNotNone(row)
        self.assertFalse(row.unhinged_enabled)
        send.assert_called_once_with(GROUP, reply_worker.UNHINGED_OFF_REPLY)

    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.classify_intent")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_command_never_reaches_classify_or_answer(self, answer, classify, send):
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(CONTACT, _dm_payload("/unhinged_off"))
        classify.assert_not_called()
        answer.assert_not_called()
        send.assert_called_once_with(CONTACT, reply_worker.UNHINGED_OFF_REPLY)

    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_non_command_message_takes_normal_path(self, answer, send):
        answer.return_value = {"answer_text": "ok", "route": "semantic", "citations": [], "sources": []}
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(CONTACT, _dm_payload("what did we decide"))
        answer.assert_called_once()
        send.assert_called_once_with(CONTACT, "ok")


# ---------------------------------------------------------------------------
# Step 4: the per-chat gate in the reply pipeline
# ---------------------------------------------------------------------------

class UnhingedGateTests(unittest.TestCase):
    def setUp(self):
        self.sm = _sqlite_session()
        self._mode = settings.banter_mode_enabled
        reply_worker._cooldown.clear()
        self.addCleanup(setattr, settings, "banter_mode_enabled", self._mode)

    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.classify_intent")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_override_false_never_enters_banter_even_if_classify_would(self, answer, classify, send):
        # Chat explicitly toggled off, global says on -> knowledge path wins.
        with self.sm() as db:
            upsert_unhinged(db, CONTACT, False)
        settings.banter_mode_enabled = True
        answer.return_value = {"answer_text": "grounded answer", "route": "semantic", "citations": [], "sources": []}
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(CONTACT, _dm_payload("lmaooo"))
        classify.assert_not_called()  # gate short-circuits before the classifier
        answer.assert_called_once()
        send.assert_called_once_with(CONTACT, "grounded answer")

    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker._banter", return_value="haha nice")
    def test_override_true_still_banter_with_global_off(self, banter, send):
        with self.sm() as db:
            upsert_unhinged(db, CONTACT, True)
        settings.banter_mode_enabled = False  # global default off
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(CONTACT, _dm_payload("lmaooo"))
        banter.assert_called_once()
        send.assert_called_once_with(CONTACT, "haha nice")

    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.classify_intent")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_global_default_applies_with_no_row(self, answer, classify, send):
        settings.banter_mode_enabled = False  # no override anywhere
        answer.return_value = {"answer_text": "ok", "route": "semantic", "citations": [], "sources": []}
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(CONTACT, _dm_payload("lmaooo"))
        classify.assert_not_called()
        answer.assert_called_once()
        send.assert_called_once_with(CONTACT, "ok")


# ---------------------------------------------------------------------------
# Step 5: optional lockdown to trusted senders
# ---------------------------------------------------------------------------

class LockdownTests(unittest.TestCase):
    def setUp(self):
        self.sm = _sqlite_session()
        self._mode = settings.banter_mode_enabled
        self._flag = settings.unhinged_toggle_restricted_to_trusted
        reply_worker._cooldown.clear()
        self.addCleanup(setattr, settings, "banter_mode_enabled", self._mode)
        self.addCleanup(setattr, settings, "unhinged_toggle_restricted_to_trusted", self._flag)

    def _chat_row(self, chat_id):
        with self.sm() as db:
            return db.query(ChatSetting).filter(ChatSetting.chat_id == chat_id).first()

    @mock.patch("app.reply.reply_worker.send_text")
    def test_restricted_nontrusted_silently_ignored(self, send):
        settings.unhinged_toggle_restricted_to_trusted = True
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(GROUP, _group_payload("/unhinged_on"))
        send.assert_not_called()  # no reply, and...
        self.assertIsNone(self._chat_row(GROUP))  # ...no state change

    @mock.patch("app.reply.reply_worker.send_text")
    def test_restricted_trusted_works(self, send):
        settings.unhinged_toggle_restricted_to_trusted = True
        with self.sm() as db:
            db.add(TrustedSender(sender_id=TRUSTED_JID))
            db.commit()
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(
                GROUP, _group_payload("/unhinged_on", participant="15551230001@s.whatsapp.net")
            )
        self.assertTrue(self._chat_row(GROUP).unhinged_enabled)
        send.assert_called_once_with(GROUP, reply_worker.UNHINGED_ON_REPLY)

    @mock.patch("app.reply.reply_worker.send_text")
    def test_unrestricted_default_allows_anyone(self, send):
        # flag defaults to False: an unknown sender may toggle (today's behavior)
        with mock.patch("app.reply.reply_worker.SessionLocal", new=self.sm):
            reply_worker.reply_to_captured(GROUP, _group_payload("/unhinged_on"))
        self.assertTrue(self._chat_row(GROUP).unhinged_enabled)
        send.assert_called_once_with(GROUP, reply_worker.UNHINGED_ON_REPLY)


if __name__ == "__main__":
    unittest.main()