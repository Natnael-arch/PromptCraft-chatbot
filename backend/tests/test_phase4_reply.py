"""Offline tests for the Phase 4 live auto-reply pipeline.

Pure/stdlib + httpx MockTransport - no DB, no network, no API keys. The WAHA
endpoint shapes (sendText body, sessions me.id, mention location in _data) were
verified against the running WAHA 2026.8.2 build inside the container.
"""

import sys
import unittest
from unittest import mock

import httpx

sys.path.insert(0, "/home/nate/promptcraft_chatbot/unipods-bot/backend")

from app.config import settings  # noqa: E402
from app.reply import rate_limit, reply_worker, sender, trigger_detector  # noqa: E402
from app.reply.trigger_detector import detect_reply, extract_mentioned_jids  # noqa: E402

BOT_JID = "15550000000@c.us"
BOT_DIGITS = "15550000000"
BOT_LID_DIGITS = "30727051714790"
CONTACT = "15551234567@c.us"
GROUP = "1234567890@g.us"

IDENTITY = {"id": BOT_JID, "aliases": {BOT_DIGITS, BOT_LID_DIGITS}}


def dm_payload(body: str, *, from_me: bool = False, contact: str = CONTACT):
    return {
        "fromMe": from_me,
        "from": contact,
        "to": None,
        "participant": None,
        "body": body,
        "_data": {},
    }


def group_payload(body, *, mention_jids=(f"{BOT_DIGITS}@s.whatsapp.net",), participant="15551234567@s.whatsapp.net"):
    return {
        "fromMe": False,
        "from": GROUP,
        "to": GROUP,
        "participant": participant,
        "body": body,
        "_data": {
            "message": {
                "extendedTextMessage": {
                    "text": body,
                    "contextInfo": {"mentionedJid": list(mention_jids)},
                }
            }
        },
    }


# ---------------------------------------------------------------------------
# trigger_detector
# ---------------------------------------------------------------------------

class TriggerDetectorTests(unittest.TestCase):
    def test_dm_always_triggers(self):
        det = detect_reply(dm_payload("what did we decide last night?"), bot_jid=BOT_JID)
        self.assertTrue(det.should_reply)
        self.assertEqual(det.chat_id, CONTACT)
        self.assertEqual(det.question, "what did we decide last night?")
        self.assertFalse(det.clarify)
        self.assertFalse(det.is_group)

    def test_dm_needs_no_bot_jid(self):
        det = detect_reply(dm_payload("hi"), bot_jid=None)
        self.assertTrue(det.should_reply)

    def test_dm_without_text_body_is_skipped(self):
        self.assertFalse(detect_reply(dm_payload(""), bot_jid=None).should_reply)
        self.assertFalse(detect_reply(dm_payload(None), bot_jid=None).should_reply)

    def test_status_broadcast_is_ignored(self):
        payload = dm_payload("hello", contact="status@broadcast")
        self.assertFalse(detect_reply(payload, bot_jid=BOT_JID).should_reply)

    def test_from_me_never_triggers(self):
        dm = detect_reply(dm_payload("hi", from_me=True), bot_jid=BOT_JID)
        self.assertFalse(dm.should_reply)
        self.assertEqual(dm.reason, "from_me (own message)")
        # fromMe loop guard wins even over a valid group mention
        payload = group_payload("@15550000000 what time?")
        payload["fromMe"] = True
        self.assertFalse(detect_reply(payload, bot_jid=BOT_JID).should_reply)

    def test_group_mention_via_mentioned_jids(self):
        det = detect_reply(
            group_payload("bot @15550000000 what time is the meeting?"),
            bot_jid=BOT_JID,
        )
        self.assertTrue(det.should_reply)
        self.assertEqual(det.chat_id, GROUP)
        self.assertTrue(det.is_group)
        self.assertNotIn("@15550000000", det.question)
        self.assertIn("meeting", det.question)

    def test_group_mention_via_body_fallback_without_mentioned_jids(self):
        payload = group_payload("hey @15550000000 summarise yesterday", mention_jids=())
        det = detect_reply(payload, bot_jid=BOT_JID)
        self.assertTrue(det.should_reply)
        self.assertEqual(det.question, "hey summarise yesterday")

    def test_group_foreign_mention_does_not_trigger(self):
        # regression: any "@<digits>" in the body must NOT count as a bot mention
        payload = group_payload("hello @15559999999 how are you", mention_jids=())
        det = detect_reply(payload, bot_jid=BOT_JID)
        self.assertFalse(det.should_reply)
        self.assertEqual(det.reason, "group message without bot mention")

    def test_group_lid_mention_triggers_via_aliases(self):
        # NOWEB surfaces the account's LID (not the phone) in group mentions
        payload = group_payload("@30727051714790 hello", mention_jids=("30727051714790@lid",))
        det = detect_reply(payload, bot_jid=BOT_JID, bot_aliases={"30727051714790"})
        self.assertTrue(det.should_reply)
        self.assertEqual(det.question, "hello")

    def test_group_lid_body_fallback_triggers_via_aliases(self):
        payload = group_payload("hey @30727051714790 what's up", mention_jids=())
        det = detect_reply(payload, bot_jid=BOT_JID, bot_aliases={"30727051714790"})
        self.assertTrue(det.should_reply)
        self.assertEqual(det.question, "hey what's up")

    def test_group_without_mention_is_ignored(self):
        payload = group_payload("hello everyone", mention_jids=())
        det = detect_reply(payload, bot_jid=BOT_JID)
        self.assertFalse(det.should_reply)
        self.assertEqual(det.reason, "group message without bot mention")

    def test_group_mention_without_question_is_clarify(self):
        payload = group_payload("@15550000000")
        det = detect_reply(payload, bot_jid=BOT_JID)
        self.assertTrue(det.should_reply)
        self.assertTrue(det.clarify)
        self.assertIsNone(det.question)

    def test_group_requires_bot_jid(self):
        payload = group_payload("bot @15550000000 q")
        det = detect_reply(payload, bot_jid=None)
        self.assertFalse(det.should_reply)
        self.assertIn("unknown", det.reason)

    def test_group_detection_via_jid_suffix(self):
        payload = group_payload("x @15550000000", participant=None)
        payload["from"] = GROUP
        payload["to"] = GROUP
        det = detect_reply(payload, bot_jid=BOT_JID)
        self.assertTrue(det.is_group)
        self.assertTrue(det.should_reply)

    def test_strip_mention_removes_all_mentions(self):
        q = trigger_detector.strip_mention("hi @15550000000 @15559999999 what's up", BOT_JID)
        self.assertEqual(q, "hi what's up")

    def test_extract_mentioned_jids_normalizes_to_cus(self):
        payload = group_payload("x")
        jids = extract_mentioned_jids(payload)
        self.assertEqual(jids, [f"{BOT_DIGITS}@c.us"])

    def test_extract_mentioned_jids_handles_missing_data(self):
        self.assertEqual(extract_mentioned_jids({"fromMe": False}), [])
        self.assertEqual(extract_mentioned_jids({"fromMe": False, "_data": None}), [])


# ---------------------------------------------------------------------------
# sender
# ---------------------------------------------------------------------------

class SenderTests(unittest.TestCase):
    def _client_with(self, handler):
        transport = httpx.MockTransport(handler)
        return httpx.Client(base_url="http://waha.test:3000", transport=transport)

    def test_truncate_text_ellipsis(self):
        self.assertEqual(sender.truncate_text("short"), "short")
        self.assertEqual(sender.truncate_text("abcdef", max_chars=3), "ab\u2026")

    def test_send_text_posts_expected_shape(self):
        captured = {}

        def handler(request):
            captured["path"] = request.url.path
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            captured["json"] = request.read().decode()
            return httpx.Response(200, json={"id": "snd_1"})

        with self._client_with(handler) as client:
            result = sender.send_text(CONTACT, "hey there", session="default", client=client)

        self.assertEqual(result, {"id": "snd_1"})
        self.assertEqual(captured["path"], "/api/sendText")
        self.assertNotIn("x-api-key", captured["headers"])  # no key configured -> no header
        body = captured["json"]
        self.assertIn('"chatId":"15551234567@c.us"', body)
        self.assertIn('"session":"default"', body)
        self.assertIn('"text":"hey there"', body)

    def test_headers_include_api_key_when_configured(self):
        with mock.patch.object(reply_worker.settings, "waha_api_key", "k123"):
            self.assertEqual(sender._headers(), {"X-Api-Key": "k123"})
        with mock.patch.object(reply_worker.settings, "waha_api_key", ""):
            self.assertEqual(sender._headers(), {})

    def test_send_text_raises_on_failure(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        with self._client_with(handler) as client:
            with self.assertRaises(sender.ReplySenderError):
                sender.send_text(CONTACT, "x", client=client)

    def test_fetch_session_me_returns_me(self):
        me = {"id": BOT_JID, "lid": "30727051714790@lid", "pushName": "Bot"}

        def handler(request):
            return httpx.Response(200, json=[{"name": "default", "id": "default", "me": me}])

        with self._client_with(handler) as client:
            self.assertEqual(sender.fetch_session_me(session="default", client=client), me)

    def test_fetch_session_me_missing_session(self):
        def handler(request):
            return httpx.Response(200, json=[])

        with self._client_with(handler) as client:
            self.assertIsNone(sender.fetch_session_me(session="default", client=client))

    def test_fetch_session_me_returns_none_on_http_error(self):
        def handler(request):
            return httpx.Response(401, text="nope")

        with self._client_with(handler) as client:
            self.assertIsNone(sender.fetch_session_me(session="default", client=client))

    def test_fetch_session_me_ignores_other_sessions(self):
        def handler(request):
            return httpx.Response(200, json=[{"name": "other", "me": {"id": "999@c.us"}}])

        with self._client_with(handler) as client:
            self.assertIsNone(sender.fetch_session_me(session="default", client=client))


# ---------------------------------------------------------------------------
# rate_limit
# ---------------------------------------------------------------------------

class RateLimitTests(unittest.TestCase):
    def setUp(self):
        self.cooldown = rate_limit.InMemoryCooldown(30.0)

    def test_cooldown_windows_per_key(self):
        self.assertTrue(self.cooldown.allowed("a"))
        self.assertFalse(self.cooldown.allowed("a"))
        self.assertTrue(self.cooldown.allowed("b"))

    def test_allowed_again_after_window(self):
        self.assertTrue(self.cooldown.allowed("a", now=0.0))
        self.assertFalse(self.cooldown.allowed("a", now=10.0))
        self.assertTrue(self.cooldown.allowed("a", now=30.0))


# ---------------------------------------------------------------------------
# reply_worker (orchestration)
# ---------------------------------------------------------------------------

class WorkerTests(unittest.TestCase):
    def setUp(self):
        reply_worker._cooldown.clear()
        # Pin the knowledge path: with BANTER_MODE_ENABLED=false, EVERY detection
        # must take the exact Phase-4 path (this doubles as the kill-switch test,
        # and keeps these pre-existing assertions about _answer deterministic
        # regardless of what the keyword vocabulary looks like). The banter path
        # itself is covered in test_phase4_banter.py.
        self._banter_mode = settings.banter_mode_enabled
        settings.banter_mode_enabled = False

    def tearDown(self):
        settings.banter_mode_enabled = self._banter_mode

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_dm_replies_with_answer(self, answer, send, resolve):
        package = {
            "chat_id": CONTACT,
            "question": "what's up",
            "answer_text": "Here's the answer.",
            "route": "semantic",
            "citations": [],
            "sources": [],
        }
        answer.return_value = package
        reply_worker.reply_to_captured(CONTACT, dm_payload("what's up"))
        answer.assert_called_once()
        send.assert_called_once_with(CONTACT, "Here's the answer.")

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_group_mention_stripped_before_answering(self, answer, send, resolve):
        answer.return_value = {"answer_text": "ok", "route": "semantic", "citations": [], "sources": []}
        payload = group_payload("bot @15550000000 what did we miss")
        reply_worker.reply_to_captured(GROUP, payload)
        _, _, question = answer.call_args.args
        self.assertEqual(question, "bot what did we miss")

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_clarify_skips_answering(self, answer, send, resolve):
        reply_worker.reply_to_captured(GROUP, group_payload("@15550000000"))
        answer.assert_not_called()
        send.assert_called_once_with(GROUP, reply_worker.CLARIFY_REPLY)

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_from_me_never_replies(self, answer, send, resolve):
        payload = dm_payload("hi", from_me=True)
        reply_worker.reply_to_captured(CONTACT, payload)
        answer.assert_not_called()
        send.assert_not_called()

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_cooldown_suppresses_rapid_followup(self, answer, send, resolve):
        answer.return_value = {"answer_text": "a", "route": "semantic", "citations": [], "sources": []}
        payload = dm_payload("first")
        reply_worker.reply_to_captured(CONTACT, payload)
        reply_worker.reply_to_captured(CONTACT, dm_payload("second"))
        self.assertEqual(send.call_count, 1)

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_answer_failure_sends_apology(self, answer, send, resolve):
        answer.side_effect = RuntimeError("boom")
        reply_worker.reply_to_captured(CONTACT, dm_payload("what happened?"))
        send.assert_called_once_with(CONTACT, reply_worker.APOLOGY_REPLY)

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    def test_send_failure_does_not_escape_worker(self, answer, send, resolve):
        answer.return_value = {"answer_text": "x", "route": "semantic", "citations": [], "sources": []}
        send.side_effect = sender.ReplySenderError("wa")
        reply_worker.reply_to_captured(CONTACT, dm_payload("q"))  # must not raise


if __name__ == "__main__":
    unittest.main()