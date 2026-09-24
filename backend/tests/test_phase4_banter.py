"""Offline tests for Phase 5: trusted-sender boost/citations + casual/banter mode.

No DB, no network, no API keys - SQL orchestration is exercised through small
fake ``db.execute`` shims (the pgvector schema makes sqlite create_all
impossible, same constraint as the other phase-4/5 test files), and the Gemini
call is either patched away or skipped via the no-key fallback.
"""

import sys
import unittest
from unittest import mock
from uuid import uuid4

sys.path.insert(0, "/home/nate/promptcraft_chatbot/unipods-bot/backend")

from app.config import settings  # noqa: E402
from app.models import Chunk, Message, TrustedSender  # noqa: E402
from app.retrieval import answer, search  # noqa: E402
from app.reply.intent_classifier import BANTER, KNOWLEDGE, classify_intent  # noqa: E402

BOT_JID = "15550000000@c.us"
BOT_LID_DIGITS = "30727051714790"
CONTACT = "15551234567@c.us"
GROUP = "1234567890@g.us"

IDENTITY = {"id": BOT_JID, "aliases": {"15550000000", BOT_LID_DIGITS}}

ANNOUNCER_JID = "15551230001@c.us"
ANNOUNCER_NAME = "Amina"
MENTIONER_JID = "15551230002@c.us"
MENTIONER_NAME = "Dave"


def _message(body, *, sender_id=MENTIONER_JID, sender_name=MENTIONER_NAME, chat_id=GROUP):
    return Message(
        id=uuid4(),
        chat_id=chat_id,
        chat_name="The Cohort",
        sender_id=sender_id,
        sender_name=sender_name,
        from_me=False,
        msg_type="text",
        body=body,
        timestamp=None,
    )


def _chunk(message_ids, *, content="some content", session_id=None):
    return Chunk(
        id=uuid4(),
        session_id=session_id,
        recording_id=None,
        message_ids=[str(m_id) for m_id in message_ids],
        source_type="text",
        voice_segments=[],
        content=content,
    )


# ---------------------------------------------------------------------------
# Fake db: dispatches executed statements to crafted rows by table/column shape.
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeDB:
    def __init__(self, *, chunks=None, messages=None, trusted=None):
        self.chunks = list(chunks or [])
        self.messages = list(messages or [])
        self.trusted = list(trusted or [])

    def execute(self, stmt):
        s = str(stmt)
        if "trusted_senders" in s:
            # Column-targeted (sender_id, weight) lookup vs full entity select -
            # distinguishable by the presence of the display_name column.
            if "trusted_senders.weight" in s and "trusted_senders.display_name" not in s:
                return _FakeResult([(t.sender_id, t.weight) for t in self.trusted])
            return _FakeResult(self.trusted)
        if "chunks" in s:
            return _FakeResult(self.chunks)
        if "messages" in s:
            # Full-entity selects (Message) vs the (id, sender_id) tuple lookup in
            # search._apply_trust_boost - distinguishable by column count in SQL.
            if "sender_id" in s and "waha_message_id" not in s:
                return _FakeResult([(m.id, m.sender_id) for m in self.messages])
            return _FakeResult(self.messages)
        raise AssertionError(f"unhandled statement: {s}")


def _dm_payload(body: str, *, from_me: bool = False, contact: str = CONTACT):
    return {
        "fromMe": from_me,
        "from": contact,
        "to": None,
        "participant": None,
        "body": body,
        "_data": {},
    }


# ---------------------------------------------------------------------------
# Part 1: trusted-sender ranking boost
# ---------------------------------------------------------------------------

class TrustBoostTests(unittest.TestCase):
    def test_lower_relevance_trusted_chunk_outranks_untrusted(self):
        ranked = [("chunk_b", 0.9), ("chunk_a", 0.2)]  # B topically closer
        chunk_senders = {"chunk_b": {MENTIONER_JID}, "chunk_a": {ANNOUNCER_JID}}
        weights = {ANNOUNCER_JID: 5.0}
        out = search._boost_scores(ranked, chunk_senders, weights)
        self.assertEqual([cid for cid, _ in out], ["chunk_a", "chunk_b"])
        # scores: a = 0.2*5 = 1.0 > b = 0.9
        self.assertAlmostEqual(dict(out)["chunk_a"], 1.0)

    def test_weight_one_leaves_ranking_unchanged(self):
        ranked = [("chunk_b", 0.9), ("chunk_a", 0.2)]
        chunk_senders = {"chunk_b": {MENTIONER_JID}, "chunk_a": {ANNOUNCER_JID}}
        weights = {ANNOUNCER_JID: 1.0}
        out = search._boost_scores(ranked, chunk_senders, weights)
        self.assertEqual([cid for cid, _ in out], ["chunk_b", "chunk_a"])
        self.assertEqual([s for _, s in out], [0.9, 0.2])

    def test_no_weights_returns_unchanged(self):
        ranked = [("chunk_b", 0.9), ("chunk_a", 0.2)]
        out = search._boost_scores(ranked, {"chunk_a": {ANNOUNCER_JID}}, {})
        self.assertEqual(out, ranked)

    def test_apply_boost_wires_messages_to_senders_to_weights(self):
        ann_msg = _message("The deadline is Friday!", sender_id=ANNOUNCER_JID, sender_name=ANNOUNCER_NAME)
        ban_msg = _message("chill vibes only")
        chunk_a = _chunk([ann_msg.id], content="The deadline is Friday!")
        chunk_b = _chunk([ban_msg.id], content="chill vibes only")
        db = _FakeDB(
            chunks=[chunk_a, chunk_b],
            messages=[ann_msg, ban_msg],
            trusted=[TrustedSender(sender_id=ANNOUNCER_JID, display_name=ANNOUNCER_NAME, weight=3.0)],
        )
        ranked = [("b", 0.8), ("a", 0.3)]
        ranked = [(str(chunk_b.id), 0.8), (str(chunk_a.id), 0.3)]
        out = search._apply_trust_boost(db, ranked)
        self.assertEqual([cid for cid, _ in out], [str(chunk_a.id), str(chunk_b.id)])

    def test_apply_boost_skips_when_nothing_trusted(self):
        msg = _message("hello")
        chunk = _chunk([msg.id])
        db = _FakeDB(chunks=[chunk], messages=[msg], trusted=[])
        ranked = [(str(chunk.id), 0.8)]
        self.assertEqual(search._apply_trust_boost(db, ranked), ranked)


# ---------------------------------------------------------------------------
# Part 1b: citations carry the announcement flag for trusted senders
# ---------------------------------------------------------------------------

class TrustCitationTests(unittest.TestCase):
    def test_semantic_citation_flags_trusted_sender(self):
        ann = _message("Deadline is Friday at 5pm", sender_id=ANNOUNCER_JID, sender_name=ANNOUNCER_NAME)
        chunk = _chunk([ann.id], content=ann.body)
        db = _FakeDB(
            chunks=[chunk],
            messages=[ann],
            trusted=[TrustedSender(sender_id=ANNOUNCER_JID, display_name=ANNOUNCER_NAME, role_label="announcer", weight=2.0)],
        )
        with mock.patch("app.retrieval.answer.hybrid_search", return_value=[(str(chunk.id), 0.5)]):
            _, citations, _ = answer._answer_semantic(db, GROUP, "what is the deadline?", [])

        self.assertEqual(len(citations), 1)
        c = citations[0]
        self.assertTrue(c["is_announcement"])
        self.assertEqual(c["role_label"], "announcer")
        self.assertEqual(c["sender_id"], ANNOUNCER_JID)
        self.assertIn("📢 [Amina, announcer]", _semantic_answer(db, chunk, ann))

    def test_untrusted_citation_stays_unflagged(self):
        msg = _message("chill vibes only")
        chunk = _chunk([msg.id], content=msg.body)
        db = _FakeDB(chunks=[chunk], messages=[msg], trusted=[])
        with mock.patch("app.retrieval.answer.hybrid_search", return_value=[(str(chunk.id), 0.5)]):
            _, citations, _ = answer._answer_semantic(db, GROUP, "anything?", [])

        self.assertEqual(len(citations), 1)
        self.assertFalse(citations[0]["is_announcement"])
        self.assertIsNone(citations[0]["role_label"])
        self.assertNotIn("📢", citations[0]["preview"])


def _semantic_answer(db, chunk, ann):
    """Rerun _answer_semantic and hand back its rendered answer text."""
    with mock.patch("app.retrieval.answer.hybrid_search", return_value=[(str(chunk.id), 0.5)]):
        text, _, _ = answer._answer_semantic(db, GROUP, "what is the deadline?", [])
    return text


# ---------------------------------------------------------------------------
# Part 2: intent classification
# ---------------------------------------------------------------------------

class IntentClassifierTests(unittest.TestCase):
    def test_program_vocabulary_is_knowledge(self):
        for q in [
            "what's the deadline?",
            "when do we submit?",
            "who's judging?",
            "did you see the announcement?",
        ]:
            self.assertEqual(classify_intent(q), KNOWLEDGE, q)

    def test_interrogative_frames_are_knowledge(self):
        for q in [
            "do you respond only to @15550000000",
            "what makes you different from other chatbots?",
            "what can you find in the group",
            "how does this work?",
            "is there a plan for tomorrow?",
        ]:
            self.assertEqual(classify_intent(q), KNOWLEDGE, q)

    def test_pure_banter_is_banter(self):
        for q in ["lmaooo", "that's wild lol", "nice one", "ruok", "😂😂😂"]:
            self.assertEqual(classify_intent(q), BANTER, q)

    def test_greetings_and_small_talk_are_banter(self):
        for q in [
            "how are you?",
            "how are you",
            "hi",
            "hey",
            "what's up",
            "what's up?",
            "how is it going?",
            "good morning",
            "hello",
        ]:
            self.assertEqual(classify_intent(q), BANTER, q)

    def test_keyword_vocabulary_is_configurable(self):
        kept = settings.knowledge_intent_keywords
        try:
            settings.knowledge_intent_keywords = ["submission"]
            self.assertEqual(classify_intent("drop the submission link"), KNOWLEDGE)
            self.assertEqual(classify_intent("when is the deadline?"), KNOWLEDGE)  # interrogative
            self.assertEqual(classify_intent("no way"), BANTER)
        finally:
            settings.knowledge_intent_keywords = kept

    def test_ambiguous_defaults_to_knowledge(self):
        # Unclear phrasing should be answered (try-hard) rather than joked away.
        self.assertEqual(classify_intent("you know what i mean"), KNOWLEDGE)


# ---------------------------------------------------------------------------
# Part 2b: banter reply + recency context + worker routing
# ---------------------------------------------------------------------------

class BanterReplyTests(unittest.TestCase):
    def tearDown(self):
        settings.gemini_api_key = ""
        settings.answer_provider = "extractive"
        settings.banter_mode_enabled = True

    def test_banter_falls_back_without_key(self):
        settings.gemini_api_key = ""
        self.assertEqual(answer._banter_reply("lol what", ""), "lol")

    @mock.patch("google.genai.Client")
    def test_banter_falls_back_on_max_tokens_truncation(self, mock_client_cls):
        settings.gemini_api_key = "fake-key"
        mock_cand = mock.MagicMock(finish_reason="MAX_TOKENS")
        mock_resp = mock.MagicMock(candidates=[mock_cand], text="Half finished response...")
        mock_client = mock.MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        reply = answer._banter_reply("how are you?", "")
        self.assertEqual(reply, "lol")

    @mock.patch("google.genai.Client", side_effect=RuntimeError("boom"))
    def test_banter_falls_back_on_api_failure(self, _client):
        settings.gemini_api_key = "fake-key"
        self.assertEqual(answer._banter_reply("lol what", "chat context"), "lol")

    def test_get_recent_context_is_chronological_and_raw(self):
        older = _message("first line", sender_name="Amina", sender_id=ANNOUNCER_JID)
        newer = _message("second line", sender_name="Dave", sender_id=MENTIONER_JID)
        # created_at DESC like the query - reversed back to chronological by the fn
        db = _FakeDB(messages=[newer, older])
        ctx = answer.get_recent_context(db, GROUP, limit=15)
        self.assertEqual(ctx, "Amina: first line\nDave: second line")


class GeminiSynthesisTruncationTests(unittest.TestCase):
    def tearDown(self):
        settings.gemini_api_key = ""
        settings.answer_provider = "extractive"

    @mock.patch("google.genai.Client")
    def test_gemini_synthesize_retries_on_max_tokens_and_falls_back_if_still_truncated(self, mock_client_cls):
        settings.gemini_api_key = "fake-key"
        mock_cand = mock.MagicMock(finish_reason="MAX_TOKENS")
        mock_resp = mock.MagicMock(candidates=[mock_cand], text="Truncated list...")
        mock_client = mock.MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        extractive_text = "Full extractive answer list"
        res = answer._gemini_synthesize("What dates are mentioned?", extractive_text)

        # Should fall back to extractive text rather than returning cut-off text
        self.assertEqual(res, extractive_text)
        # Verify it retried with the concise reframing prompt (2 calls total)
        self.assertEqual(mock_client.models.generate_content.call_count, 2)


class BanterRoutingTests(unittest.TestCase):
    def setUp(self):
        from app.reply import reply_worker
        self.reply_worker = reply_worker
        reply_worker._cooldown.clear()
        self._banter_mode = settings.banter_mode_enabled

    def tearDown(self):
        settings.banter_mode_enabled = self._banter_mode

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker._banter", return_value="haha nice one")
    def test_banter_message_gets_casual_reply_no_retrieval(self, banter, send, resolve):
        self.reply_worker.reply_to_captured(CONTACT, _dm_payload("lmaooo"))
        banter.assert_called_once()
        send.assert_called_once_with(CONTACT, "haha nice one")

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker.answer_question")
    @mock.patch("app.reply.reply_worker.classify_intent")
    def test_kill_switch_routes_knowledge_path_without_classifying(self, classify, answer, send, resolve):
        settings.banter_mode_enabled = False
        answer.return_value = {"answer_text": "solid answer", "route": "semantic", "citations": [], "sources": []}
        self.reply_worker.reply_to_captured(CONTACT, _dm_payload("lmaooo"))
        classify.assert_not_called()  # classifier never consulted when disabled
        answer.assert_called_once()
        send.assert_called_once_with(CONTACT, "solid answer")

    @mock.patch("app.reply.reply_worker.resolve_bot_identity", return_value=IDENTITY)
    @mock.patch("app.reply.reply_worker.send_text")
    @mock.patch("app.reply.reply_worker._banter", side_effect=RuntimeError("boom"))
    def test_banter_pipeline_failure_sends_fallback(self, banter, send, resolve):
        # A crashed banter path must contain: send the safe non-apology fallback
        # ("lol") instead of escalating to APOLOGY_REPLY or escaping the worker.
        self.reply_worker.reply_to_captured(CONTACT, _dm_payload("lmaooo"))
        send.assert_called_once_with(CONTACT, "lol")


if __name__ == "__main__":
    unittest.main()