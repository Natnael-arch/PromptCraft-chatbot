"""Phase 4 worker: answer an incoming message and reply over WAHA.

Runs as a FastAPI BackgroundTask (after the webhook response is already sent),
so it may take seconds without stalling WAHA's delivery. A fresh DB session is
opened here: the request-scoped session from the webhook handler is closed by
the time background tasks run.
"""

import logging

from app.config import settings
from app.db import SessionLocal
from app.retrieval.answer import answer_question
from app.reply.rate_limit import InMemoryCooldown
from app.reply.sender import ReplySenderError, fetch_session_me_id, send_text
from app.reply.trigger_detector import Detection, detect_reply

logger = logging.getLogger(__name__)

# The required clarifying reply when a group @-mentions the bot with no question.
CLARIFY_REPLY = "What's up?"

# Apologetic fallback when the answer pipeline or send fails for any reason.
APOLOGY_REPLY = (
    "Sorry, I hit an internal issue answering that. Try again in a few minutes."
)

# Per-chat in-memory cooldown (Phase 4; Redis replaces this in Phase 5).
_cooldown = InMemoryCooldown(settings.reply_cooldown_seconds)

# Bot JID discovered from WAHA session info; cached so we don't hit /api/sessions
# for every message. Explicit BOT_WHATSAPP_ID env config always wins.
_bot_jid_cache: str | None = None


def resolve_bot_jid() -> str | None:
    """Return the bot's own WhatsApp JID, preferring the explicit config."""
    global _bot_jid_cache
    if settings.bot_whatsapp_id.strip():
        return settings.bot_whatsapp_id.strip()
    if _bot_jid_cache is None:
        _bot_jid_cache = fetch_session_me_id()
        if _bot_jid_cache:
            logger.info("Discovered bot JID from WAHA session info: %s", _bot_jid_cache)
        else:
            logger.warning(
                "No BOT_WHATSAPP_ID and no paired WAHA session; group @-mentions "
                "will be ignored until the session is paired."
            )
    return _bot_jid_cache


def _send_best_effort(chat_id: str, text: str, *, source: str) -> None:
    """Try to deliver a reply; a failed send is logged, never raised further."""
    try:
        send_text(chat_id, text)
        logger.info("Replied (%s) to chat=%s", source, chat_id)
    except ReplySenderError:
        logger.exception("Message was not sent (reply source=%s, chat=%s)", source, chat_id)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected failure sending reply (source=%s, chat=%s)", source, chat_id)


def _answer(chat_id: str, question: str) -> str:
    """Run the same retrieval+answer pipeline /ask uses, inside a fresh session."""
    db = SessionLocal()
    try:
        result = answer_question(db, chat_id, question, embedding_provider=None)
        return result["answer_text"]
    finally:
        db.close()


def _handle_detection(detection: Detection) -> None:
    """Reply pipeline for a detection the cooldown already approved."""
    chat_id = detection.chat_id

    if detection.clarify:
        _send_best_effort(chat_id, CLARIFY_REPLY, source="clarify")
        return

    question = detection.question or ""
    try:
        answer_text = _answer(chat_id, question)
    except Exception:  # noqa: BLE001
        logger.exception("Auto-reply answer failed for chat=%s question=%r", chat_id, question[:200])
        _send_best_effort(chat_id, APOLOGY_REPLY, source="apology")
        return

    _send_best_effort(chat_id, answer_text, source="answer")


def reply_to_captured(chat_id: str, payload: dict) -> None:
    """Top-level worker entry: detect a trigger and (if any) answer it.

    Kept deliberately defensive - webhook health must never depend on the reply
    pipeline. Every failure is logged; a genuine attempt failure also produces
    the apologetic fallback message.
    """
    try:
        bot_jid = resolve_bot_jid()
        detection = detect_reply(payload, bot_jid=bot_jid)
    except Exception:  # noqa: BLE001
        logger.exception("Auto-reply trigger detection crashed for chat=%s", chat_id)
        return

    if not detection.should_reply:
        logger.info(
            "No auto-reply (reason=%s) chat=%s group=%s",
            detection.reason, chat_id, detection.is_group,
        )
        return

    if not _cooldown.allowed(str(detection.chat_id)):
        logger.info("Auto-reply rate-limited for chat=%s", detection.chat_id)
        return

    _handle_detection(detection)