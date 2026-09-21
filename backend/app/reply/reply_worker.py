"""Phase 4 worker: answer an incoming message and reply over WAHA.

Runs as a FastAPI BackgroundTask (after the webhook response is already sent),
so it may take seconds without stalling WAHA's delivery. A fresh DB session is
opened here: the request-scoped session from the webhook handler is closed by
the time background tasks run.
"""

import logging
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import TrustedSender
from app.reply.chat_settings import get_unhinged_enabled, upsert_unhinged
from app.reply.intent_classifier import classify_intent
from app.reply.rate_limit import InMemoryCooldown
from app.reply.sender import ReplySenderError, fetch_session_me, send_text
from app.reply.trigger_detector import (
    Detection,
    detect_command,
    detect_reply,
    resolve_chat_id,
    resolve_sender_id,
)
from app.retrieval.answer import (
    BANTER_FALLBACK,
    _banter_reply,
    answer_question,
    get_recent_context,
)

logger = logging.getLogger(__name__)

# The required clarifying reply when a group @-mentions the bot with no question.
CLARIFY_REPLY = "What's up?"

# Apologetic fallback when the answer pipeline or send fails for any reason.
APOLOGY_REPLY = (
    "Sorry, I hit an internal issue answering that. Try again in a few minutes."
)

# Confirmation replies for the /unhinged_* slash commands - kept short and in
# the bot's normal voice rather than a robotic settings-changed announcement.
UNHINGED_ON_REPLY = "😈 Unhinged mode is now ON for this chat."
UNHINGED_OFF_REPLY = "Unhinged mode is now off. Back to business."

# Per-chat in-memory cooldown (Phase 4; Redis replaces this in Phase 5).
_cooldown = InMemoryCooldown(settings.reply_cooldown_seconds)

# Bot identity discovered from the WAHA session, cached after first use.
# Shape: {"id": "251947711181@c.us", "aliases": {"251947711181", "30727051714790"}}.
# The aliases set holds every digit-string the bot can be @-mentioned under:
# the phone number AND the account's lid (NOWEB surfaces the lid, not the phone,
# in group mentions). Explicit BOT_WHATSAPP_ID config is merged in regardless.
_bot_identity_cache: dict | None = None


def resolve_bot_identity() -> dict:
    """Return {"id": <bot JID>, "aliases": {digits...}} for mention matching."""
    global _bot_identity_cache
    if _bot_identity_cache is not None:
        return _bot_identity_cache

    me = fetch_session_me()
    aliases: set[str] = set()
    if me:
        if me.get("id"):
            aliases.add(_digits(me["id"]))
        if me.get("lid"):
            aliases.add(_digits(me["lid"]))

    configured = settings.bot_whatsapp_id.strip()
    if configured:
        aliases.add(_digits(configured))

    if aliases:
        _bot_identity_cache = {
            "id": (me or {}).get("id") or configured or None,
            "aliases": aliases,
        }
    else:
        logger.warning(
            "No BOT_WHATSAPP_ID and no paired WAHA session; group @-mentions "
            "will be ignored until the session is paired."
        )
        _bot_identity_cache = {"id": None, "aliases": set()}
    return _bot_identity_cache


def _digits(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\D", "", value)


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


def _banter(chat_id: str, question: str) -> str:
    """Casual/banter reply, inside a fresh session.

    Pulls the last few raw messages purely to give Gemini a sense of the room
    (tone/context), then asks for a short in-character reply. ``_banter_reply``
    already degrades to ``BANTER_FALLBACK`` on any failure; a failed context
    fetch also yields the fallback so spam messages never block the bot.
    """
    db = SessionLocal()
    try:
        try:
            recent = get_recent_context(db, chat_id, limit=15)
        except Exception:  # noqa: BLE001
            logger.exception("Banter context fetch failed for chat=%s", chat_id)
            recent = ""
    finally:
        db.close()
    return _banter_reply(question, recent)


def _handle_command(payload: dict, command: str) -> None:
    """Apply a recognized /unhinged_* command: upsert the per-chat setting, reply.

    Runs before ANY mention/DM gating, so a bare command in a group - no
    @-mention of the bot - still works. ``chat_id`` is resolved from the payload
    (the shared trigger_detector helper), a fresh DB session persists the
    override, a confirmation is sent via the normal send path, and the message is
    fully handled - it never falls through to detect_reply/classify_intent.

    Optional lockdown (UNHINGED_TOGGLE_RESTRICTED_TO_TRUSTED=true): the command
    is only honored when the author is in ``trusted_senders``. Otherwise it is
    silently ignored (log only, no state change, no reply) so probing whether the
    command is admin-only doesn't look broken.
    """
    chat_id = resolve_chat_id(payload)
    if not chat_id:
        logger.warning("Unhinged command=%s ignored: cannot resolve chat_id", command)
        return

    enabled = command == "unhinged_on"
    db = SessionLocal()
    try:
        if settings.unhinged_toggle_restricted_to_trusted:
            sender_id = resolve_sender_id(payload)
            trusted = None
            if sender_id:
                trusted = db.execute(
                    select(TrustedSender.sender_id).where(
                        TrustedSender.sender_id == sender_id
                    )
                ).scalars().first()
            if not trusted:
                logger.info(
                    "Unhinged toggle blocked: sender not trusted (command=%s chat=%s sender=%s)",
                    command, chat_id, sender_id,
                )
                return
        upsert_unhinged(db, chat_id, enabled)
    finally:
        db.close()

    _send_best_effort(
        chat_id, UNHINGED_ON_REPLY if enabled else UNHINGED_OFF_REPLY, source="command"
    )


def _handle_detection(detection: Detection) -> None:
    """Reply pipeline for a detection the cooldown already approved."""
    chat_id = detection.chat_id

    if detection.clarify:
        _send_best_effort(chat_id, CLARIFY_REPLY, source="clarify")
        return

    question = detection.question or ""

    # Phase 5 casual mode: non-knowledge detections get a short in-character
    # reply instead of the retrieval pipeline. The gate is now PER-CHAT: a
    # chat_settings.unhinged_enabled override wins over the global
    # settings.banter_mode_enabled default (which is kept as the fallback for
    # chats with no override). When the result is false the classifier is not
    # even consulted, exactly like the old global-off behavior.
    db = SessionLocal()
    try:
        unhinged = get_unhinged_enabled(db, chat_id)
    except Exception:  # noqa: BLE001
        # The settings read is on the hot path of every reply; a DB hiccup must
        # not break replying, so degrade to the global default instead.
        logger.exception("chat_settings read failed for chat=%s; using global default", chat_id)
        unhinged = settings.banter_mode_enabled
    finally:
        db.close()

    if unhinged and classify_intent(question) == "banter":
        try:
            text = _banter(chat_id, question)
        except Exception:  # noqa: BLE001
            logger.exception("Banter reply failed for chat=%s question=%r", chat_id, question[:200])
            text = BANTER_FALLBACK
        _send_best_effort(chat_id, text, source="banter")
        return

    try:
        answer_text = _answer(chat_id, question)
    except Exception:  # noqa: BLE001
        logger.exception("Auto-reply answer failed for chat=%s question=%r", chat_id, question[:200])
        _send_best_effort(chat_id, APOLOGY_REPLY, source="apology")
        return

    _send_best_effort(chat_id, answer_text, source="answer")


def reply_to_captured(chat_id: str, payload: dict) -> None:
    """Top-level worker entry: detect a trigger and (if any) answer it.

    Consumes a slash command FIRST, before any mention/DM gating: a bare
    "/unhinged_on" in a group with no @-mention of the bot is handled here
    entirely (persist the per-chat setting, send confirmation, return). Once a
    command is handled it never falls through to detect_reply or the intent
    classifier.

    Kept deliberately defensive - webhook health must never depend on the reply
    pipeline. Every failure is logged; a genuine attempt failure also produces
    the apologetic fallback message.
    """
    try:
        command = detect_command(payload)
    except Exception:  # noqa: BLE001
        logger.exception("Command detection crashed for chat=%s", chat_id)
        command = None

    if command:
        _handle_command(payload, command)
        return

    try:
        identity = resolve_bot_identity()
        bot_jid = identity.get("id")
        aliases = identity.get("aliases") or set()
        detection = detect_reply(
            payload, bot_jid=bot_jid, bot_aliases=aliases or None
        )
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