"""Shared WAHA message-object -> Message row mapping.

WAHA delivers the same message object in two places:

* the live webhook envelope's ``payload`` (app/webhook.py), and
* the NOWEB store's ``GET /api/{session}/chats/{chatId}/messages`` list
  (app/ingest/waha_store_backfill.py).

This module is the single source of truth for turning that object into a
``messages`` row, so backfilled store history is indistinguishable from rows
captured live via webhook. The webhook envelope's extra context (``session``
name and the ``me`` "who am I" dict) is passed as optional call arguments here,
so the store path can inject the same context without an envelope.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from app.models import Message

logger = logging.getLogger(__name__)

# Events we know how to process. `message` is the main incoming-message event;
# `message.any` fires for every message creation (including the bot's own). The
# mapper itself is event-agnostic; the webhook handler does the routing.
KNOWN_TYPES = {"text", "image", "video", "audio", "sticker", "document"}


def normalize_jid(value: Any) -> str | None:
    """NOWEB reports contacts as 123@s.whatsapp.net; canonicalize those to @c.us."""
    if value is None:
        return None
    as_str = str(value).strip()
    if not as_str:
        return None
    return as_str.replace("@s.whatsapp.net", "@c.us")


def timestamp_to_utc(value: Any) -> datetime | None:
    """WAHA message timestamps are unix seconds; tolerate floats/strings/None."""
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def infer_msg_type(payload: dict) -> str:
    """Best-effort text/image/video/audio/document/sticker/other inference.

    Current WAHA releases omit the top-level `type` field, so we infer from the media
    mime type first, then engine hints in `_data`, then the presence of text.
    """
    media = payload.get("media") or {}
    mime = str(media.get("mimetype") or "").lower()
    if mime:
        if mime == "image/webp":
            return "sticker"
        leading = mime.split("/", 1)[0]
        if leading in {"image", "video", "audio", "text"}:
            return leading
        return "document"

    raw_type = str(payload.get("type") or "").lower()
    if raw_type == "chat":  # WAHA's internal name for a plain text message
        return "text"
    if raw_type in KNOWN_TYPES:
        return raw_type

    data_type = str((payload.get("_data") or {}).get("type") or "").lower()
    if data_type == "ptt":  # NOWEB pushes voice notes as type "ptt"
        return "audio"
    if data_type in KNOWN_TYPES:
        return data_type

    body = payload.get("body")
    if body is not None and str(body).strip():
        return "text"
    return "other"


def build_message(
    payload: dict,
    *,
    session_name: str = "default",
    me: dict | None = None,
    raw_payload: Any = None,
) -> Message:
    """Map a WAHA message object to a Message row. Raises on impossible payloads.

    ``payload`` is the message object in either delivery context (webhook event
    ``payload`` or a store list item). ``me`` is the WAHA session identity dict
    (``{"id": ..., "pushName": ...}``) used to attribute bot-sent rows; the
    store path passes a synthesized one from ``settings.bot_whatsapp_id``.
    ``raw_payload`` is stored verbatim (webhook: the full envelope) for
    debugging/replay; it defaults to the payload itself.
    """
    me = me or {}
    from_me = bool(payload.get("fromMe", False))

    # The chat JID is `from` for inbound messages and `to` for outbound ones. NOWEB
    # group messages carry the group in `from` and the individual sender in
    # `participant`. Store messages only carry `from`, so `to` is a webhook-only
    # extra that the fallback simply never sees.
    if payload.get("chatId"):  # not in current events; kept for forward-compat
        chat_id = normalize_jid(payload["chatId"])
    elif from_me:
        chat_id = normalize_jid(payload.get("to") or payload.get("from"))
    else:
        chat_id = normalize_jid(payload.get("from") or payload.get("to"))
    chat_id = chat_id or ""

    if from_me:
        sender_id = normalize_jid(
            me.get("id") or payload.get("participant") or payload.get("from")
        )
    else:
        sender_id = normalize_jid(payload.get("participant") or payload.get("from"))

    sender_data = payload.get("_data") or {}
    sender_name = payload.get("pushName") or sender_data.get("pushName")
    if not sender_name and from_me:
        sender_name = me.get("pushName")

    body_value = payload.get("body")
    if body_value is not None and not isinstance(body_value, str):
        body_value = str(body_value)

    return Message(
        waha_message_id=str(payload["id"]) if payload.get("id") else None,
        session_name=session_name,
        chat_id=chat_id,
        chat_name=None,  # chat/group names are not included in WAHA message events
        is_group=chat_id.endswith("@g.us"),
        sender_id=sender_id,
        sender_name=str(sender_name) if sender_name is not None else None,
        from_me=from_me,
        msg_type=infer_msg_type(payload),
        body=body_value,
        media_mime=(payload.get("media") or {}).get("mimetype"),
        media_path=None,  # media downloads land in Phase 3
        reply_to_waha_id=(payload.get("replyTo") or {}).get("id"),
        timestamp=timestamp_to_utc(payload.get("timestamp")),
        raw_payload=raw_payload if raw_payload is not None else payload,
    )