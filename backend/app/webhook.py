"""POST /webhook/waha - the receiving end of WAHA's webhook delivery.

Payload shape (verified against current WAHA docs, https://waha.devlike.pro/docs/how-to/events/):

    {
      "id": "evt_...",            # event ULID (not persisted)
      "timestamp": 1741249702485,# event timestamp ms (not persisted)
      "event": "message",
      "session": "default",
      "me": {"id": "...@c.us", "pushName": "..."},
      "payload": {                # <--- the message object
        "id": "true_...@c.us_...",     # our waha_message_id, used for dedup
        "timestamp": 1667561485,       # original message time, unix SECONDS
        "from": "<chat JID>",          # group JID for group msgs, contact JID for 1:1
        "fromMe": false,
        "participant": "<sender JID>", # present for group messages
        "to": "<chat JID>",
        "body": "text",
        "hasMedia": false,
        "media": {"url": "...", "mimetype": "...", "filename": "..."} | null,
        "replyTo": {"id": "...", "body": "..."} | null,
        "_data": {...}                 # engine internals (pushname, type, ...)
      },
      "engine": "NOWEB"
    }

The exact envelope changes between WAHA releases, so this handler is deliberately
defensive: every payload is captured verbatim in messages.raw_payload, and anything
we cannot map is diverted to the messages_unparsed fallback table rather than dropped.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Message, MessageUnparsed
from app.schemas import WebhookResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["waha"])

# Events we know how to process. `message` is the main incoming-message event;
# `message.any` fires for every message creation (including the bot's own) with an
# identical payload shape. All other event types are logged and no-op'd.
MESSAGE_EVENTS = {"message", "message.any"}

KNOWN_TYPES = {"text", "image", "video", "audio", "sticker", "document"}


def _normalize_jid(value: Any) -> str | None:
    """NOWEB reports contacts as 123@s.whatsapp.net; canonicalize those to @c.us."""
    if value is None:
        return None
    as_str = str(value).strip()
    if not as_str:
        return None
    return as_str.replace("@s.whatsapp.net", "@c.us")


def _timestamp_to_utc(value: Any) -> datetime | None:
    """WAHA message timestamps are unix seconds; tolerate floats/strings/None."""
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _infer_msg_type(payload: dict) -> str:
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


def _build_message(body: dict, payload: dict) -> Message:
    """Map a WAHA webhook body to a Message row. Raises on impossible payloads."""
    from_me = bool(payload.get("fromMe", False))
    me = body.get("me") or {}

    # The chat JID is `from` for inbound messages and `to` for outbound ones. NOWEB
    # group messages carry the group in `from` and the individual sender in
    # `participant`.
    if payload.get("chatId"):  # not in current events; kept for forward-compat
        chat_id = _normalize_jid(payload["chatId"])
    elif from_me:
        chat_id = _normalize_jid(payload.get("to") or payload.get("from"))
    else:
        chat_id = _normalize_jid(payload.get("from") or payload.get("to"))
    chat_id = chat_id or ""

    if from_me:
        sender_id = _normalize_jid(
            me.get("id") or payload.get("participant") or payload.get("from")
        )
    else:
        sender_id = _normalize_jid(payload.get("participant") or payload.get("from"))

    sender_data = payload.get("_data") or {}
    sender_name = payload.get("pushName") or sender_data.get("pushName")
    if not sender_name and from_me:
        sender_name = me.get("pushName")

    body_value = payload.get("body")
    if body_value is not None and not isinstance(body_value, str):
        body_value = str(body_value)

    return Message(
        waha_message_id=str(payload["id"]) if payload.get("id") else None,
        session_name=str(body.get("session") or "default"),
        chat_id=chat_id,
        chat_name=None,  # chat/group names are not included in WAHA message events
        is_group=chat_id.endswith("@g.us"),
        sender_id=sender_id,
        sender_name=str(sender_name) if sender_name is not None else None,
        from_me=from_me,
        msg_type=_infer_msg_type(payload),
        body=body_value,
        media_mime=(payload.get("media") or {}).get("mimetype"),
        media_path=None,  # media downloads land in Phase 3
        reply_to_waha_id=(payload.get("replyTo") or {}).get("id"),
        timestamp=_timestamp_to_utc(payload.get("timestamp")),
        raw_payload=body,  # the full original webhook body, for debugging/replay
    )


def _store_unparsed(db: Session, event: str, reason: str, raw: Any) -> None:
    """Best-effort fallback write; never let a capture failure lose data."""
    try:
        db.add(
            MessageUnparsed(
                event=event,
                session_name=None,
                reason=reason,
                raw_payload=raw,
            )
        )
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("Could not persist unparsed message")


@router.post("/webhook/waha", response_model=WebhookResponse, status_code=200)
async def waha_webhook(
    request: Request, db: Session = Depends(get_db)
) -> WebhookResponse:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {"_raw": body}

    event = body.get("event")
    if event not in MESSAGE_EVENTS:
        # Recognized WAHA event but not handled yet (session.status, message.ack,
        # group.v2.*, ...). Log and acknowledge - capture-only in Phase 1.
        logger.info("Ignoring unhandled WAHA event=%s session=%s", event, body.get("session"))
        return WebhookResponse(status="ok")

    payload = body.get("payload")
    if not isinstance(payload, dict):
        _store_unparsed(db, event, "payload_not_an_object", body)
        return WebhookResponse(status="ok")

    waha_message_id = payload.get("id")
    if not waha_message_id:
        # Cannot deduplicate without WAHA's message id; keep the data anyway.
        _store_unparsed(db, event, "missing_payload_id", body)
        return WebhookResponse(status="ok")

    # Idempotency: WAHA retries webhooks, so a message id we already stored is
    # skipped. The unique constraint catches concurrent duplicate delivery too.
    duplicate = db.execute(
        select(Message.id).where(Message.waha_message_id == str(waha_message_id))
    ).first()
    if duplicate:
        return WebhookResponse(status="ok", deduplicated=True)

    try:
        message = _build_message(body, payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to map WAHA message payload; diverting to messages_unparsed")
        _store_unparsed(db, event, f"mapping_error: {exc}", body)
        return WebhookResponse(status="ok")

    db.add(message)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.info("Duplicate message (concurrent delivery), skipped: %s", waha_message_id)
        return WebhookResponse(status="ok", deduplicated=True)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("DB error while storing message; diverting to messages_unparsed")
        _store_unparsed(db, event, f"db_error: {exc}", body)
        return WebhookResponse(status="ok")

    logger.info(
        "Captured message id=%s session=%s chat=%s from_me=%s type=%s",
        waha_message_id,
        body.get("session"),
        message.chat_id,
        message.from_me,
        message.msg_type,
    )
    return WebhookResponse(status="ok", stored=True)