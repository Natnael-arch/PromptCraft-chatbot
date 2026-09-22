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
                                       # group @-mentions live here, in
                                       # _data.message.extendedTextMessage.contextInfo.mentionedJid
      },
      "engine": "NOWEB"
    }

The exact envelope changes between WAHA releases, so this handler is deliberately
defensive: every payload is captured verbatim in messages.raw_payload, and anything
we cannot map is diverted to the messages_unparsed fallback table rather than dropped.
"""

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingest.live_ingest import schedule_incremental_ingest
from app.ingest.waha_mapper import build_message
from app.models import Message, MessageUnparsed
from app.reply.reply_worker import reply_to_captured
from app.schemas import WebhookResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["waha"])

# Events we know how to process. `message` is the main incoming-message event;
# `message.any` fires for every message creation (including the bot's own) with an
# identical payload shape. All other event types are logged and no-op'd.
MESSAGE_EVENTS = {"message", "message.any"}


def _build_message(body: dict, payload: dict) -> Message:
    """Map a WAHA webhook body to a Message row (shared mapper, webhook context).

    The payload -> Message logic lives in app/ingest/waha_mapper.build_message so
    the store-history backfill reuses the exact same mapping; only the envelope
    context (session name, the `me` identity dict, and the raw envelope body) is
    supplied here.
    """
    return build_message(
        payload,
        session_name=str(body.get("session") or "default"),
        me=body.get("me") or None,
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
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
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

    # Phase 4: schedule the auto-reply worker as a background task so WAHA's
    # webhook delivery is acknowledged instantly. The worker runs after this
    # response is sent, opens its own DB session, and answers via the same
    # answer_question pipeline /ask uses (no self-HTTP round-trip). Any failure
    # inside it is logged and does not affect this response.
    background_tasks.add_task(reply_to_captured, message.chat_id, payload)
    # Phase 4: keep this chat's sessions/chunks searchable. Debounced, serialized
    # and cursor-skipping (app/ingest/live_ingest.py), so a busy group rebuilds
    # at most once per ingest_debounce_seconds and never re-embeds unchanged
    # history. Also a background task: the webhook ack stays instant.
    background_tasks.add_task(schedule_incremental_ingest, message.chat_id)
    return WebhookResponse(status="ok", stored=True)