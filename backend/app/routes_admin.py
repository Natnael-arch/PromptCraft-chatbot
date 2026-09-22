"""Admin routes.

POST/DELETE/GET on ``/admin/trusted-senders`` manage the senders whose content
is weighted higher in retrieval (``weight``) and flagged in citations
(``role_label``/``display_name``). POST ``/admin/store-backfill/{chat_id}``
materializes WAHA's NOWEB store history for a chat into ``messages`` (messages
that arrived before webhook capture was listening, including the bot's own
replies). All endpoints are gated on the ``X-Admin-Token`` header matching
``ADMIN_TOKEN``.

Auth note: this is a hackathon-grade stopgap - one shared static token read from
env, compared with ``secrets.compare_digest``, no rate-limiting and no roles. It
protects the trusted-sender list (whose contents bias what the bot surfaces),
not application credentials; replace with a real auth layer before this touches
anything sensitive.
"""

import logging
import secrets

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.ingest.embedder import EmbeddingError
from app.ingest.waha_store_backfill import backfill_chat_from_store
from app.models import TrustedSender
from app.schemas import (
    AdminChangeResponse,
    StoreBackfillResponse,
    TrustedSenderCreate,
    TrustedSenderRead,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(x_admin_token: str = Header(default="", alias="X-Admin-Token")) -> None:
    """Dependency: reject the call unless the header matches ADMIN_TOKEN.

    An unset ADMIN_TOKEN means the endpoint is unusable rather than open (SAFER
    by default: turning the list on by accident is worse than a 401).
    """
    expected = settings.admin_token.strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Admin routes not configured (ADMIN_TOKEN unset)")
    if not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token")


@router.get("/trusted-senders", response_model=list[TrustedSenderRead])
def list_trusted(
    db: Session = Depends(get_db),
    _: None = Depends(_require_admin),
) -> list[TrustedSender]:
    rows = db.execute(
        select(TrustedSender).order_by(TrustedSender.added_at.desc())
    ).scalars().all()
    return list(rows)


@router.post("/trusted-senders", response_model=AdminChangeResponse)
def add_trusted(
    payload: TrustedSenderCreate,
    db: Session = Depends(get_db),
    _: None = Depends(_require_admin),
) -> AdminChangeResponse:
    """Insert, or update when ``sender_id`` already exists (idempotent upsert).

    ``display_name``/``role_label``/``weight`` fall back to the table defaults
    when omitted (``role_label='announcer'``, ``weight=2.0``).
    """
    row = db.execute(
        select(TrustedSender).where(TrustedSender.sender_id == payload.sender_id)
    ).scalars().first()

    if row is None:
        row = TrustedSender(
            sender_id=payload.sender_id,
            display_name=payload.display_name,
            role_label=payload.role_label,
            weight=payload.weight,
        )
        db.add(row)
        action = "added"
    else:
        # Only overwrite the fields the caller actually sent, keeping the stored
        # role_label/weight for partial updates.
        if payload.display_name is not None:
            row.display_name = payload.display_name
        if payload.role_label is not None:
            row.role_label = payload.role_label
        if payload.weight is not None:
            row.weight = payload.weight
        action = "updated"

    db.commit()
    logger.info("trusted_senders %s sender_id=%s", action, payload.sender_id)
    return AdminChangeResponse(ok=True, sender_id=payload.sender_id, action=action)


@router.delete("/trusted-senders/{sender_id}", response_model=AdminChangeResponse)
def remove_trusted(
    sender_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(_require_admin),
) -> AdminChangeResponse:
    row = db.execute(
        select(TrustedSender).where(TrustedSender.sender_id == sender_id)
    ).scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No trusted sender {sender_id!r}")

    db.delete(row)
    db.commit()
    logger.info("trusted_senders removed sender_id=%s", sender_id)
    return AdminChangeResponse(ok=True, sender_id=sender_id, action="removed")


@router.post("/store-backfill/{chat_id}", response_model=StoreBackfillResponse)
def store_backfill(
    chat_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(_require_admin),
) -> StoreBackfillResponse:
    """Materialize WAHA's NOWEB store history for a chat into `messages`.

    NOWEB's store keeps messages that predate (or occurred between) webhook
    captures - including the bot's own ``true_...`` sent messages that the
    webhook path never saw. Fetches the store for ``chat_id``, inserts whatever
    is missing (same mapping as the webhook, ``fromMe=true`` included), rebuilds
    the chat's sessions/chunks, and commits. Idempotent: re-running inserts
    nothing new.
    """
    session_name = settings.waha_session
    try:
        stats = backfill_chat_from_store(db, session_name, chat_id)
        db.commit()
    except httpx.HTTPError as exc:
        db.rollback()
        logger.warning("store_backfill: WAHA read failed chat=%s", chat_id)
        raise HTTPException(
            status_code=502, detail=f"WAHA store read failed: {exc}"
        ) from exc
    except EmbeddingError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc
    except IntegrityError as exc:
        db.rollback()
        logger.exception("store_backfill integrity error chat=%s", chat_id)
        raise HTTPException(status_code=409, detail="Backfill conflicted with existing rows") from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("store_backfill failed chat=%s", chat_id)
        raise HTTPException(status_code=500, detail=f"Store backfill failed: {exc}") from exc

    logger.info("store_backfill chat=%s %s", chat_id, stats)
    return StoreBackfillResponse(
        chat_id=chat_id, session_name=session_name, **stats
    )