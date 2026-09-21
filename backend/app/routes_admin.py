"""Admin routes for the Phase 5 trusted-sender list.

POST/DELETE/GET on ``/admin/trusted-senders`` manage the senders whose content
is weighted higher in retrieval (``weight``) and flagged in citations
(``role_label``/``display_name``). All three are gated on the ``X-Admin-Token``
header matching ``ADMIN_TOKEN``.

Auth note: this is a hackathon-grade stopgap - one shared static token read from
env, compared with ``secrets.compare_digest``, no rate-limiting and no roles. It
protects the trusted-sender list (whose contents bias what the bot surfaces),
not application credentials; replace with a real auth layer before this touches
anything sensitive.
"""

import logging
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import TrustedSender
from app.schemas import (
    AdminChangeResponse,
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