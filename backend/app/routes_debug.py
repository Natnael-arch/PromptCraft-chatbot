"""GET /health and GET /messages - manual verification for Phase 1."""

import logging

import httpx
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Message
from app.schemas import HealthResponse, MessageRead

logger = logging.getLogger(__name__)

router = APIRouter(tags=["debug"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        logger.exception("Health: database check failed")
        db_ok = False

    waha_ok = True
    try:
        headers = {"X-Api-Key": settings.waha_api_key} if settings.waha_api_key else {}
        # /api/sessions doubles as a liveness + auth probe: 2xx means WAHA is up and
        # our API key is accepted.
        response = httpx.get(
            f"{settings.waha_base_url}/api/sessions", headers=headers, timeout=3.0
        )
        waha_ok = response.status_code < 400
        if not waha_ok:
            logger.warning("Health: WAHA returned HTTP %s", response.status_code)
    except Exception:  # noqa: BLE001
        logger.exception("Health: WAHA check failed")
        waha_ok = False

    return HealthResponse(
        status="ok" if db_ok and waha_ok else "degraded",
        db=db_ok,
        waha=waha_ok,
        embedding_provider=settings.embedding_provider,
        embedding_provider_is_mock=(settings.embedding_provider.lower() == "mock"),
    )


@router.get("/messages", response_model=list[MessageRead])
def list_messages(
    chat_id: str | None = Query(
        default=None, description="Filter by chat/group JID, e.g. 1234567890@g.us"
    ),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[Message]:
    stmt = (
        select(Message)
        .order_by(Message.timestamp.desc().nulls_last(), Message.created_at.desc())
        .limit(limit)
    )
    if chat_id:
        stmt = stmt.where(Message.chat_id == chat_id)
    return list(db.execute(stmt).scalars().all())