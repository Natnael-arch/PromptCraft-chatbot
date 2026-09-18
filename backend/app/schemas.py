from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MessageRead(BaseModel):
    """Message row as returned by GET /messages."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    waha_message_id: str | None
    session_name: str
    chat_id: str
    chat_name: str | None
    is_group: bool
    sender_id: str | None
    sender_name: str | None
    from_me: bool
    msg_type: str
    body: str | None
    media_mime: str | None
    media_path: str | None
    reply_to_waha_id: str | None
    timestamp: datetime | None
    created_at: datetime


class HealthResponse(BaseModel):
    """GET /health output. status is "ok" when both dependencies are reachable."""

    status: str
    db: bool
    waha: bool


class WebhookResponse(BaseModel):
    """Response for WAHA's webhook delivery. Always a fast 200."""

    status: str
    stored: bool = False
    deduplicated: bool = False