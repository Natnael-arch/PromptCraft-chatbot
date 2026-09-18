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


# --- Phase 2: ingest + ask ------------------------------------------------------

class IngestResponse(BaseModel):
    """POST /ingest/export summary."""

    chat_id: str
    chat_name: str | None
    source: str
    messages_parsed: int  # records parsed out of the file
    messages_imported: int  # new rows written to `messages`
    messages_duplicate: int  # already present (idempotent re-import)
    sessions_built: int
    chunks_written: int
    msg_type_counts: dict[str, int]  # parsed message types, for diagnostics


class AskRequest(BaseModel):
    """POST /ask body."""

    chat_id: str
    question: str


class Citation(BaseModel):
    """One sourced message backing part of the answer."""

    message_id: str
    sender_name: str | None
    timestamp: str | None
    chat_id: str
    chat_name: str | None
    preview: str


class AnswerSource(BaseModel):
    """Retrieval source: which chunk/session/messages the answer pulled from."""

    chunk_id: str | None = None
    session_id: str | None = None
    score: float | None = None
    message_ids: list[str] = []
    content_preview: str | None = None


class AnswerResponse(BaseModel):
    """POST /ask output: a cited answer built from retrieved group history."""

    chat_id: str
    question: str
    route: str  # "semantic" | "time_range"
    answer_text: str
    citations: list[Citation] = []
    sources: list[AnswerSource] = []