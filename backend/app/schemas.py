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
    embedding_provider: str
    embedding_provider_is_mock: bool


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
    """One sourced message backing part of the answer.

    ``is_announcement``/``role_label`` are set when the message's sender is in
    ``trusted_senders`` (Phase 5): the citation is then visibly flagged as an
    announcement rather than small talk.
    """

    message_id: str
    sender_name: str | None
    timestamp: str | None
    chat_id: str
    chat_name: str | None
    preview: str
    sender_id: str | None = None
    is_announcement: bool = False
    role_label: str | None = None


class AnswerSource(BaseModel):
    """Retrieval source: which chunk/session/messages the answer pulled from.

    Text chunks carry ``session_id``/``message_ids``; voice chunks (Phase 3)
    carry ``source_type="voice"`` with ``recording_id`` plus the speaker and
    segment time range of the exact moment cited (e.g. "Speaker 2, 04:12-04:38").
    """

    chunk_id: str | None = None
    session_id: str | None = None
    score: float | None = None
    message_ids: list[str] = []
    content_preview: str | None = None
    # --- voice (Phase 3) ---
    source_type: str | None = None  # "text" | "voice" (None up to 2026-09-20)
    recording_id: str | None = None
    speaker: str | None = None
    segment_start: float | None = None  # seconds into the recording
    segment_end: float | None = None
    segment: str | None = None  # formatted "Speaker 2, 04:12-04:38"


class AnswerResponse(BaseModel):
    """POST /ask output: a cited answer built from retrieved group history."""

    chat_id: str
    question: str
    route: str  # "semantic" | "time_range"
    answer_text: str
    citations: list[Citation] = []
    sources: list[AnswerSource] = []


# --- Phase 3: voice ------------------------------------------------------------

class RecordingRead(BaseModel):
    """A `recordings` row as returned by GET /voice/recordings."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    chat_id: str
    uploaded_by: str | None
    original_filename: str
    mime_type: str
    sha256: str
    duration_seconds: float | None
    status: str
    raw_transcript_json: dict | None
    error: str | None
    created_at: datetime


class VoiceUploadResponse(BaseModel):
    """POST /voice/upload progress summary (returned for both fresh + duplicate)."""

    recording_id: str
    chat_id: str
    status: str  # done | failed
    duplicate: bool  # True when an identical file (sha256) was already stored
    duration_seconds: float | None
    speakers: list[str]
    chunk_count: int
    original_filename: str
    uploaded_by: str | None


class TrustedSenderCreate(BaseModel):
    """POST /admin/trusted-senders payload.

    ``sender_id`` is the WhatsApp JID stored in ``messages.sender_id``.
    ``role_label``/``weight`` default to the table defaults so a phone-number-only
    upsert is a valid call.
    """

    sender_id: str
    display_name: str | None = None
    role_label: str | None = None  # defaults to 'announcer' in the table
    weight: float | None = None  # defaults to 2.0 in the table


class TrustedSenderRead(BaseModel):
    """A `trusted_senders` row as returned by GET /admin/trusted-senders."""

    model_config = ConfigDict(from_attributes=True)

    sender_id: str
    display_name: str | None
    role_label: str
    weight: float
    added_at: datetime


class AdminChangeResponse(BaseModel):
    """Confirmation body for POST/DELETE /admin/trusted-senders."""

    ok: bool
    sender_id: str
    action: str  # added | updated | removed


class StoreBackfillResponse(BaseModel):
    """POST /admin/store-backfill/{chat_id} summary."""

    chat_id: str
    session_name: str
    fetched: int  # messages read from WAHA's NOWEB store
    inserted: int  # new rows written to `messages`
    already_existed: int  # already present live (including @lid-suffixed rows)
    sessions_built: int
    chunks_written: int