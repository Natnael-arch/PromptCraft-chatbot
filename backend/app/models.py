import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Message(Base):
    """One captured WhatsApp message. This is the Phase 1 capture target."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # WAHA's own message id ("true_...@c.us_..."). Unique so webhook retries dedupe.
    waha_message_id: Mapped[str | None] = mapped_column(
        Text, unique=True, nullable=True, index=True
    )
    session_name: Mapped[str] = mapped_column(Text, default="default")
    chat_id: Mapped[str] = mapped_column(Text, index=True)  # chat/group JID, e.g. 123@g.us
    # Not present in WAHA message events; populated by a later phase if needed.
    chat_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_group: Mapped[bool] = mapped_column(Boolean, default=False)
    sender_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    sender_name: Mapped[str | None] = mapped_column(Text, nullable=True)  # pushname if available
    from_me: Mapped[bool] = mapped_column(Boolean, default=False)  # True = bot itself sent it
    # text / image / video / audio / document / sticker / other
    msg_type: Mapped[str] = mapped_column(Text, default="other")
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_mime: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Local path once media is downloaded (Phase 3); left null for now.
    media_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    reply_to_waha_id: Mapped[str | None] = mapped_column(  # quoted message id, if a reply
        Text, nullable=True
    )
    timestamp: Mapped[datetime | None] = mapped_column(  # original WhatsApp message time
        DateTime(timezone=True), nullable=True, index=True
    )
    raw_payload: Mapped[dict] = mapped_column(JSONB)  # full webhook body, for debugging/replay
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MessageUnparsed(Base):
    """Fallback sink for webhooks we could not map into Message - never drop data."""

    __tablename__ = "messages_unparsed"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    event: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    raw_payload: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Session(Base):
    """A burst of activity in one chat, grouped for chunking (Phase 2).

    Chat logs break fixed-size-chunk RAG: short context-free messages don't embed
    usefully in isolation, so sessionizer.py groups messages by a time-gap heuristic
    and each session gets a synthesized context header (participants + rough topic)
    that is prepended to its chunk text. message_ids lists every message in the
    session (content and non-content) so time-range answers can report on all of it.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    chat_id: Mapped[str] = mapped_column(Text, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # One-line synthesized summary (participants, date range, rough topic). Stored
    # alongside the chunk rather than only concatenated into chunk.content so
    # downstream code can display/reuse it without re-deriving it.
    header_text: Mapped[str] = mapped_column(Text)
    # UUIDs (as strings) of every Message in the session.
    message_ids: Mapped[list] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Chunk(Base):
    """A retrievable unit of group history: one embedded text span (Phase 2).

    Phase 1 provisioned this table (empty) with chunks.message_id as a 1:1 placeholder.
    Phase 2 builds chunks from *sessions* that may span many messages, so message_id
    had to become nullable: this is why the migration adds session_id and message_ids
    (the message UUIDs whose text is embedded in this chunk). Backfill chunks set
    message_id to the single message when the chunk covers exactly one message, and to
    NULL otherwise; a future per-message live-embedding path can still use message_id.
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True, nullable=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        # Why session_id instead of just message_id: one chunk spans a session slice
        # (see sessionizer.py). Keying chunks on session_id makes embedding idempotent
        # (delete-and-replace that session's chunks) and lets time-range questions pull
        # all chunks of a session without joining message tables.
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    # UUIDs of the Messages whose text is embedded in this chunk.
    message_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(Vector(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )