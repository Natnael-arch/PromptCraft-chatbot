import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Text, func, text
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


class ChatIngestState(Base):
    """Durable per-chat cursor for the incremental live-ingest pipeline (Phase 4).

    One row per chat that has ever had a webhook-captured message run through the
    live ingest path. ``last_message_id`` is the most recent `messages` row the
    chat's sessions/chunks have been rebuilt against - the next trigger skips the
    rebuild entirely when nothing is newer, so active groups don't re-embed their
    whole history on every single message. ``last_rebuilt_at`` is informational
    (monitoring / debugging); the cooldown decision uses the in-memory clock.
    """

    __tablename__ = "chat_ingest_state"

    chat_id: Mapped[str] = mapped_column(Text, primary_key=True)
    last_message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    last_rebuilt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ChatSetting(Base):
    """Per-chat runtime settings (Phase 5/6: the /unhinged_* toggle).

    One row per chat that has ever had an explicit override set via a slash
    command. ``unhinged_enabled`` is a tri-state:

    * ``True``  - casual/banter replies are ON in this chat regardless of the
      global ``BANTER_MODE_ENABLED`` env default.
    * ``False`` - casual/banter replies are OFF in this chat (knowledge path
      only), regardless of the global default.
    * ``NULL``  - fall back to ``settings.banter_mode_enabled``.

    No row at all is treated the same as NULL (reading never creates a row, so
    untouched chats simply use the global default).
    """

    __tablename__ = "chat_settings"

    chat_id: Mapped[str] = mapped_column(Text, primary_key=True)
    unhinged_enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TrustedSender(Base):
    """A sender whose content is weighted higher in retrieval/citations (Phase 5).

    Announcers/leads post the authoritative updates in a hackathon group
    (deadlines, submission links, judging criteria). Retrieval multiplies the
    relevance score of chunks containing their messages by ``weight`` (default
    2.0, so an announcement chunk can outrank a topically-closer casual mention),
    and citations from them carry ``is_announcement`` + ``role_label`` so the
    answer visibly flags the source as an announcement rather than small talk.

    ``sender_id`` matches ``Message.sender_id`` (a WhatsApp JID).
    """

    __tablename__ = "trusted_senders"

    sender_id: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    role_label: Mapped[str] = mapped_column(Text, default="announcer", server_default="announcer")
    weight: Mapped[float] = mapped_column(Float, default=2.0, server_default="2.0")
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Recording(Base):
    """One uploaded voice note / call recording (Phase 3).

    Voice notes bypass the `messages`/`sessions` tables entirely: the raw bytes
    are hashed (sha256) for idempotent re-uploads, transcribed via Gemini, and
    the diarized transcript is stored here as raw_transcript_json so chunks can be
    re-built without re-transcribing. The transcription pipeline never drops a
    failing response: status is set to "failed" with the raw response preserved
    for debugging.
    """

    __tablename__ = "recordings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    chat_id: Mapped[str] = mapped_column(Text, index=True)
    uploaded_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str] = mapped_column(Text)
    # sha256 of the uploaded audio bytes - idempotency key. Re-uploading the same
    # file returns the existing row instead of re-transcribing.
    sha256: Mapped[str] = mapped_column(Text, unique=True, index=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # pending / transcribing / done / failed
    status: Mapped[str] = mapped_column(Text, default="pending", server_default="pending")
    # Full Gemini diarized transcript response ({segments: [...]}), kept verbatim
    # so chunks can be re-sessionized without a second transcription call.
    raw_transcript_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    # Phase 3 voice: a chunk either belongs to a text session (session_id) or a
    # voice recording (recording_id). source_type distinguishes the two so hybrid
    # retrieval can pull from both halves of the same chunks table.
    source_type: Mapped[str] = mapped_column(
        Text, default="text", server_default="text"
    )
    recording_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("recordings.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    # For voice chunks: the [speaker, start, end, text] segments this chunk was
    # built from, so selections can cite the exact moment ("Speaker 2, 04:12-04:38")
    # instead of a session message range.
    voice_segments: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(Vector(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )