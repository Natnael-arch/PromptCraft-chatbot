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


class Chunk(Base):
    """Empty placeholder for Phase 2 embeddings - provisioned now to avoid a migration later."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(Vector(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())