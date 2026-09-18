"""initial schema: messages, messages_unparsed, chunks

Revision ID: 0001
Revises:
Create Date: 2026-09-18
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None


def upgrade() -> None:
    # pgvector ships with the `pgvector/pgvector:pg16` image; make sure it's live.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("waha_message_id", sa.Text(), nullable=True),
        sa.Column("session_name", sa.Text(), nullable=False, server_default="default"),
        sa.Column("chat_id", sa.Text(), nullable=False),
        sa.Column("chat_name", sa.Text(), nullable=True),
        sa.Column("is_group", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("sender_id", sa.Text(), nullable=True),
        sa.Column("sender_name", sa.Text(), nullable=True),
        sa.Column("from_me", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("msg_type", sa.Text(), nullable=False, server_default="other"),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("media_mime", sa.Text(), nullable=True),
        sa.Column("media_path", sa.Text(), nullable=True),
        sa.Column("reply_to_waha_id", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_messages_chat_id", "messages", ["chat_id"])
    op.create_index("ix_messages_sender_id", "messages", ["sender_id"])
    op.create_index("ix_messages_timestamp", "messages", ["timestamp"])
    op.create_index("ix_messages_waha_message_id", "messages", ["waha_message_id"], unique=True)

    op.create_table(
        "messages_unparsed",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("event", sa.Text(), nullable=True),
        sa.Column("session_name", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_chunks_message_id", "chunks", ["message_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_message_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("messages_unparsed")
    op.drop_index("ix_messages_waha_message_id", table_name="messages")
    op.drop_table("messages")
    op.execute("DROP EXTENSION IF EXISTS vector")