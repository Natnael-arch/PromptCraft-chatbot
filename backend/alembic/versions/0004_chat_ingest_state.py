"""Phase 4: chat_ingest_state - durable per-chat cursor for live incremental ingest

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-21

Adds the table the live-ingest background task uses to skip rebuilds when a chat
has no messages newer than the last rebuilt message:

* `chat_ingest_state` - one row per chat, tracking `last_message_id` (the most
  recent `messages` row that row's sessions/chunks have been rebuilt against)
  and `last_rebuilt_at` (informational). `last_message_id` points at
  `messages.id` with ON DELETE SET NULL so deleting a captured message never
  blocks the write path.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "0003"


def upgrade() -> None:
    op.create_table(
        "chat_ingest_state",
        sa.Column("chat_id", sa.Text(), nullable=False),
        sa.Column("last_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("last_rebuilt_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("chat_id"),
        sa.ForeignKeyConstraint(
            ["last_message_id"],
            ["messages.id"],
            ondelete="SET NULL",
        ),
    )


def downgrade() -> None:
    op.drop_table("chat_ingest_state")