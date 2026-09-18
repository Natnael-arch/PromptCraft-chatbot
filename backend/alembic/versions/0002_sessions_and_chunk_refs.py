"""Phase 2: sessions table + chunk session/message refs + retrieval indexes

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-18
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    # ---- sessions: activity bursts grouped by sessionizer.py ----
    op.create_table(
        "sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("chat_id", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("header_text", sa.Text(), nullable=False),
        sa.Column("message_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_sessions_chat_id", "sessions", ["chat_id"])
    op.create_index("ix_sessions_started_at", "sessions", ["started_at"])
    # Composite index powers time-range question routing ("what happened this week").
    op.create_index("ix_sessions_chat_started", "sessions", ["chat_id", "started_at"])

    # ---- chunks: Phase 1 provisioned message_id as 1:1; Phase 2 backfill chunks
    # ---- span whole sessions, so message_id becomes nullable and we add
    # ---- session_id (idempotent delete-and-replace) + message_ids (source refs).
    op.alter_column("chunks", "message_id", nullable=True)
    op.add_column(
        "chunks",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_chunks_session_id_sessions",
        "chunks",
        "sessions",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_chunks_session_id", "chunks", ["session_id"])
    op.add_column("chunks", sa.Column("message_ids", postgresql.JSONB(), nullable=True))

    # pgvector ANN index (cosine) - requires pgvector >= 0.5, which the pinned
    # pg16 image ships. Keeping chunks small (session-sliced) makes this cheap.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    # Keyword leg of hybrid search (Phase 2): GIN over the English tsvector so
    # GET /ask does not degrade to a full-table scan. English stemmer is a
    # pragmatic default for chat content; swap the config name if the group is
    # predominantly non-English.
    op.execute(
        "CREATE INDEX ix_chunks_content_fts ON chunks "
        "USING gin (to_tsvector('english', content))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_fts")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_index("ix_chunks_session_id", table_name="chunks")
    op.drop_constraint("fk_chunks_session_id_sessions", "chunks", type_="foreignkey")
    op.drop_column("chunks", "message_ids")
    op.drop_column("chunks", "session_id")
    op.alter_column("chunks", "message_id", nullable=False)
    op.drop_index("ix_sessions_chat_started", table_name="sessions")
    op.drop_index("ix_sessions_started_at", table_name="sessions")
    op.drop_index("ix_sessions_chat_id", table_name="sessions")
    op.drop_table("sessions")