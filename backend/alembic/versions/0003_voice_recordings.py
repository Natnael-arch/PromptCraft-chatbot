"""Phase 3: recordings table + voice chunks (recording_id/source_type/voice_segments)

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-20

Adds the Phase 3 voice path WITHOUT touching Phase 2's session logic:

* new `recordings` table - one row per uploaded voice note / call recording, with
  a sha256 idempotency key and the full Gemini diarized transcript stored raw.
* `chunks` gains a nullable `recording_id` FK (voice chunks reference a recording
  instead of a session), a `source_type` column ('text' | 'voice'), and a
  `voice_segments` JSONB holding the speaker/timestamp segments a voice chunk was
  built from - the source on which /ask builds "Speaker 2, 04:12-04:38" citations.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.create_table(
        "recordings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("chat_id", sa.Text(), nullable=False),
        sa.Column("uploaded_by", sa.Text(), nullable=True),
        sa.Column("original_filename", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=False),
        # sha256 of the uploaded bytes - idempotency key for re-uploads.
        sa.Column("sha256", sa.Text(), nullable=False, unique=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        # pending / transcribing / done / failed
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="pending",
        ),
        # Full Gemini diarized transcript ({"segments": [...]}), kept verbatim.
        sa.Column("raw_transcript_json", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_recordings_chat_id", "recordings", ["chat_id"])
    op.create_index("ix_recordings_sha256", "recordings", ["sha256"], unique=True)

    # ---- chunks: text chunks keep session_id; voice chunks point at a recording.
    op.add_column(
        "chunks",
        sa.Column(
            "source_type",
            sa.Text(),
            nullable=False,
            server_default="text",
        ),
    )
    op.create_check_constraint(
        "ck_chunks_source_type",
        "chunks",
        "source_type IN ('text', 'voice')",
    )
    op.add_column(
        "chunks",
        sa.Column("recording_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_chunks_recording_id_recordings",
        "chunks",
        "recordings",
        ["recording_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_chunks_recording_id", "chunks", ["recording_id"])
    op.add_column("chunks", sa.Column("voice_segments", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "voice_segments")
    op.drop_index("ix_chunks_recording_id", table_name="chunks")
    op.drop_constraint("fk_chunks_recording_id_recordings", "chunks", type_="foreignkey")
    op.drop_column("chunks", "recording_id")
    op.drop_constraint("ck_chunks_source_type", "chunks", type_="check")
    op.drop_column("chunks", "source_type")
    op.drop_index("ix_recordings_sha256", table_name="recordings", unique=True)
    op.drop_index("ix_recordings_chat_id", table_name="recordings")
    op.drop_table("recordings")