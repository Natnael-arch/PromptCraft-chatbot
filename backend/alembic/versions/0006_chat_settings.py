"""Phase 5/6: chat_settings - per-chat /unhinged_* override for casual/banter mode

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-21

Adds the table backing per-chat runtime overrides of the casual/banter mode.
Each chat that has ever received an explicit toggle gets a row; chats with no
row simply use the global ``BANTER_MODE_ENABLED`` env default:

* `chat_id` - PK, the chat/group JID
* `unhinged_enabled` - nullable tri-state. True/False override the global default
  for this chat; NULL (and missing rows) mean "use the global default".
* `updated_at` - when the override was last written
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0006"
down_revision = "0005"


def upgrade() -> None:
    op.create_table(
        "chat_settings",
        sa.Column("chat_id", sa.Text(), nullable=False),
        sa.Column("unhinged_enabled", sa.Boolean(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("chat_id"),
    )


def downgrade() -> None:
    op.drop_table("chat_settings")