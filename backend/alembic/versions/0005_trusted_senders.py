"""Phase 5: trusted_senders - weighted-announcer boost for retrieval/citations

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-21

Adds the table backing "trusted senders": the announcers/leads whose posts are
weighted higher in hybrid_search ranking (multiplicative boost on the merged
score) and flagged in citations with a role label. One row per sender, keyed by
the WhatsApp JID that `messages.sender_id` stores:

* `sender_id` - PK, the WhatsApp JID
* `display_name` - human name for citation bullets (falls back to sender_name)
* `role_label` - free text shown in citations (default 'announcer')
* `weight` - multiplier applied post-scoring in hybrid_search (default 2.0)
* `added_at` - when the sender was trusted
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0005"
down_revision = "0004"


def upgrade() -> None:
    op.create_table(
        "trusted_senders",
        sa.Column("sender_id", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column(
            "role_label",
            sa.Text(),
            nullable=False,
            server_default="announcer",
        ),
        sa.Column(
            "weight",
            sa.Float(),
            nullable=False,
            server_default="2.0",
        ),
        sa.Column(
            "added_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("sender_id"),
    )


def downgrade() -> None:
    op.drop_table("trusted_senders")