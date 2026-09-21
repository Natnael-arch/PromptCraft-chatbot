"""Per-chat settings: read/upsert the ``unhinged_enabled`` override (Phase 5/6).

The casual/banter mode used to be governed solely by the global
``BANTER_MODE_ENABLED`` env flag. Phase 5/6 makes it per-chat: a
``chat_settings`` row overrides the global default, so one group can be
"unhinged" while another (or a DM) is not, and the setting survives restarts
(it lives in Postgres, not memory).

The global env flag has NOT disappeared - it is simply demoted to *the default
for chats with no override*, which is what ``get_unhinged_enabled`` falls back
to. ``None`` rows and missing rows behave identically ("use the default").
"""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ChatSetting

logger = logging.getLogger(__name__)


def get_unhinged_enabled(db: Session, chat_id: str) -> bool:
    """Whether casual/banter replies are on for ``chat_id``.

    Returns the per-chat ``unhinged_enabled`` override when set (True/False),
    otherwise the global ``settings.banter_mode_enabled`` default. Reading never
    creates a row - a chat that has never been toggled just follows the global.
    """
    row = db.execute(
        select(ChatSetting).where(ChatSetting.chat_id == chat_id)
    ).scalars().first()
    if row is not None and row.unhinged_enabled is not None:
        return row.unhinged_enabled
    return settings.banter_mode_enabled


def upsert_unhinged(db: Session, chat_id: str, enabled: bool) -> ChatSetting:
    """Create or update the chat's ``unhinged_enabled`` override and commit.

    Idempotent: toggling the same command twice just rewrites the row.
    """
    row = db.execute(
        select(ChatSetting).where(ChatSetting.chat_id == chat_id)
    ).scalars().first()
    if row is None:
        row = ChatSetting(chat_id=chat_id, unhinged_enabled=enabled)
        db.add(row)
    else:
        row.unhinged_enabled = enabled
    db.commit()
    logger.info(
        "chat_settings unhinged_enabled=%s for chat=%s", enabled, chat_id
    )
    return row