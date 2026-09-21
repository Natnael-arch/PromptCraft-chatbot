"""Phase 4 live-ingest worker: keep a chat's sessions/chunks in sync with webhook.

Runs as a FastAPI BackgroundTask (scheduled from webhook.py right after the
message row is committed), so it never blocks WAHA's delivery ack. It mirrors
the defensive style of app/reply/reply_worker.py: a fresh DB session is opened
here, everything is logged, and no exception ever propagates out of the task.

Cheapness comes from two independent mechanisms:

* an in-memory per-chat debounce (same InMemoryCooldown class reply_worker uses):
  a burst of messages collapses into a single rebuild per window;
* a durable per-chat cursor (chat_ingest_state.last_message_id): a rebuild is
  skipped entirely when nothing is newer than the last rebuilt message, so a
  chat's full history is never re-embedded just because it got one new message.

The rebuild (sessionize -> delete-and-replace sessions -> embed -> write chunks)
lives in app/ingest/rebuild.py, shared with the manual /ingest/export endpoint.

Like the reply cooldown, the debounce is process-local by design (Phase 5 moves
to Redis); a threaded dict of per-chat locks serializes rebuilds within this
process so two near-simultaneous deliveries can never run overlapping
delete-and-replace rebuilds for the same chat.
"""

import logging
import threading
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.ingest.rebuild import rebuild_sessions_and_chunks
from app.models import ChatIngestState, Message
from app.reply.rate_limit import InMemoryCooldown

logger = logging.getLogger(__name__)

# Per-chat debounce instance, independent of the reply cooldown. Reads the
# configured window at import time (same pattern as reply_worker._cooldown).
_ingest_cooldown = InMemoryCooldown(settings.ingest_debounce_seconds)

# Per-chat rebuild locks + the guard protecting the dict. Locks are never
# removed - fine at this scale (one dict entry per chat that ever rebuilt).
_INGEST_LOCKS: dict[str, threading.Lock] = {}
_INGEST_LOCKS_GUARD = threading.Lock()


def _reset() -> None:
    """Test hook: drop debounce state and per-chat locks."""
    _ingest_cooldown.clear()
    with _INGEST_LOCKS_GUARD:
        _INGEST_LOCKS.clear()


def _lock_for(chat_id: str) -> threading.Lock:
    """Return the process-local lock serializing this chat's rebuilds."""
    with _INGEST_LOCKS_GUARD:
        lock = _INGEST_LOCKS.get(chat_id)
        if lock is None:
            lock = threading.Lock()
            _INGEST_LOCKS[chat_id] = lock
        return lock


def _newest_message_id(db, chat_id: str):
    """Most recent captured `messages` row for this chat (by DB insert time).

    ``created_at`` (server default now()) is the right monotonic key for a live
    sink: WhatsApp's own ``timestamp`` is the client's clock and can jump around,
    while every webhook capture lands in row-insert order.
    """
    return db.execute(
        select(Message.id)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def schedule_incremental_ingest(chat_id: str) -> None:
    """Debounced, serialized, cursor-skipping rebuild for a chat (fire-and-forget).

    Called from the webhook background tasks after a message is committed. Never
    raises: webhook health and delivery must not depend on the ingest pipeline.
    """
    if not chat_id:
        return
    if not settings.ingest_auto_enabled:
        logger.info("Live ingest disabled (INGEST_AUTO_ENABLED=false); skip chat=%s", chat_id)
        return

    # Debounce first: a burst of messages collapses into one rebuild per window.
    if not _ingest_cooldown.allowed(str(chat_id)):
        logger.info("Live ingest debounced (recent rebuild) chat=%s", chat_id)
        return

    db = SessionLocal()
    try:
        with _lock_for(chat_id):
            newest = _newest_message_id(db, chat_id)
            if newest is None:
                # No messages captured for this chat yet - nothing to index.
                logger.info("Live ingest no messages for chat=%s; skipping", chat_id)
                return

            cursor = db.execute(
                select(ChatIngestState).where(ChatIngestState.chat_id == chat_id)
            ).scalar_one_or_none()
            if cursor is not None and cursor.last_message_id == newest:
                # Nothing newer than the last rebuild - skip the re-embed entirely.
                logger.info("Live ingest nothing new (cursor %s) chat=%s", newest, chat_id)
                return

            stats = rebuild_sessions_and_chunks(db, chat_id)

            row = cursor if cursor is not None else ChatIngestState(chat_id=chat_id)
            row.last_message_id = newest
            row.last_rebuilt_at = datetime.now(timezone.utc)
            if cursor is None:
                db.add(row)
            db.commit()

            logger.info(
                "Incremental ingest chat=%s sessions=%d chunks=%d cursor=%s",
                chat_id,
                stats["sessions_built"],
                stats["chunks_written"],
                newest,
            )
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("Incremental ingest failed for chat=%s", chat_id)
    finally:
        db.close()