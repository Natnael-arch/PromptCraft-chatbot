"""Shared session/chunk rebuild pipeline.

Single source of truth for "sessionize this chat's messages -> delete-and-replace
its `sessions` rows (chunks cascade) -> re-embed". Both callers use it:

* the manual file import ``POST /ingest/export`` (routes_ingest.py), and
* the automatic live-ingest background task (app/ingest/live_ingest.py), which
  (re)builds a chat's index whenever fresh messages arrive via the webhook.

The caller owns the transaction boundary - this module never commits or rolls
back, so the export endpoint can wrap the whole import in one transaction while
the live path commits once after rebuilding.
"""

import logging

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.ingest.embedder import embed_chunks, get_embedder, replace_chunks_for_session
from app.ingest.sessionizer import sessionize
from app.models import Message, Session as ChatSession

logger = logging.getLogger(__name__)


def matching_message_dicts(db: Session, chat_id: str) -> list[dict]:
    """All of a chat's messages reshaped for the sessionizer.

    Mirrors the shape the sessionizer expects: ``id``, ``sender_name``, ``body``,
    ``msg_type``, ``timestamp`` (aware datetime) and ``content`` (bool deciding
    whether the message contributes embeddable text). Messages without a usable
    timestamp are skipped by sessionize but remain stored in `messages`.
    """
    rows = db.execute(
        select(Message).where(Message.chat_id == chat_id)
    ).scalars().all()
    return [
        {
            "id": str(m.id),
            "sender_name": m.sender_name,
            "body": m.body,
            "msg_type": m.msg_type,
            "timestamp": m.timestamp,
            "content": bool(m.body) and m.msg_type not in {"system", "media"},
        }
        for m in rows
    ]


def rebuild_sessions_and_chunks(db: Session, chat_id: str) -> dict:
    """Rebuild a chat's sessions/chunks from its current `messages` rows.

    Delete-and-replace this chat's sessions (chunks cascade via ``session_id``
    FK) and re-run sessionize + embedding over the COMPLETE message history, so
    repeated runs stay dedup-free and always reflect every captured message.

    Returns ``{"sessions_built": int, "chunks_written": int}``.

    Deliberately does NOT commit - the caller (HTTP route or background ingest
    task) decides when the transaction is complete. May raise an embedding or
    integrity error; callers are expected to rollback/translate as appropriate.
    """
    messages = matching_message_dicts(db, chat_id)
    sessions = sessionize(
        messages,
        gap_minutes=settings.session_gap_minutes,
        max_messages=settings.session_max_messages,
        max_chunk_chars=settings.session_max_chunk_chars,
    )

    # Delete-and-replace this chat's sessions + chunks (cascade deletes chunks).
    db.execute(delete(ChatSession).where(ChatSession.chat_id == chat_id))

    provider = get_embedder()
    sessions_built = 0
    chunks_written = 0
    for session in sessions:
        session_row = ChatSession(
            chat_id=chat_id,
            started_at=session.started_at,
            ended_at=session.ended_at,
            header_text=session.header_text,
            message_ids=session.message_ids,
        )
        db.add(session_row)
        db.flush()  # get session_row.id for the chunk FK
        sessions_built += 1
        if session.chunks:
            contents = [c.content for c in session.chunks]
            vectors = embed_chunks(provider, contents)
            replace_chunks_for_session(db, str(session_row.id), session.chunks, vectors)
            chunks_written += len(session.chunks)

    return {"sessions_built": sessions_built, "chunks_written": chunks_written}