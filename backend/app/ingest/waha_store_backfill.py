"""Backfill a chat's missing history from WAHA's NOWEB store into `messages`.

NOWEB keeps a searchable store of messages it has seen (config
``noweb.store.enabled``; with ``fullSync`` defaulting to off it holds a recent
window). The webhook path in app/webhook.py only ever sees messages that arrive
WHILE the backend is listening - so anything that happened before the session
was re-paired (or while webhook delivery was down), notably the bot's own
``true_...`` sent messages, lives only in the store. This module pulls the
store's messages for one chat and inserts the missing ones, then rebuilds that
chat's sessions/chunks so the backfilled rows are retrievable.

Design:

* Fetches via the canonical read endpoint the docs specify for NOWEB with the
  store enabled: ``GET /api/{session}/chats/{chatId}/messages`` (the ``@`` in a
  chat JID is URL-escaped as ``%40``), paginating by ``offset`` until an empty
  page comes back and skipping media downloads (Phase 3 handles media).
* Routes every message through the SAME mapping the webhook uses
  (app/ingest/waha_mapper.build_message), so store rows are indistinguishable
  from live captures and bot-sent messages (``fromMe=true``) are NOT skipped.
* Dedups on ``waha_message_id`` like the webhook, plus one wrinkle: webhook
  captures of GROUP messages persist ids with a trailing ``_<participant>@lid``
  that WAHA's store ids do not carry. The store id is therefore also matched
  against the stripped base of every stored id, so a message captured live once
  is never re-inserted as a near-duplicate with a different id.
* Leaves the transaction boundary to the caller (the admin route commits), like
  rebuild_sessions_and_chunks does.
"""

import logging
import re
from urllib.parse import quote

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.ingest.rebuild import rebuild_sessions_and_chunks
from app.ingest.waha_mapper import build_message
from app.models import Message

logger = logging.getLogger(__name__)

# Pagination step for GET /{session}/chats/{chatId}/messages (docs: advance
# offset by the limit amount even when a page comes back short).
STORE_PAGE_SIZE = 100
# Safety valve against a pathological server that never returns an empty page.
STORE_MAX_PAGES = 1000

# Webhook captures of GROUP messages append "_<participant>@lid" to the store id.
_LID_SUFFIX = re.compile(r"_[^_]+@lid$")


def _strip_lid(waha_id: str) -> str:
    """Base form of a waha_message_id: the store id minus any trailing @lid suffix."""
    return _LID_SUFFIX.sub("", waha_id)


def _waha_headers() -> dict[str, str]:
    return {"X-Api-Key": settings.waha_api_key} if settings.waha_api_key else {}


def fetch_store_messages(session_name: str, chat_id: str) -> list[dict]:
    """Read every stored message for one chat from WAHA's NOWEB store.

    Returns the raw message objects exactly as the webhook ``payload`` shape. An
    empty list means the store holds nothing for the chat (or the store is
    disabled / fullSync off and nothing was captured). Raises on HTTP/parse
    errors; callers translate to a 502/500.
    """
    base = settings.waha_base_url.rstrip("/")
    url = f"{base}/api/{quote(session_name)}/chats/{quote(chat_id, safe='')}/messages"

    messages: list[dict] = []
    offset = 0
    with httpx.Client(timeout=30.0) as client:
        for _ in range(STORE_MAX_PAGES):
            response = client.get(
                url,
                headers=_waha_headers(),
                params={
                    "limit": STORE_PAGE_SIZE,
                    "offset": offset,
                    "downloadMedia": "false",
                },
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise ValueError(
                    f"WAHA store returned {type(page).__name__}, expected a message list"
                )
            messages.extend(page)
            if not page:
                break
            offset += STORE_PAGE_SIZE
        else:
            logger.warning(
                "WAHA store pagination hit safety cap (%d pages), chat=%s",
                STORE_MAX_PAGES,
                chat_id,
            )
    return messages


def _existing_ids(db: Session) -> tuple[set[str], set[str]]:
    """Every stored waha_message_id, plus its @lid-stripped base form."""
    rows = db.execute(
        select(Message.waha_message_id).where(Message.waha_message_id.isnot(None))
    ).all()
    raw: set[str] = set()
    for (wid,) in rows:
        raw.add(wid)
    bases = {_strip_lid(wid) for wid in raw}
    return raw, bases


def backfill_chat_from_store(db: Session, session_name: str, chat_id: str) -> dict:
    """Insert a chat's store-only messages and rebuild its sessions/chunks.

    Returns ``{"fetched", "inserted", "already_existed", "sessions_built",
    "chunks_written"}``. Does NOT commit or roll back - the admin route owns the
    transaction so it can translate embedding/integrity errors into HTTP statuses.
    """
    messages = fetch_store_messages(session_name, chat_id)
    existing, existing_bases = _existing_ids(db)

    bot_id = settings.bot_whatsapp_id
    inserted = 0
    skipped = 0
    for msg in messages:
        waha_id = str(msg.get("id") or "")
        if not waha_id:
            # Cannot deduplicate without WAHA's message id; keep it anyway.
            logger.warning("Store message without id; skipped chat=%s", chat_id)
            skipped += 1
            continue
        if waha_id in existing or waha_id in existing_bases:
            # Already captured live (exact id, or the store id that webhook
            # persisted with the group @lid suffix).
            skipped += 1
            continue

        me = {"id": bot_id} if (msg.get("fromMe") and bot_id) else None
        row = build_message(
            msg,
            session_name=session_name,
            me=me,
            raw_payload={"source": "waha_store", "message": msg},
        )
        db.add(row)
        existing.add(waha_id)
        inserted += 1

    if messages:
        # Flush the pending inserts FIRST: SessionLocal runs with autoflush=False,
        # so a later SELECT (inside the rebuild) would otherwise not see the rows we
        # just added, and the session/chunk index would silently miss them.
        db.flush()
        # Rebuild against the CURRENT DB state so the refresh covers the newly
        # backfilled rows (and stays dedup-free on re-runs - the rebuild is also the
        # repair path if a previous run ever left the index behind the messages).
        stats = rebuild_sessions_and_chunks(db, chat_id)
    else:
        stats = {"sessions_built": 0, "chunks_written": 0}

    logger.info(
        "Store backfill chat=%s session=%s fetched=%d inserted=%d skipped=%d sessions=%d chunks=%d",
        chat_id,
        session_name,
        len(messages),
        inserted,
        skipped,
        stats["sessions_built"],
        stats["chunks_written"],
    )
    return {
        "fetched": len(messages),
        "inserted": inserted,
        "already_existed": skipped,
        "sessions_built": stats["sessions_built"],
        "chunks_written": stats["chunks_written"],
    }