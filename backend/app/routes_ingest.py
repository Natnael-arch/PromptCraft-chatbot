"""POST /ingest/export - upload a WhatsApp chat export and import it.

Takes a .txt (or .zip) file plus a chats.fake group JID as multipart form data,
then runs the Phase 2 backfill pipeline in one transaction:

    parse (auto-detect format) -> insert messages (idempotent, synthetic ids)
        -> sessionize (gap heuristic) -> embed (batched) -> write chunks

The live WAHA webhook capture path is untouched: this endpoint only ever inserts
*new* message rows and rebuilds sessions/chunks for the imported chat.

Idempotency: the same file re-imported inserts zero new messages, and that chat's
sessions/chunks are delete-and-replaced, so no duplicates accumulate.
"""

import logging
from collections import Counter

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.ingest.embedder import EmbeddingError
from app.ingest.rebuild import rebuild_sessions_and_chunks
from app.ingest.whatsapp_export_parser import ExportParseError, parse_export
from app.models import Message
from app.schemas import IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB; exports with media get large


def _insert_messages(db: Session, chat_id: str, records, source_name: str) -> tuple[int, Counter]:
    """Insert new parsed messages; skip any whose synthetic dedup id already exists."""
    from app.ingest.whatsapp_export_parser import synthetic_message_id

    existing: set[str] = set()
    rows = db.execute(
        select(Message.waha_message_id).where(
            Message.waha_message_id.isnot(None), Message.chat_id == chat_id
        )
    ).all()
    for (wid,) in rows:
        existing.add(wid)

    imported = 0
    for record in records:
        dedup_id = synthetic_message_id(
            record.chat_id, record.sender_name, record.local_timestamp, record.body
        )
        if dedup_id in existing:
            continue
        message = Message(
            waha_message_id=dedup_id,
            session_name="export",  # distinguishes backfilled rows from live 'default'
            chat_id=record.chat_id,
            chat_name=record.chat_name,
            is_group=record.chat_id.endswith("@g.us"),
            sender_id=record.sender_id,
            sender_name=record.sender_name,
            from_me=record.from_me,
            msg_type=record.msg_type,
            body=record.body,
            media_mime=None,
            media_path=None,
            reply_to_waha_id=None,
            timestamp=record.timestamp,
            raw_payload={
                "source": "chat_export",
                "source_name": source_name,
                "raw_line": record.raw_line or None,
            },
        )
        db.add(message)
        existing.add(dedup_id)
        imported += 1
    return imported, Counter(r.msg_type for r in records)


@router.post("/ingest/export", response_model=IngestResponse)
def ingest_export(
    file: UploadFile = File(..., description="WhatsApp export .txt or .zip"),
    chat_id: str = Form(..., description="Group JID, e.g. 123456789@g.us"),
    chat_name: str | None = Form(default=None),
    tz_name: str = Form(default="UTC", description="IANA tz of the exported times"),
    db: Session = Depends(get_db),
) -> IngestResponse:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Export exceeds 100MB upload limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    try:
        parsed = parse_export(
            data,
            chat_id=chat_id,
            chat_name=chat_name,
            tz_name=tz_name,
            bot_name=settings.export_bot_name,
            source_name=file.filename or "upload.txt",
        )
    except ExportParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    records = parsed.records

    try:
        # ---- messages (idempotent) ----
        imported, type_counts = _insert_messages(
            db, chat_id, records, parsed.source
        )
        # Sessions/chunks are rebuilt for the WHOLE chat from current DB state so
        # re-imports (and imports after live captures) stay consistent and dedup-free.
        stats = rebuild_sessions_and_chunks(db, chat_id)

        duplicates = len(records) - imported
        response = IngestResponse(
            chat_id=chat_id,
            chat_name=parsed.chat_name,
            source=parsed.source,
            messages_parsed=len(records),
            messages_imported=imported,
            messages_duplicate=max(duplicates, 0),
            sessions_built=stats["sessions_built"],
            chunks_written=stats["chunks_written"],
            msg_type_counts=dict(type_counts),
        )
        db.commit()
    except EmbeddingError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc
    except IntegrityError as exc:
        db.rollback()
        logger.exception("Ingest integrity error")
        raise HTTPException(status_code=409, detail="Ingest conflicted with existing data") from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    logger.info(
        "Ingested chat=%s parsed=%d imported=%d dup=%d sessions=%d chunks=%d",
        chat_id,
        len(records),
        imported,
        duplicates,
        stats["sessions_built"],
        stats["chunks_written"],
    )
    return response