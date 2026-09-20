"""POST /voice/upload + GET /voice/recordings - Phase 3 voice ingestion.

Synchronous upload flow (queueing through Redis is a Phase 5 concern, mirroring
Phase 2's /ingest/export):

    read + validate audio -> sha256 idempotency check -> recordings row
        -> Gemini transcribe (diarized, timestamped)
        -> voice_sessionizer chunking (Phase-2-style context headers)
        -> Phase 2 embedder -> chunks rows (source_type='voice', recording_id)

Voice chunks are ordinary rows in the same `chunks` table as text chunks, so the
existing hybrid_search picks them up automatically (see app/retrieval/search.py).
A failing transcription NEVER drops the audio: the row is marked `failed` with
the raw Gemini response preserved (raw_transcript_json) for debugging.
"""

import hashlib
import logging
from datetime import datetime, timezone
from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.ingest.embedder import EmbeddingError, embed_chunks, get_embedder
from app.models import Chunk, Recording
from app.schemas import RecordingRead, VoiceUploadResponse
from app.voice.transcriber import (
    AudioFormatError,
    TranscriptionError,
    TranscriptionRequestError,
    get_transcriber,
    normalize_mime_type,
)
from app.voice.voice_sessionizer import (
    build_voice_chunks,
    build_voice_header,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])


def _speaker_labels(raw_transcript) -> list[str]:
    """Distinct speaker labels (order of first appearance) from a transcript."""
    if not raw_transcript or not isinstance(raw_transcript.get("segments"), list):
        return []
    seen: list[str] = []
    for seg in raw_transcript["segments"]:
        name = (seg.get("speaker") or "").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _replace_chunks_for_recording(
    db: Session, recording_id: str, chunk_builds, vectors
) -> int:
    """Idempotent chunk writer keyed on recording_id (delete-and-replace),
    mirroring embedder.replace_chunks_for_session for the text path."""
    db.execute(delete(Chunk).where(Chunk.recording_id == recording_id))
    for build, vector in zip(chunk_builds, vectors):
        db.add(
            Chunk(
                source_type="voice",
                recording_id=recording_id,
                session_id=None,
                message_ids=None,
                voice_segments=[dict(s) for s in build.segments],
                content=build.content,
                embedding=vector,
            )
        )
    return len(chunk_builds)


def _build_response(
    rec: Recording,
    *,
    duplicate: bool,
    db: Session,
) -> VoiceUploadResponse:
    chunk_count = db.execute(
        select(func.count(Chunk.id)).where(Chunk.recording_id == rec.id)
    ).scalar_one()
    return VoiceUploadResponse(
        recording_id=str(rec.id),
        chat_id=rec.chat_id,
        status=rec.status,
        duplicate=duplicate,
        duration_seconds=rec.duration_seconds,
        speakers=_speaker_labels(rec.raw_transcript_json),
        chunk_count=chunk_count,
        original_filename=rec.original_filename,
        uploaded_by=rec.uploaded_by,
    )


@router.post("/voice/upload", response_model=VoiceUploadResponse)
def voice_upload(
    file: UploadFile = File(..., description="Ogg/Opus, MP3, M4A, WAV or WebM audio"),
    chat_id: str = Form(..., description="Chat/group JID, e.g. 123456789@g.us"),
    uploaded_by: str | None = Form(default=None, description="Who (a WhatsApp pushname) uploaded it"),
    db: Session = Depends(get_db),
) -> VoiceUploadResponse:
    data = file.file.read(settings.voice_max_upload_bytes + 1)
    if len(data) > settings.voice_max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds {settings.voice_max_upload_bytes // (1024 * 1024)}MB upload limit",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        mime_type = normalize_mime_type(file.content_type, file.filename or "")
    except AudioFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    digest = hashlib.sha256(data).hexdigest()

    # Idempotency: identical bytes are the same recording, so re-uploading the
    # exact same file returns the existing row instead of re-transcribing /
    # re-embedding. sha256 is globally unique on `recordings` as a second
    # guarantee against a concurrent duplicate upload.
    existing = db.execute(
        select(Recording).where(Recording.sha256 == digest)
    ).scalars().first()
    if existing is not None:
        return _build_response(existing, duplicate=True, db=db)

    rec = Recording(
        chat_id=chat_id,
        uploaded_by=uploaded_by,
        original_filename=file.filename or "voice_note",
        mime_type=mime_type,
        sha256=digest,
        status="pending",
    )
    db.add(rec)
    db.flush()  # grab rec.id before the long transcription run

    provider = get_transcriber()
    try:
        # --- transcribe ---
        result = provider.transcribe(data, mime_type, filename=rec.original_filename)
        seg_dicts = [asdict(s) for s in result.segments]
        if not seg_dicts:
            raise TranscriptionError(
                "Transcription produced zero segments; marking recording failed"
            )

        # --- sessionize (Phase-2-style context headers, same char budget) ---
        header = build_voice_header(
            seg_dicts,
            uploaded_at=datetime.now(timezone.utc),
            duration_seconds=result.duration_seconds,
        )
        chunk_builds = build_voice_chunks(seg_dicts, header=header)

        # --- embed through Phase 2's embedder (no second pipeline) ---
        embedder = get_embedder()
        vectors = embed_chunks(embedder, [c.content for c in chunk_builds])
        if len(vectors) != len(chunk_builds):
            raise EmbeddingError(
                f"Embedder returned {len(vectors)} vectors for {len(chunk_builds)} chunks"
            )
        written = _replace_chunks_for_recording(db, str(rec.id), chunk_builds, vectors)

        rec.status = "done"
        rec.duration_seconds = result.duration_seconds
        rec.raw_transcript_json = result.raw_response
        rec.error = None
        db.commit()
        logger.info(
            "Voice ingest done chat=%s recording=%s segments=%d chunks=%d duration=%s",
            chat_id, rec.id, len(seg_dicts), written, result.duration_seconds,
        )
        return _build_response(rec, duplicate=False, db=db)

    except (TranscriptionError, TranscriptionRequestError) as exc:
        db.rollback()  # undo any partial flush; keep rec (re-attach via add)
        raw = getattr(exc, "raw", None)
        rec.status = "failed"
        rec.error = str(exc)
        rec.raw_transcript_json = (
            raw if isinstance(raw, dict) else {"raw_text": raw or ""}
        )
        db.add(rec)
        db.commit()
        logger.exception("Voice transcription failed for recording=%s", rec.id)
        raise HTTPException(
            status_code=502,
            detail=f"Transcription failed: {exc} (recording {rec.id} saved as failed for debugging)",
        ) from exc

    except EmbeddingError as exc:
        db.rollback()
        rec.status = "failed"
        rec.error = str(exc)
        db.add(rec)
        db.commit()
        logger.exception("Voice embedding failed for recording=%s", rec.id)
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        rec.status = "failed"
        rec.error = f"{type(exc).__name__}: {exc}"
        db.add(rec)
        db.commit()
        logger.exception("Voice ingest failed for recording=%s", rec.id)
        raise HTTPException(status_code=500, detail=f"Voice ingest failed: {exc}") from exc


@router.get("/voice/recordings", response_model=list[RecordingRead])
def list_recordings(
    chat_id: str | None = Query(
        default=None, description="Filter by chat/group JID, e.g. 1234567890@g.us"
    ),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[Recording]:
    stmt = (
        select(Recording)
        .order_by(Recording.created_at.desc())
        .limit(limit)
    )
    if chat_id:
        stmt = stmt.where(Recording.chat_id == chat_id)
    return list(db.execute(stmt).scalars().all())