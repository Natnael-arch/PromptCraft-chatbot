"""POST /ask - answer a question against the group's embedded history."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingest.embedder import EmbeddingError, get_embedder
from app.models import Message
from app.retrieval.answer import answer_question
from app.schemas import AnswerResponse, AskRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AnswerResponse)
def ask(
    body: AskRequest,
    db: Session = Depends(get_db),
) -> AnswerResponse:
    chat_id = body.chat_id.strip()
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    # 404 when nothing has ever been captured/imported for this chat.
    count = db.execute(
        select(func.count(Message.id)).where(Message.chat_id == chat_id)
    ).scalar_one()
    if count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No history for chat_id={chat_id}. Import a chat export first "
            "(POST /ingest/export) or wait for live webhook capture.",
        )

    try:
        provider = get_embedder()  # semantic path only; time_range ignores vectors
        return AnswerResponse(**answer_question(db, chat_id, question, embedding_provider=provider))
    except EmbeddingError as exc:
        logger.exception("Ask failed at embedding time")
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc