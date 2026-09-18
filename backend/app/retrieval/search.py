"""Hybrid retrieval: pgvector cosine similarity + Postgres full-text search, merged.

Do NOT use pure vector search alone on questions that are really asking for a
complete time-range summary ("what happened this week", "what did I miss
yesterday") - vector search on a broad temporal question returns
semantically-similar *noise*, not a complete set of what actually happened.

``answer.py`` routes by question type before this module is called. This module
handles the *semantic* path only (and is safe to use for time-scoped subsets
via a ``chat_id`` filter that callers can narrow by session range before calling
``hybrid_search`` if needed - not implemented here to keep the function stateless).
"""

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, Session as ChatSession

logger = logging.getLogger(__name__)


def _fts_condition(query: str):
    """plainto_tsquery handles unaccented English well enough for group chat."""
    return func.to_tsvector("english", Chunk.content).op("@@")(
        func.plainto_tsquery("english", query)
    )


def vector_search(
    db: Session,
    chat_id: str,
    qvec: list[float],
    *,
    k: int,
) -> list[tuple[str, float]]:
    """Top-k nearest neighbors by cosine distance over the chunk embedding column.

    Returns [(chunk_id_str, similarity_score)] sorted descending.
    """
    distance = Chunk.embedding.cosine_distance(qvec)
    score_expr = (1 - distance).label("score")
    stmt = (
        select(Chunk.id.label("id"), score_expr)
        .join(ChatSession, Chunk.session_id == ChatSession.id)
        .where(
            ChatSession.chat_id == chat_id,
            Chunk.session_id.isnot(None),
        )
        .order_by(distance)
        .limit(k)
    )
    return [(str(row.id), row.score) for row in db.execute(stmt).all()]


def keyword_search(
    db: Session,
    chat_id: str,
    query: str,
    *,
    k: int,
) -> list[tuple[str, float]]:
    """Keyword/phrase leg: Postgres tsvector GIN index powering plainto_tsquery.

    ts_rank_cd scores by proximity; results are the best-matching ranked documents
    from the FTS index. Returns [(chunk_id_str, rank_score)] sorted descending.
    """
    query = query.strip()
    if not query:
        return []

    cd = func.ts_rank_cd(
        func.to_tsvector("english", Chunk.content),
        func.plainto_tsquery("english", query),
        normalization=32,  # rank_cd normalization by document length
    ).label("score")

    stmt = (
        select(Chunk.id.label("id"), cd)
        .join(ChatSession, Chunk.session_id == ChatSession.id)
        .where(
            ChatSession.chat_id == chat_id,
            Chunk.session_id.isnot(None),
            _fts_condition(query),
        )
        .order_by(cd.desc())
        .limit(k)
    )
    return [(str(row.id), row.score) for row in db.execute(stmt).all()]


def hybrid_search(
    db: Session,
    chat_id: str,
    query: str,
    qvec: list[float],
    *,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Merge vector + FTS via reciprocal-rank fusion (RRF).

    The original paper's constant is k = 60. Each list contributes
    ``1/(k + rank)`` to a cumulative score; the two lists are combined and the
    top_k are returned. This is simple, fast and swaps easily for a real reranker
    later (replace ``_rrf`` with whatever handles an external reranker service).
    """
    top_k = top_k or settings.retrieval_top_k
    k = settings.retrieval_rrf_k

    vector_hits = vector_search(db, chat_id, qvec, k=top_k)
    fts_hits = keyword_search(db, chat_id, query, k=top_k)

    merged = _rrf([vector_hits, fts_hits], k=k)[:top_k]
    return merged


def _rrf(
    ranked_lists: list[list[tuple[str, float]]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion of pre-sorted result lists.

    Each item's contribution is ``1 / (k + rank)`` where rank is 0-indexed.
    A real reranker could be swapped in here (its interface would be
    ``rerank(query, candidates, top_k) -> list[id, score]``).
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (cid, _score) in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])