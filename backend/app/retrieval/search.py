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

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, Recording, Session as ChatSession

logger = logging.getLogger(__name__)


def _fts_condition(query: str):
    """plainto_tsquery handles unaccented English well enough for group chat."""
    return func.to_tsvector("english", Chunk.content).op("@@")(
        func.plainto_tsquery("english", query)
    )


def _chunk_scope(chat_id: str):
    """Single scope predicate covering BOTH source types of the chunks table.

    A chunk is retrievable when it belongs to this chat either as a text chunk
    (parent `sessions` row carries chat_id) or a voice chunk (parent `recordings`
    row carries chat_id). There is deliberately no per-source branching here:
    voice and text chunks are ordinary rows in `chunks`, so both queries below
    just LEFT JOIN the two parents and apply this one predicate - which also
    guarantees no half-finished row (a chunk with neither FK set) ever leaks into
    retrieval results.
    """
    return or_(
        and_(Chunk.session_id.isnot(None), ChatSession.chat_id == chat_id),
        and_(Chunk.recording_id.isnot(None), Recording.chat_id == chat_id),
    )


def vector_search(
    db: Session,
    chat_id: str,
    qvec: list[float],
    *,
    k: int,
) -> list[tuple[str, float]]:
    """Top-k nearest neighbors by cosine distance over the chunk embedding column.

    Returns [(chunk_id_str, similarity_score)] sorted descending. Includes text
    AND voice chunks for the chat via ``_chunk_scope``.
    """
    distance = Chunk.embedding.cosine_distance(qvec)
    score_expr = (1 - distance).label("score")
    stmt = (
        select(Chunk.id.label("id"), score_expr)
        .join(ChatSession, Chunk.session_id == ChatSession.id, isouter=True)
        .join(Recording, Chunk.recording_id == Recording.id, isouter=True)
        .where(_chunk_scope(chat_id))
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
    Includes text AND voice chunks for the chat via ``_chunk_scope``.
    """
    query = query.strip()
    if not query:
        return []

    cd = func.ts_rank_cd(
        func.to_tsvector("english", Chunk.content),
        func.plainto_tsquery("english", query),
        # rank_cd normalization by document length.
        # Passed POSITIONALLY: Postgres ts_rank_cd(vector, query, normalization)
        # does not accept a `normalization=` keyword through SQLAlchemy's generic
        # func (it raises TypeError at statement-build time), so this is the form
        # that actually reaches Postgres. 32 == rank / (rank + 1).
        32,
    ).label("score")

    stmt = (
        select(Chunk.id.label("id"), cd)
        .join(ChatSession, Chunk.session_id == ChatSession.id, isouter=True)
        .join(Recording, Chunk.recording_id == Recording.id, isouter=True)
        .where(
            _chunk_scope(chat_id),
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