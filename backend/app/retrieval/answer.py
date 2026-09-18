"""Build a cited answer from retrieved sessions/chunks.

This is the retrieve-then-answer core. Retrieval is the Phase-2 extractive
pipeline: it pulls supporting chunks (or time-range sessions), quotes relevant
snippets, and attaches structured citations (message_id, sender, timestamp, chat)
so any downstream code or human can trace the source.

Answer generation is swappable via ``ANSWER_PROVIDER``:

* ``extractive`` (default, offline) - returns the Phase-2 bullet list verbatim.
  No API key, no LLM call.
* ``gemini`` - the retrieved content/citations are passed to Gemini
  (default ``gemini-2.5-flash``, overridable with ``ANSWER_MODEL``) which
  synthesizes a natural-language answer. The retrieval and the structured
  ``citations``/``sources`` lists are unchanged - only ``answer_text`` is
  synthesized.

The ``answer_text``/``sources`` return contract is stable across providers, so
``routes_ask.py`` needs no changes.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, Message, Session as ChatSession
from app.retrieval.search import hybrid_search


# ---------------------------------------------------------------------------
# Answer generation: extractive (default) vs. Gemini synthesis
# ---------------------------------------------------------------------------

def _extractive_answer(question: str, answer_text: str) -> str:
    """Return the Phase-2 extractive answer text unchanged (offline default)."""
    return answer_text


def _gemini_synthesize(question: str, answer_text: str) -> str:
    """Synthesize ``answer_text`` over the retrieved context with Gemini.

    The ``answer_text`` argument carries the retrieved chunks/citations as an
    extractive bullet list; we pass it as context and ask Gemini for a concise,
    grounded answer instead. Falls back to the extractive text on any failure
    (no key, network, API error) so /ask keeps working offline.
    """
    api_key = settings.gemini_api_key
    if not api_key:
        return answer_text
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return answer_text

    prompt = (
        "You are summarizing a WhatsApp group conversation history. "
        "Using ONLY the context below, answer the user's question in a concise "
        "natural-language reply (a few sentences). Stay grounded in the context; "
        "do not invent facts. If the context has no relevant information, say so.\n\n"
        f"CONTEXT:\n{answer_text}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=settings.answer_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=500,
            ),
        )
        return (response.text or "").strip() or answer_text
    except Exception:
        return answer_text


def _apply_answer_provider(question: str, answer_text: str) -> str:
    """Route ``answer_text`` through the configured ANSWER_PROVIDER."""
    provider = settings.answer_provider
    if provider == "gemini":
        return _gemini_synthesize(question, answer_text)
    return _extractive_answer(question, answer_text)


# ---------------------------------------------------------------------------
# 1. Pre-retrieval route classifier: time-range vs. semantic
# ---------------------------------------------------------------------------
# This is an *explicit* pre-retrieval routing step (rule-based, swappable).
# It inspects the raw question and returns which retrieval strategy to run:
#   "time_range"  -> pull sessions in a resolved date range directly from Postgres
#   "semantic"    -> hybrid_search over the full session corpus

_TIME_ROUTE_RE = re.compile(
    r"\b"
    r"what (?:did i miss|happened)"
    r"|what's been going on"
    r"|what's the latest"
    r"|tell me about what"
    r"|catch ?me up"
    r"|what happened last night|today|this (?:morning|afternoon|evening)|tonight"
    r"|this (?:week|month|year)"
    r"|yesterday|last (?:week|month|year|night|few days|24 hours|hours|minutes)"
    r"|past (?:week|month|year|days|hours|minutes|\d+\s*(?:days?|weeks?|hours?))"
    r"|since (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|since \w+"
    r"|(?:\d{1,2}/\d{1,2}/\d{2,4})"  # literal dates like 11/12/24
    r"\b",
    re.IGNORECASE,
)


def route_question(question: str) -> str:
    """Return ``"time_range"`` if the question is temporal, else ``"semantic"``.

    This is the *explicit* step the spec asked for: classify before retrieving.
    The implementation is intentionally simple (regex over known phrases + dates)
    and is meant to be replaced with an LLM classifier or vector-similarity
    route when that is available.
    """
    return "time_range" if _TIME_ROUTE_RE.search(question) else "semantic"


# ---------------------------------------------------------------------------
# 2. Resolve the natural-language date range into concrete datetimes
# ---------------------------------------------------------------------------

def _parse_time_range(
    question: str,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Best-effort extraction of a (start, end) UTC window from the question.

    Returns (start_utc, end_utc). When the question mentions an absolute
    date it is interpreted as 00:00..23:59 of that day. When only a
    relative window is named (this week / last month), ``now`` is used
    as the anchor. When parsing fails, a conservative 48-hour lookback
    is returned.
    """
    now = now or datetime.now(timezone.utc)
    q = question.lower().strip()

    def _floor_day(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    def _ceil_day(dt: datetime) -> datetime:
        return _floor_day(dt) + timedelta(days=1) - timedelta(seconds=1)

    def _start_of_week(dt: datetime) -> datetime:
        # Monday as start of week (Python default).
        return _floor_day(dt - timedelta(days=dt.weekday()))

    # absolute date like 11/12/24 / 11/12/2024
    abs_match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", q)
    if abs_match:
        parts = abs_match.group(1).split("/")
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
        if year < 100:
            year += 2000 if year >= 70 else 1900
        try:
            d = datetime(year, month, day, tzinfo=timezone.utc)
            return _floor_day(d), _ceil_day(d)
        except ValueError:
            pass

    # --- relative phrases ---
    if re.search(r"\byesterday\b", q):
        d = now - timedelta(days=1)
        return _floor_day(d), _ceil_day(d)

    if re.search(r"\btoday\b|\btonight\b|\bthis (?:morning|afternoon|evening)\b", q):
        return _floor_day(now), now

    if re.search(r"\bthis week\b", q):
        return _start_of_week(now), now

    if re.search(r"\bthis month\b", q):
        return _floor_day(now.replace(day=1)), now

    if re.search(r"\bthis year\b", q):
        return _floor_day(now.replace(month=1, day=1)), now

    if re.search(r"\blast week\b", q):
        return _start_of_week(now - timedelta(weeks=1)), _start_of_week(now)

    if re.search(r"\blast month\b", q):
        prev = (now.month - 2) % 12 + 1
        prev_year = now.year if now.month > 1 else now.year - 1
        start = datetime(prev_year, prev, 1, tzinfo=timezone.utc)
        return _floor_day(start), _floor_day(now.replace(day=1))

    if re.search(r"\blast year\b", q):
        start = datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)
        return _floor_day(start), _floor_day(now.replace(month=1, day=1))

    if re.search(r"\blast night\b", q):
        return _floor_day(now - timedelta(days=1)) + timedelta(hours=22), now.replace(hour=6)

    if re.search(r"\bthis morning\b", q):
        return _floor_day(now), now.replace(hour=12)

    if re.search(r"\blast (?:few days|24 hours|hours|minutes)\b|\bpast (?:week|month|days|hours|minutes)\b", q):
        # Default a short lookback.
        return now - timedelta(days=3), now

    # catch-all ("what did I miss", "what happened", "lately", "recently")
    return now - timedelta(days=2), now


# ---------------------------------------------------------------------------
# 3. Time-range answer builder
# ---------------------------------------------------------------------------

def _answer_time_range(
    db: Session,
    chat_id: str,
    question: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Answer by pulling sessions that overlap the resolved date range."""
    start, end = _parse_time_range(question)

    overlap = (
        ChatSession.chat_id == chat_id,
        ChatSession.started_at <= end,
        ChatSession.ended_at >= start,
    )
    sessions = list(
        db.execute(
            select(ChatSession)
            .where(*overlap)
            .order_by(ChatSession.started_at)
            .limit(50)
        ).scalars().all()
    )

    citations: list[dict[str, Any]] = []
    bullets: list[str] = []

    if not sessions:
        return (
            f"Nothing was found in the history for this chat between "
            f"{start:%Y-%m-%d %H:%M}Z and {end:%Y-%m-%d %H:%M}Z.",
            citations,
            [],
        )

    all_session_ids = []
    for i, session in enumerate(sessions, start=1):
        msg_ids = [id_str for id_str in (session.message_ids or [])]
        if msg_ids:
            rows = db.execute(
                select(Message).where(Message.id.in_(msg_ids)).order_by(Message.timestamp)
            ).scalars().all()
        else:
            rows = []
        sample_msgs = [m for m in rows if m.body and m.msg_type != "system"][:3]
        bullets.append(
            f"• **{session.header_text}** — {len(rows)} messages."
        )
        for m in sample_msgs:
            preview = (m.body or "").replace("\n", " ")[:200]
            bullet = f"   – *\"{preview}\"* ({m.sender_name}, {m.timestamp.strftime('%Y-%m-%d %H:%M') if m.timestamp else 'unknown'})"
            bullets.append(bullet)
            citations.append(
                {
                    "message_id": str(m.id),
                    "sender_name": m.sender_name,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    "chat_id": chat_id,
                    "chat_name": m.chat_name,
                    "preview": preview,
                }
            )
        all_session_ids.append(str(session.id))

    header = (
        f"Here's what happened in this chat from {start:%Y-%m-%d} to {end:%Y-%m-%d} "
        f"({len(sessions)} sessions):\n\n"
    )
    return header + "\n".join(bullets), citations, [{"session_id": s_id} for s_id in all_session_ids]


# ---------------------------------------------------------------------------
# 4. Semantic answer builder
# ---------------------------------------------------------------------------

def _answer_semantic(
    db: Session,
    chat_id: str,
    question: str,
    qvec: list[float],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    hits = hybrid_search(db, chat_id, question, qvec, top_k=settings.retrieval_top_k)
    if not hits:
        return (
            "I couldn't find anything directly relevant to that question in the "
            "group history. Try broadening your question or asking about a specific "
            "date or topic.",
            [],
            [],
        )

    chunk_ids = [cid for cid, _ in hits]
    score_map = {cid: score for cid, score in hits}
    chunks = db.execute(
        select(Chunk).where(Chunk.id.in_(chunk_ids))
    ).scalars().all()
    chunks_by_id = {str(c.id): c for c in chunks}

    citations: list[dict[str, Any]] = []
    source_list: list[dict[str, Any]] = []
    bullets: list[str] = []

    for rank, chunk_id in enumerate(chunk_ids, start=1):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        source_msgs = []
        if chunk.message_ids:
            source_msgs = db.execute(
                select(Message).where(Message.id.in_(chunk.message_ids)).order_by(Message.timestamp)
            ).scalars().all()
        # First content message as the quote; full source list as structured citations.
        preview_msg = next(
            (m for m in source_msgs if m.body and m.msg_type != "system"), None
        )
        preview = (preview_msg.body or "").replace("\n", " ")[:250] if preview_msg else (chunk.content or "")[:250]
        bullets.append(
            f"[{rank}] *\"{preview}\"* "
            f"— {preview_msg.sender_name}, {preview_msg.timestamp.strftime('%Y-%m-%d') if preview_msg and preview_msg.timestamp else 'unknown date'}, "
            f"score: {score_map.get(chunk_id, 0):.3f}"
        )
        for m in source_msgs:
            if m.body:
                citations.append(
                    {
                        "message_id": str(m.id),
                        "sender_name": m.sender_name,
                        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                        "chat_id": chat_id,
                        "chat_name": m.chat_name,
                        "preview": (m.body or "")[:250],
                    }
                )
        source_list.append(
            {
                "chunk_id": chunk_id,
                "session_id": str(chunk.session_id),
                "score": score_map.get(chunk_id, 0.0),
                "message_ids": chunk.message_ids or [],
                "content_preview": (chunk.content or "")[:500],
            }
        )

    header = f"Based on the group history, here's what I found about **\"{question}\"**:\n\n"
    return header + "\n".join(bullets), citations, source_list


# ---------------------------------------------------------------------------
# 5. Top-level orchestrator
# ---------------------------------------------------------------------------

def answer_question(
    db: Session,
    chat_id: str,
    question: str,
    embedding_provider=None,
) -> dict[str, Any]:
    """Answer a question against the group's embedded history.

    The ``embedding_provider`` is only needed to compute the vector for the
    semantic path; when the question routes to time_range it is ignored.
    """
    route = route_question(question)

    if route == "time_range":
        answer_text, citations, sources = _answer_time_range(db, chat_id, question)
        return {
            "chat_id": chat_id,
            "question": question,
            "route": route,
            "answer_text": _apply_answer_provider(question, answer_text),
            "citations": citations,
            "sources": sources,
        }

    # Semantic path: embed the question, then hybrid_search.
    if embedding_provider is None:
        from app.ingest.embedder import get_embedder
        embedding_provider = get_embedder()
    qvec = embedding_provider.embed([question])[0]
    answer_text, citations, sources = _answer_semantic(db, chat_id, question, qvec)
    return {
        "chat_id": chat_id,
        "question": question,
        "route": route,
        "answer_text": _apply_answer_provider(question, answer_text),
        "citations": citations,
        "sources": sources,
    }