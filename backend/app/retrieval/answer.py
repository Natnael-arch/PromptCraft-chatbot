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

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, Message, Recording, Session as ChatSession, TrustedSender
from app.retrieval.search import hybrid_search
from app.voice.voice_sessionizer import format_timestamp

logger = logging.getLogger(__name__)


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

    client = genai.Client(api_key=api_key)
    max_tokens = 2048  # Raised from 500 to prevent MAX_TOKENS truncation on list answers

    for attempt in range(2):
        if attempt == 0:
            prompt = (
                "You are summarizing a WhatsApp group conversation history. "
                "Using ONLY the context below, answer the user's question in a clear, concise "
                "natural-language reply (complete all bullet points/items without cutting off). "
                "Stay grounded in the context; do not invent facts. "
                "If the context has no relevant information, say so.\n\n"
                f"CONTEXT:\n{answer_text}\n\n"
                f"QUESTION: {question}\n\n"
                "ANSWER:"
            )
        else:
            # Concise-summary reframing on truncation retry
            prompt = (
                "You are summarizing a WhatsApp group conversation history. "
                "PROVIDE A VERY CONCISE SUMMARY listing all relevant items in a condensed format. "
                "Do not write overly verbose descriptions. Complete the reply without truncation.\n\n"
                f"CONTEXT:\n{answer_text}\n\n"
                f"QUESTION: {question}\n\n"
                "CONCISE ANSWER:"
            )

        try:
            response = client.models.generate_content(
                model=settings.answer_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=max_tokens,
                ),
            )

            candidate = response.candidates[0] if (response and response.candidates) else None
            finish_reason = str(getattr(candidate, "finish_reason", "")) if candidate else ""
            text = (response.text or "").strip()

            if finish_reason and finish_reason.upper() != "STOP" and "STOP" not in finish_reason.upper():
                logger.warning(
                    "Gemini answer synthesis truncated (finish_reason=%s, attempt=%d, len=%d).",
                    finish_reason, attempt + 1, len(text)
                )
                if attempt == 0:
                    continue  # Retry with concise reframing prompt
                logger.error(
                    "Gemini answer synthesis cut off on retry (finish_reason=%s). Falling back to extractive answer.",
                    finish_reason
                )
                return answer_text

            if text:
                return text
        except Exception:
            logger.exception("Gemini answer synthesis attempt %d failed", attempt + 1)
            if attempt == 0:
                continue

    return answer_text


def _apply_answer_provider(question: str, answer_text: str) -> str:
    """Route ``answer_text`` through the configured ANSWER_PROVIDER."""
    provider = settings.answer_provider
    if provider == "gemini":
        return _gemini_synthesize(question, answer_text)
    return _extractive_answer(question, answer_text)


# ---------------------------------------------------------------------------
# Voice citations: speaker + timestamp for voice-sourced chunks
# ---------------------------------------------------------------------------

def _distinct_speakers(segments: list[dict]) -> list[str]:
    """Distinct speaker labels in first-appearance order (used by time-range)."""
    seen: list[str] = []
    for seg in segments:
        name = (str(seg.get("speaker") or "")).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _voice_chunk_source(chunk: Chunk) -> dict[str, Any]:
    """Derive the voice citation fields ("Speaker 2, 04:12-04:38") for a chunk.

    Reads the diarized segments this chunk was built from (chunks.voice_segments)
    rather than re-parsing free text. Returns empty defaults for chunks that have
    no segment metadata so retrieval never crashes on malformed rows.
    """
    segments = chunk.voice_segments or []
    speakers: list[str] = []
    for seg in segments:
        name = (str(seg.get("speaker") or "")).strip()
        if name and name not in speakers:
            speakers.append(name)
    starts = [float(seg.get("start", 0) or 0) for seg in segments]
    ends = [float(seg.get("end", 0) or 0) for seg in segments]
    seg_start = min(starts) if starts else None
    seg_end = max(ends) if ends else None
    primary = speakers[0] if speakers else None
    segment = None
    if primary is not None and seg_start is not None and seg_end is not None:
        segment = (
            f"{primary}, {format_timestamp(seg_start)}\u2013{format_timestamp(seg_end)}"
        )
    return {
        "speakers": speakers,
        "speaker": primary,
        "segment_start": seg_start,
        "segment_end": seg_end,
        "segment": segment,
    }


# ---------------------------------------------------------------------------
# Phase 5: trusted-sender citation labeling. Announcers/leads are flagged in
# citations so a reader can see at a glance that a source was an official
# announcement rather than a passing comment. Both answer builders (semantic +
# time-range) share these helpers so the two paths never diverge.
# ---------------------------------------------------------------------------

def _trusted_labels(db: Session) -> dict[str, dict[str, str | None]]:
    """sender_id -> {"display_name", "role_label"} from trusted_senders.

    Called once per answer; the table is tiny and PK-scoped by sender_id.
    """
    rows = db.execute(select(TrustedSender)).scalars().all()
    return {
        t.sender_id: {"display_name": t.display_name, "role_label": t.role_label}
        for t in rows
    }


def _cite_message(
    m: Message,
    chat_id: str,
    trusted: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    """Structured citation for one message, carrying its trust annotation.

    ``is_announcement``/``role_label`` are True/set when ``m.sender_id`` is a
    trusted sender; ``sender_id`` is included so clients can link the citation
    back to the trusted_senders table without another lookup.
    """
    label = trusted.get(m.sender_id)
    return {
        "message_id": str(m.id),
        "sender_name": m.sender_name,
        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
        "chat_id": chat_id,
        "chat_name": m.chat_name,
        "preview": (m.body or "")[:250],
        "sender_id": m.sender_id,
        "is_announcement": bool(label),
        "role_label": label["role_label"] if label else None,
    }


def _sender_label(m: Message | None, trusted: dict[str, dict[str, str | None]]) -> str:
    """Bullet-name for a message: display-name/role for trusted senders, else JID.

    Trusted senders render as "📢 [Amina, announcer]" instead of a bare pushname,
    so the announcement flag survives even in the raw extractive bullet text.
    """
    label = trusted.get(m.sender_id) if m is not None else None
    if label:
        name = label["display_name"] or m.sender_name or "unknown"
        return f"📢 [{name}, {label['role_label']}]"
    return (m.sender_name if m is not None else None) or "unknown"


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

    # Phase 3: also surface voice recordings uploaded within the range. The
    # recordings.created_at time is the upload time (the closest proxy we store
    # for "when the call happened"), so a "what did we decide on the call
    # yesterday" question reaches voice chunks too.
    recordings = list(
        db.execute(
            select(Recording)
            .where(
                Recording.chat_id == chat_id,
                Recording.created_at >= start,
                Recording.created_at <= end,
            )
            .order_by(Recording.created_at)
        ).scalars().all()
    )

    citations: list[dict[str, Any]] = []
    bullets: list[str] = []
    sources: list[dict[str, Any]] = []

    if not sessions and not recordings:
        return (
            f"Nothing was found in the history for this chat between "
            f"{start:%Y-%m-%d %H:%M}Z and {end:%Y-%m-%d %H:%M}Z.",
            citations,
            [],
        )

    trusted = _trusted_labels(db)

    # ---- voice recordings in range ----
    for rec in recordings:
        if rec.status != "done":
            continue
        raw = rec.raw_transcript_json or {}
        segs = raw.get("segments") or [] if isinstance(raw.get("segments"), list) else []
        speakers = _distinct_speakers(segs)
        voices = [s for s in segs if isinstance(s, dict)]
        previews = [
            f"   – *\"{(s.get('text') or '')[:200]}\"* "
            f"({s.get('speaker')}, {rec.created_at.strftime('%Y-%m-%d')})"
            for s in voices[:3]
        ]
        duration = (
            f"{rec.duration_seconds:.0f}s"
            if rec.duration_seconds is not None
            else "unknown length"
        )
        bullets.append(
            f"• **Voice call** — {rec.created_at.strftime('%Y-%m-%d')}, "
            f"{duration}, speakers: {', '.join(speakers) or 'unknown'}."
        )
        bullets.extend(previews)
        sources.append(
            {
                "chunk_id": None,
                "session_id": None,
                "source_type": "voice",
                "recording_id": str(rec.id),
                "score": None,
                "speaker": ", ".join(speakers) or None,
                "segment_start": None,
                "segment_end": None,
                "segment": None,
                "message_ids": [],
                "content_preview": "/".join(
                    (s.get("text") or "")[:200] for s in voices[:3]
                ),
            }
        )

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
            bullet = f"   – *\"{preview}\"* ({_sender_label(m, trusted)}, {m.timestamp.strftime('%Y-%m-%d %H:%M') if m.timestamp else 'unknown'})"
            bullets.append(bullet)
            citations.append(_cite_message(m, chat_id, trusted))
        sources.append(
            {
                "chunk_id": None,
                "session_id": str(session.id),
                "source_type": "text",
                "recording_id": None,
                "score": None,
                "speaker": None,
                "segment_start": None,
                "segment_end": None,
                "segment": None,
                "message_ids": msg_ids,
                "content_preview": None,
            }
        )

    counts = f"{len(sessions)} sessions" if sessions else "no text sessions"
    if recordings:
        # count only done recordings, matching the voice bullets emitted above
        done = sum(1 for r in recordings if r.status == "done")
        if done:
            counts += f", {done} voice recording{'s' if done > 1 else ''}"
    header = (
        f"Here's what happened in this chat from {start:%Y-%m-%d} to {end:%Y-%m-%d} "
        f"({counts}):\n\n"
    )
    return header + "\n".join(bullets), citations, sources


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

    trusted = _trusted_labels(db)

    for rank, chunk_id in enumerate(chunk_ids, start=1):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        if chunk.source_type == "voice":
            # Voice chunk: cite the exact moment, not a message range.
            voice = _voice_chunk_source(chunk)
            segs = chunk.voice_segments or []
            first_seg = next((s for s in segs if (s.get("text") or "").strip()), None)
            preview = (first_seg.get("text") or chunk.content or "")[:250]
            speaker_ref = voice["segment"] or voice["speaker"] or "unknown"
            bullets.append(
                f"[{rank}] *\"{preview}\"* "
                f"— {speaker_ref}, score: {score_map.get(chunk_id, 0):.3f}"
            )
            source_list.append(
                {
                    "chunk_id": chunk_id,
                    "session_id": None,
                    "source_type": "voice",
                    "recording_id": str(chunk.recording_id) if chunk.recording_id else None,
                    "score": score_map.get(chunk_id, 0.0),
                    "speaker": voice["speaker"],
                    "segment_start": voice["segment_start"],
                    "segment_end": voice["segment_end"],
                    "segment": voice["segment"],
                    "message_ids": [],
                    "content_preview": (chunk.content or "")[:500],
                }
            )
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
        chunk_text = (chunk.content or "").strip()
        bullets.append(
            f"[{rank}] {chunk_text}\n"
            f"— {_sender_label(preview_msg, trusted)}, {preview_msg.timestamp.strftime('%Y-%m-%d') if preview_msg and preview_msg.timestamp else 'unknown date'}, "
            f"score: {score_map.get(chunk_id, 0):.3f}"
        )
        for m in source_msgs:
            if m.body:
                citations.append(_cite_message(m, chat_id, trusted))
        source_list.append(
            {
                "chunk_id": chunk_id,
                "session_id": str(chunk.session_id),
                "source_type": "text",
                "recording_id": None,
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


# ---------------------------------------------------------------------------
# 6. Casual/banter mode (Phase 5). A second, lighter personality for the group:
# jokes and nonsense get a short in-character reply instead of being force-fed
# through retrieval. The retrieval/answer code above is untouched - reply_worker
# routes here only after classify_intent() says "banter".
# ---------------------------------------------------------------------------

BANTER_FALLBACK = "lol"

_BANTER_SYSTEM_PROMPT = (
    "You are replying inside a real WhatsApp group chat among real people in a "
    "hackathon cohort. Your tone here is playful, witty, occasionally sarcastic \u2014 "
    "like a clever group member, not a customer support bot.\n"
    "- Never make a joke that targets, mocks, or singles out a specific named "
    "person in the group, even lightly. General, self-deprecating, or absurdist "
    "humor only.\n"
    "- Never joke about anything outside this program's context \u2014 no political, "
    "religious, or otherwise sensitive topics. When in doubt, keep it about the "
    "hackathon, coding, deadlines, chat culture, or something absurd and unrelated "
    "to anyone present.\n"
    "- Keep it short \u2014 one or two sentences, this is a chat message not a monologue.\n"
    "- You are NOT answering a factual question here; do not state anything as fact "
    "about the program, deadlines, or announcements even in jest. If the message "
    "actually seems to be a real question in disguise, say so lightly and suggest "
    "they ask directly instead of guessing.\n"
)


def get_recent_context(db: Session, chat_id: str, limit: int = 15) -> str:
    """Chronological "Sender: message" lines from the chat's raw message history.

    Used ONLY to give banter mode a sense of the room (tone/context). It comes
    straight from `messages`, deliberately bypassing retrieval, and is not a
    source of facts - the banter system prompt states exactly that.
    """
    rows = db.execute(
        select(Message)
        .where(
            Message.chat_id == chat_id,
            Message.body.isnot(None),
            Message.body != "",
        )
        .order_by(Message.created_at.desc())
        .limit(limit)
    ).scalars().all()
    lines = []
    for m in reversed(rows):
        speaker = m.sender_name or m.sender_id or "unknown"
        lines.append(f"{speaker}: {m.body}")
    return "\n".join(lines)


def _banter_reply(question: str, recent_context: str) -> str:
    """One/two-sentence in-character reply for a non-knowledge message.

    Mirrors the defensive shape of ``_gemini_synthesize``: no key, missing SDK,
    or any API failure drops to ``BANTER_FALLBACK`` ("lol") so a stuck group can
    never stop the bot, and the fallback is deliberately never an apology.
    """
    api_key = settings.gemini_api_key
    if not api_key:
        return BANTER_FALLBACK
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return BANTER_FALLBACK

    prompt = (
        _BANTER_SYSTEM_PROMPT
        + "\n"
        "Recent chat (for tone/context only \u2014 NEVER quote or state anything from "
        "here as fact):\n"
        f"{recent_context or '(nothing yet)'}\n\n"
        f"LATEST MESSAGE: {question}\n\n"
        "YOUR REPLY:"
    )
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=settings.answer_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.9,
                max_output_tokens=512,  # Raised from 120
            ),
        )
        candidate = response.candidates[0] if (response and response.candidates) else None
        finish_reason = str(getattr(candidate, "finish_reason", "")) if candidate else ""
        text = (response.text or "").strip()

        if finish_reason and finish_reason.upper() != "STOP" and "STOP" not in finish_reason.upper():
            logger.warning("Banter reply truncated (finish_reason=%s). Falling back.", finish_reason)
            return BANTER_FALLBACK

        return text or BANTER_FALLBACK
    except Exception:
        return BANTER_FALLBACK