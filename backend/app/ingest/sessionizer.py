"""Group chat logs into sessions, then chunk each session for embedding.

Chat logs break fixed-size-chunk RAG: short, context-free messages ("yeah", "lol",
"+1", "link?") do not embed usefully in isolation. So messages are grouped into
sessions first:

* a gap of N minutes of silence starts a new session (gap heuristic), and
* a hard cap on messages per session stops one continuously-active day from
  becoming a single un-embed-able blob.

Each session gets a synthesized one-line context header (participants, date range,
rough topic) that is PREPENDED to its chunk text before embedding - that is the fix
for short messages embedding poorly on their own.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

SessionId = str
MessageId = str

_STOPWORDS = frozenset(
    """a an and are at be been but by can can't did do for from had has have he her
    here his how i i'm if in is it its just like me my no not of on one or our out
    she so some that the their them there they this to up us was we were what when
    where which who why will with you your yes ok yeah well really""".split()
)
_TOPIC_WORD_RE = re.compile(r"[a-z]{3,}")


@dataclass
class ChunkBuild:
    """One embeddable piece of a session: context header + message lines."""

    content: str
    message_ids: list[MessageId]


@dataclass
class SessionBuild:
    """A computed session: metadata + the display lines that become chunks."""

    started_at: datetime
    ended_at: datetime
    message_ids: list[MessageId]  # ALL messages in the session (content + not)
    header_text: str
    chunks: list[ChunkBuild] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)


def _participants(messages: list[dict]) -> list[str]:
    seen: list[str] = []
    for m in messages:
        name = (m.get("sender_name") or "").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _topics(messages: list[dict], top_n: int = 3) -> list[str]:
    counts: dict[str, int] = {}
    for m in messages:
        for word in _TOPIC_WORD_RE.findall((m.get("body") or "").lower()):
            if word in _STOPWORDS or len(word) < 3:
                continue
            counts[word] = counts.get(word, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:top_n]


def _display_line(m: dict) -> str:
    ts = m.get("timestamp")
    stamp = ts.strftime("%H:%M") + "Z" if ts else "--:--"
    sender = (m.get("sender_name") or "?").strip()
    body = (m.get("body") or "").strip().replace("\n", " ")
    return f"[{stamp}] {sender}: {body}"


def build_header(session: list[dict], participants: list[str], topics: list[str]) -> str:
    start, end = session[0]["timestamp"], session[-1]["timestamp"]
    when = (
        f"{start:%Y-%m-%d %H:%M}Z - {end:%H:%M}Z"
        if start is not None
        else "unknown time"
    )
    header = (
        f"Session · {when} · {len(session)} messages · "
        f"participants: {', '.join(participants) or 'unknown'}"
    )
    if topics:
        header += f" · topics: {', '.join(topics)}"
    return header


def _split_chunks(
    header_text: str,
    lines: list[tuple[MessageId, str]],
    max_chunk_chars: int,
) -> list[ChunkBuild]:
    """Split session lines into chunks at MESSAGE boundaries under max_chunk_chars."""
    chunks: list[ChunkBuild] = []
    current_ids: list[MessageId] = []
    current_lines: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_ids, current_lines, current_len
        if current_lines:
            chunks.append(
                ChunkBuild(content=header_text + "\n\n" + "\n".join(current_lines), message_ids=list(current_ids))
            )
        current_ids, current_lines, current_len = [], [], 0

    for mid, line in lines:
        line_len = len(line)
        if line_len > max_chunk_chars:
            # Defensive: a single abnormally long message must not blow the budget.
            line = line[:max_chunk_chars]
            line_len = max_chunk_chars
        if current_lines and current_len + line_len > max_chunk_chars:
            flush()
        current_ids.append(mid)
        current_lines.append(line)
        current_len += line_len

    flush()
    return chunks


def sessionize(
    messages: list[dict],
    *,
    gap_minutes: int = 30,
    max_messages: int = 200,
    max_chunk_chars: int = 8000,
) -> list[SessionBuild]:
    """Group sorted (by timestamp) message records into SessionBuild objects.

    `messages` items are dicts with at least: id, sender_name, timestamp
    (aware datetime), body, msg_type, content (bool). Messages without a usable
    timestamp cannot be ordered and are skipped from sessionization (they are still
    stored in `messages` by the caller).
    """
    ordered = [m for m in messages if m.get("timestamp") is not None]
    ordered.sort(key=lambda m: m["timestamp"])
    sessions: list[SessionBuild] = []
    current: list[dict] = []

    def close() -> None:
        if not current:
            return
        participants = _participants(current)
        topics = _topics([m for m in current if m.get("content")])
        header_text = build_header(current, participants, topics)
        # Display lines only from content-bearing messages (skip system messages
        # and media-omitted events - they carry no embeddable text).
        all_ids = [m["id"] for m in current]
        content_ids = [m["id"] for m in current if m.get("content")]
        lines = [(m["id"], _display_line(m)) for m in current if m.get("content")]
        chunks = _split_chunks(header_text, lines, max_chunk_chars)
        sessions.append(
            SessionBuild(
                started_at=current[0]["timestamp"],
                ended_at=current[-1]["timestamp"],
                message_ids=all_ids,
                header_text=header_text,
                chunks=chunks,
                participants=participants,
            )
        )
        current.clear()

    for i, m in enumerate(ordered):
        previous = current[-1] if current else None
        gap_ok = previous is not None and (m["timestamp"] - previous["timestamp"]).total_seconds() / 60 > gap_minutes
        if gap_ok or len(current) >= max_messages:
            close()
        current.append(m)
    close()

    # Sessions are computed in reverse-intuitive order (later messages first when a
    # gap closes current); sort by started_at before returning.
    sessions.sort(key=lambda s: s.started_at)
    return sessions