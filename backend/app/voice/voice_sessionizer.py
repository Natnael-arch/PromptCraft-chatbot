"""Turn a diarized transcript into chunk-ready segments for embedding (Phase 3).

Text chat logs and diarized call transcripts share the same embedding problem:
raw short utterances ("yeah", "agreed", "what's the ETA?") do not embed usefully
in isolation. Phase 2 solved it by grouping messages into sessions and
prepending a synthesized context header to each chunk; voice follows the same
recipe but the "session" is implicit (one recording is one continuous session):

* a per-chunk context header (speakers + rough topic + time window),
* chunking at segment boundaries under the SAME char budget Phase 2 uses
  (``session_max_chunk_chars``), with a speaker-turn hard cap mirroring
  ``session_max_messages`` (``voice_segments_per_chunk``),
* each chunk remembers which diarized segments it was built from so /ask can
  cite "Speaker 2, 04:12-04:38" instead of a message timestamp range.

The embedding itself is Phase 2's embedder.py - nothing is duplicated here.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from app.config import settings

_STOPWORDS = frozenset(
    """a an and are at be been but by can can't did do for from had has have he her
    here his how i i'm if in is it its just like me my no not of on one or our out
    she so some that the their them there they this to up us was we were what when
    where which who why will with you your yes ok yeah well really""".split()
)
_TOPIC_WORD_RE = re.compile(r"[a-z]{3,}")

# Format the header exactly like this every time - answer.py parses the window
# and speaker list back out of the chunk header to build voice citations.
_HEADER_RE = re.compile(
    r"^Voice call · (?P<when>[^·]+) · (?P<duration>[^·]+) · "
    r"speakers: (?P<speakers>[^·]+) · topics: (?P<topics>.+)$"
)


@dataclass
class VoiceChunkBuild:
    """One embeddable piece of a recording: header + timestamped speaker lines."""

    content: str
    segments: list[dict]  # {speaker, start, end, text} refs for citation building


def format_timestamp(seconds: float) -> str:
    """MM:SS, or H:MM:SS past an hour (e.g. '04:12', '1:02:33')."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _speakers(segments: list[dict]) -> list[str]:
    seen: list[str] = []
    for seg in segments:
        name = (seg.get("speaker") or "unknown").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _topics(segments: list[dict], top_n: int = 3) -> list[str]:
    counts: dict[str, int] = {}
    for seg in segments:
        for word in _TOPIC_WORD_RE.findall((seg.get("text") or "").lower()):
            if word in _STOPWORDS or len(word) < 3:
                continue
            counts[word] = counts.get(word, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:top_n]


def build_voice_header(
    segments: list[dict],
    *,
    uploaded_at: datetime | None = None,
    duration_seconds: float | None = None,
) -> str:
    """One-line context header prepended to every chunk, Phase-2 style.

    Mirrors sessionizer.build_header's shape ("Session · when · n messages ·
    participants · topics") but for a single recording:
    "Voice call · 2026-09-18 21:04Z · 12:34 · speakers: Speaker 1, Speaker 2 ·
    topics: budget, launch".
    """
    when = f"{uploaded_at:%Y-%m-%d %H:%M}Z" if uploaded_at else "unknown time"
    dur = format_timestamp(duration_seconds) if duration_seconds else "--:--"
    speaker_line = ", ".join(_speakers(segments)) or "unknown"
    topics_line = ", ".join(_topics(segments)) or "unspecified"
    return (
        f"Voice call · {when} · {dur} · speakers: {speaker_line} · topics: {topics_line}"
    )


def _segment_line(seg: dict) -> str:
    speaker = (seg.get("speaker") or "unknown").strip()
    start = format_timestamp(float(seg.get("start", 0)))
    end = format_timestamp(float(seg.get("end", 0)))
    text = (seg.get("text") or "").strip().replace("\n", " ")
    return f"[{start}-{end}] {speaker}: {text}"


def build_voice_chunks(
    segments: list[dict],
    *,
    header: str,
    max_chunk_chars: int | None = None,
    segments_per_chunk: int | None = None,
) -> list[VoiceChunkBuild]:
    """Split a diarized transcript into chunks at segment boundaries.

    ``max_chunk_chars`` defaults to Phase 2's ``session_max_chunk_chars`` and
    ``segments_per_chunk`` to ``voice_segments_per_chunk`` so voice and text
    chunks stay aligned on insertion cost and search granularity.
    """
    max_chunk_chars = max_chunk_chars or settings.session_max_chunk_chars
    segments_per_chunk = segments_per_chunk or settings.voice_segments_per_chunk

    chunks: list[VoiceChunkBuild] = []
    current: list[dict] = []
    current_len = len(header) + 2

    def flush() -> None:
        nonlocal current, current_len
        if current:
            lines = [_segment_line(s) for s in current]
            chunks.append(
                VoiceChunkBuild(
                    content=header + "\n\n" + "\n".join(lines),
                    segments=list(current),
                )
            )
        current, current_len = [], len(header) + 2

    for seg in segments:
        line_len = len(_segment_line(seg))
        if line_len > max_chunk_chars:
            # Defensive: a single massive segment must not blow the char budget.
            seg = {**seg, "text": (seg.get("text") or "")[:max_chunk_chars]}
            line_len = max_chunk_chars
        over_char_budget = current and current_len + line_len > max_chunk_chars
        if over_char_budget or len(current) >= segments_per_chunk:
            flush()
        current.append(seg)
        current_len += line_len

    flush()
    return chunks


def parse_chunk_header(content: str) -> dict[str, str]:
    """Re-parse a voice chunk's header back into (when, duration, speakers, topics).

    Lets retrieval render a "Speaker 2, 04:12-04:38" citation from the chunk
    without a second DB round-trip. Mirrors the exact format build_voice_header
    emits; returns empty strings when the content is not a voice chunk (or the
    header was truncated), never raising.
    """
    match = _HEADER_RE.match(content or "")
    if not match:
        return {"when": "", "duration": "", "speakers": "", "topics": ""}
    return {
        "when": match.group("when"),
        "duration": match.group("duration"),
        "speakers": match.group("speakers"),
        "topics": match.group("topics"),
    }