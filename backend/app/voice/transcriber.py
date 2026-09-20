"""Transcribe uploaded audio into diarized, timestamps-segmented text via Gemini.

WhatsApp voice notes are Ogg-Opus containers, so the mime allow-list is the set
Gemini's audio-adder tier accepts (see config.py for the researched limits):

  audio/ogg (WhatsApp .ogg/.opus/.oga), audio/mpeg|mp3|mpga,
  audio/mp4|m4a (call recordings exported as .m4a), audio/wav, audio/webm,
  audio/x-aac, audio/flac, audio/pcm

Not every format that could *exist* on disk is worth sending to Gemini: the
route validates the mime type up-front and rejects anything outside this list
with a clear error instead of letting a garbage/unsupported file burn a billing
request.

Transcription strategy (model: ``TRANSCRIBE_MODEL``, default gemini-2.5-flash):

* Files <= 20MB are sent inline via ``Part.from_bytes``; larger files go through
  the Files API (``client.files.upload``) and are referenced per request, since
  Gemini caps inline request bodies at 20MB.
* When a duration is knowable (mutagen) and exceeds ``TRANSCRIBE_WINDOW_SECONDS``,
  the recording is transcribed in fixed windows by asking for each window's
  chunk on the same timeline, then merged. Window boundaries are well under the
  ~8.4h / ~1M-token single-request limit so structured JSON stays reliable.
* Output is requested as strict JSON matching ``{segments: [{speaker, start, end,
  text}]}`` via the response_schema/response_mime_type parameters (not free-text
  parsing). Malformed/empty responses are an error with the raw text preserved so
  the caller can store it for debugging rather than silently dropping audio.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any

from app.config import settings

# Gemini inline requests are capped at 20MB total; above this, use the Files API.
FILES_API_MIN_BYTES = 20 * 1024 * 1024

# MIME types Gemini's audio tier accepts today (source: Gemini-pod model cards,
# 2026). Kept as the single source of truth for upload validation.
ALLOWED_MIME_TYPES = frozenset(
    {
        "audio/ogg",  # WhatsApp voice notes are Ogg-Opus
        "audio/opus",
        "audio/oga",
        "audio/mpeg",
        "audio/mp3",
        "audio/mpga",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/x-aac",
        "audio/flac",
        "audio/pcm",
    }
)

_EXTENSION_MIME = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".oga": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/x-aac",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


class AudioFormatError(ValueError):
    """Upload declared an unsupported/missing audio mime type."""


class TranscriptionError(RuntimeError):
    """Gemini returned malformed/empty transcript output.

    Carries the raw response (parsed dict, or raw text string) so the caller can
    persist it for debugging instead of letting it vanish.
    """

    def __init__(self, message: str, raw: Any = None):
        super().__init__(message)
        self.raw = raw


class TranscriptionRequestError(RuntimeError):
    """The Gemini API call failed (network, auth, quota, ...)."""


@dataclass
class TranscriptSegment:
    speaker: str
    start: float  # seconds from the start of the recording
    end: float
    text: str


@dataclass
class TranscriptResult:
    segments: list[TranscriptSegment]
    raw_response: dict  # verbatim Gemini output, for debugging / re-chunking
    duration_seconds: float | None
    windowed: bool = False


_TS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?$")


def _parse_timestamp(value: Any) -> float:
    """Parse 'MM:SS', 'H:MM:SS' or (defensively) bare float seconds into seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        m = _TS_RE.match(value)
        if m:
            h = int(m.group(1) or 0)
            mm = int(m.group(2))
            ss = int(m.group(3))
            frac = float("0." + m.group(4)) if m.group(4) else 0.0
            return h * 3600 + mm * 60 + ss + frac
        try:
            return float(value)
        except ValueError:
            pass
    raise TranscriptionError(
        f"Unparseable segment timestamp {value!r} (expected MM:SS or H:MM:SS)"
    )


def normalize_mime_type(declared: str | None, filename: str = "") -> str:
    """Return the canonical allowed mime type for an upload, or raise AudioFormatError.

    Uses the declared ``Content-Type`` when it is one of the allowed audio types,
    and falls back to the filename extension when the client sent a generic type
    (empty, ``application/octet-stream`` or ``application/ogg`` for WhatsApp's
    .opus/.ogg notes).
    """
    declared = (declared or "").split(";")[0].strip().lower()
    if declared in ALLOWED_MIME_TYPES:
        return declared

    ext = os.path.splitext(filename or "")[1].lower()
    fallback = _EXTENSION_MIME.get(ext)
    if declared in {"", "application/octet-stream", "application/ogg", "application/opus"} and fallback:
        return fallback

    raise AudioFormatError(
        f"Unsupported audio type {declared or '(none)'!r} for {filename or 'upload'}. "
        f"Supported: {', '.join(sorted(ALLOWED_MIME_TYPES))}. "
        "WhatsApp voice notes (.ogg/.opus), .mp3, .m4a, .wav and .webm are accepted."
    )


def read_duration(data: bytes) -> float | None:
    """Best-effort duration in seconds from audio container metadata (mutagen).

    Pure-Python and format-agnostic (Ogg/Opus, MP3, MP4/M4A, WAV, FLAC). Returns
    None when the format cannot be decoded here so the caller can proceed without
    a duration (single-shot transcription) instead of failing the upload.
    """
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None
    try:
        audio = MutagenFile(__import__("io").BytesIO(data))
        if audio is not None and getattr(audio, "info", None) is not None:
            return float(audio.info.length)
    except Exception:  # noqa: BLE001 - any undecodable file just yields None
        return None
    return None


def _sha256(data: bytes) -> str:
    """Idempotency key for an uploaded recording."""
    return hashlib.sha256(data).hexdigest()


def _segment_schema() -> dict:
    """Strict JSON schema: {segments: [{speaker, start, end, text}]}."""
    return {
        "type": "OBJECT",
        "properties": {
            "segments": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "speaker": {"type": "STRING"},
                        "start": {"type": "STRING"},  # "MM:SS" / "H:MM:SS"
                        "end": {"type": "STRING"},
                        "text": {"type": "STRING"},
                    },
                    "required": ["speaker", "start", "end", "text"],
                },
            }
        },
        "required": ["segments"],
    }


def _transcribe_prompt(window: tuple[float, float] | None) -> str:
    prompt = (
        "Transcribe all speech in the attached audio recording into diarized "
        "segments. Requirements:\n"
        "1. Assign each distinct speaker a stable label: 'Speaker 1', 'Speaker 2', "
        "etc. Use the same label for the same voice for the whole recording.\n"
        "2. Every segment has a start and end timestamp as 'MM:SS' (use 'H:MM:SS' "
        "for parts past an hour).\n"
        "3. Timestamps are relative to the BEGINNING of the recording (00:00 = "
        "start of the recording).\n"
        "4. Include every word spoken; do not summarize.\n"
    )
    if window is not None:
        start, end = window
        prompt += (
            f"5. You must only transcribe the portion from "
            f"{_fmt(start)} to {_fmt(end)}; ignore anything outside that range.\n"
        )
    prompt += 'Return a JSON object shaped exactly like {"segments": [{"speaker": ..., "start": ..., "end": ..., "text": ...}]}.'
    return prompt


def _fmt(seconds: float) -> str:
    """Format seconds as MM:SS (or H:MM:SS past one hour) for prompts."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_segments(raw: dict) -> list[TranscriptSegment]:
    """Validate Gemini's JSON response and coerce to TranscriptSegment objects."""
    segments_raw = raw.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        raise TranscriptionError(
            "Gemini returned no diarized segments (shape was not "
            '{"segments": [{speaker, start, end, text}]}).',
            raw=raw,
        )
    segments: list[TranscriptSegment] = []
    for i, item in enumerate(segments_raw):
        if not isinstance(item, dict):
            raise TranscriptionError(f"Segment {i} is not an object: {item!r}", raw=raw)
        for key in ("speaker", "start", "end", "text"):
            if key not in item:
                raise TranscriptionError(f"Segment {i} missing required key {key!r}", raw=raw)
        text = str(item["text"]).strip()
        if not text:
            raise TranscriptionError(f"Segment {i} has empty text", raw=raw)
        segments.append(
            TranscriptSegment(
                speaker=str(item["speaker"]).strip() or "unknown",
                start=_parse_timestamp(item["start"]),
                end=_parse_timestamp(item["end"]),
                text=text,
            )
        )
    segments.sort(key=lambda s: s.start)
    return segments


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


class GeminiTranscriber:
    """Transcribe audio via the google-genai SDK (lazy-imported, like Phase 3's
    embedder) so the module imports cleanly without the SDK installed."""

    name: str = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        window_seconds: int | None = None,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.transcribe_model
        self.window_seconds = window_seconds or settings.transcribe_window_seconds
        self._client = self._make_client() if self.api_key else None

    def _make_client(self):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - guarded at runtime
            raise TranscriptionRequestError(
                "google-genai is not installed; voice transcription needs it "
                "('pip install google-genai', already in requirements.txt)."
            ) from exc
        return genai.Client(api_key=self.api_key)

    def transcribe(
        self,
        data: bytes,
        mime_type: str,
        *,
        filename: str = "voice_note",
    ) -> TranscriptResult:
        """Transcribe raw audio bytes into diarized segments (single-or-windowed)."""
        if not data:
            raise AudioFormatError("Empty audio upload")
        mime_type = normalize_mime_type(mime_type, filename)
        if not self._client:
            raise TranscriptionRequestError(
                "GEMINI_API_KEY is not set - voice transcription requires a real "
                "Gemini key (there is no offline mock for audio understanding)."
            )

        duration = read_duration(data)
        use_files = len(data) > FILES_API_MIN_BYTES

        if duration is not None and duration > self.window_seconds:
            segments, raw = self._transcribe_windows(
                data, mime_type, duration, use_files=use_files
            )
            return TranscriptResult(
                segments=segments,
                raw_response=raw,
                duration_seconds=duration,
                windowed=True,
            )

        segments, raw = self._transcribe_once(data, mime_type, use_files=use_files)
        return TranscriptResult(
            segments=segments,
            raw_response=raw,
            duration_seconds=duration,
            windowed=False,
        )

    # ------------------------------------------------------------------ internals

    def _upload(self, data: bytes, mime_type: str):
        """Host a large file on the Files API; returns something with .uri."""
        try:
            from google.genai import types as genai_types
            from google.genai import http_options
        except ImportError:  # pragma: no cover
            raise TranscriptionRequestError("google-genai is not installed")
        suffix = os.path.splitext(_MIME_TO_EXT.get(mime_type, ".ogg"))[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            file = self._client.files.upload(path=tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return genai_types.Part.from_uri(file_uri=file.uri, mime_type=mime_type)

    def _request(self, part: Any, window: tuple[float, float] | None) -> dict:
        """One generate_content call -> validated raw {segments: [...]} dict."""
        from google.genai import types as genai_types

        response = self._client.models.generate_content(
            model=self.model,
            contents=[_transcribe_prompt(window), part],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_segment_schema(),
                temperature=0.2,
                max_output_tokens=16384,
            ),
        )
        text = getattr(response, "text", None) or ""
        text = _strip_code_fence(text)
        if not text:
            raise TranscriptionRequestError(
                "Gemini returned an empty transcription response"
            )
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TranscriptionError(
                f"Gemini returned non-JSON despite response_schema: {text[:300]!r}",
                raw={"raw_text": text},
            ) from exc
        return raw

    def _transcribe_once(
        self,
        data: bytes,
        mime_type: str,
        *,
        use_files: bool,
    ) -> tuple[list[TranscriptSegment], dict]:
        """Single-request transcription (short files, or Files-API for >20MB)."""
        part = self._upload(data, mime_type) if use_files else self._part_inline(data, mime_type)
        raw = self._request(part, window=None)
        return _parse_segments(raw), raw

    def _transcribe_windows(
        self,
        data: bytes,
        mime_type: str,
        duration: float,
        *,
        use_files: bool,
    ) -> tuple[list[TranscriptSegment], dict]:
        """Windowed transcription of long recordings, merged into one timeline.

        Each window request reuses the same audio source (inline part re-created
        per call, or a single Files-API upload referenced by URI), asks Gemini to
        transcribe only that window's time span on the recording's own timeline,
        then merges segments sorted by start. A defensive fallback offsets
        segments that come back window-relative when a model ignores the
        absolute-timeline instruction.
        """
        part = self._upload(data, mime_type) if use_files else None
        window_seconds = self.window_seconds
        # ceil(duration / window) so the final partial window (e.g. 90-100s of
        # a 100s recording at 30s windows) is included.
        num_windows = max(1, math.ceil(duration / window_seconds))
        windows = [
            (i * window_seconds, min((i + 1) * window_seconds, duration))
            for i in range(num_windows)
        ]

        merged: list[TranscriptSegment] = []
        raw_records: dict[str, Any] = {"windows": []}
        for start, end in windows:
            source = part if part is not None else self._part_inline(data, mime_type)
            raw = self._request(source, window=(start, end))
            segs = _parse_segments(raw)
            self._offset_window_relative(segs, start, end)
            merged.extend(segs)
            raw_records["windows"].append({"window": [start, end], "response": raw})

        raw_records["segments"] = [
            {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
            for s in merged
        ]
        merged.sort(key=lambda s: s.start)
        if not merged:
            raise TranscriptionError(
                f"Windowed transcription produced no segments across {len(windows)} windows"
            )
        return merged, raw_records

    def _offset_window_relative(
        self, segments: list[TranscriptSegment], start: float, end: float
    ) -> None:
        """Defensive: nudge window-relative timestamps onto the recording timeline."""
        slack = 5.0
        max_end = max((s.end for s in segments), default=0.0)
        if max_end <= (end - start) + slack and start > 0:
            for s in segments:
                s.start += start
                s.end += start

    @staticmethod
    def _part_inline(data: bytes, mime_type: str):
        try:
            from google.genai import types as genai_types
        except ImportError:  # pragma: no cover
            raise TranscriptionRequestError("google-genai is not installed")
        return genai_types.Part.from_bytes(data=data, mime_type=mime_type)


def get_transcriber(**overrides) -> GeminiTranscriber:
    """Factory mirroring embedder.get_embedder for the voice phase."""
    return GeminiTranscriber(
        api_key=overrides.get("api_key", settings.gemini_api_key),
        model=overrides.get("model", settings.transcribe_model),
        window_seconds=overrides.get(
            "window_seconds", settings.transcribe_window_seconds
        ),
    )


# Mirrors _EXTENSION_MIME lightly so _upload can give tempfiles a sane suffix.
_MIME_TO_EXT = {v: k for k, v in _EXTENSION_MIME.items()}