"""Parse WhatsApp "Export chat" files into record dicts shaped like live messages.

WhatsApp's exported .txt differs by OS, locale and WhatsApp version:

* Android (US, MM/DD/YY):  `[11/12/2024, 10:15:30 AM] Alice: hey`
* Android (intl, DD/MM/YY): `[12/11/24, 22:15:30] Alice: hey`
* iOS (bracket, 12h):       `[11/12/2024, 10:15:30\u202FAM] Alice: hey`  (U+202F narrow nbsp)
* iOS (dash separator):     `16/09/2026, 21:35:47 \u2013 Alice: hey`
* ... and locally-adapted variants of all of the above.

We therefore never assume one fixed format: the line-start pattern is auto-detected
from the first ~50 content lines, and every line that does NOT start a new message
is a continuation line and is re-attached to the previous message's body.

Each exported line becomes a record in the same shape as a live-captured message
(sender_name, body, timestamp, ...) so backfill and live messages unify downstream.
There is no WAHA id for these, so we synthesize a stable dedup key
(chat_id, sender_name, timestamp, body) that makes re-importing the same file
idempotent.
"""

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# --- Message typing ------------------------------------------------------------------
# Messages that carry chat content the embedder should actually embed.
SYSTEM_TYPE = "system"
MEDIA_TYPE = "media"  # generic "<Media omitted>" we cannot classify further

_SAMPLE_LINES = 50
_ZIP_MAGIC = b"PK\x03\x04"


def _dt_format_re_time(am_pm: bool) -> str:
    base = r"\d{1,2}:\d{2}(?::\d{2})?"
    if am_pm:
        return base + r"[\s\u202F\u00A0]*[AaPp][Mm]"
    return base


# (name, regex). ORDER MATTERS on ties: bracket+12h is tried first. Every pattern
# captures three groups in the same positions: 1=date, 2=time, 3=sender-name,
# 4=body, 5=system line body (a timestamped line with no "Sender: " prefix).
def _patterns():
    dt = r"(?P<date>\d{1,2}/\d{1,2}/\d{2,4})"
    sep = r"[\s\u202F\u00A0]*"
    rest = r"(?:(?P<sender>.+?):\s+(?P<body>.*)|(?P<sysbody>.+))"
    return [
        (
            "bracket_12h",
            re.compile(
                rf"^\[(?P<date>\d{{1,2}}/\d{{1,2}}/\d{{2,4}}),{sep}"
                rf"(?P<time>{_dt_format_re_time(am_pm=True)})]{sep}"
                rf"(?:(?P<sender>.+?):[ ]+(?P<body>.*)|(?P<sysbody>.+))$"
            ),
        ),
        (
            "bracket_24h",
            re.compile(
                rf"^\[(?P<date>\d{{1,2}}/\d{{1,2}}/\d{{2,4}}),{sep}"
                rf"(?P<time>{_dt_format_re_time(am_pm=False)})]{sep}"
                rf"(?:(?P<sender>.+?):[ ]+(?P<body>.*)|(?P<sysbody>.+))$"
            ),
        ),
        (
            "dash_12h",
            re.compile(
                rf"^{dt},{sep}(?P<time>{_dt_format_re_time(am_pm=True)})"
                rf"{sep}[\u2013\u2014-]{sep}"
                rf"(?:(?P<sender>.+?):[ ]+(?P<body>.*)|(?P<sysbody>.+))$"
            ),
        ),
        (
            "dash_24h",
            re.compile(
                rf"^{dt},{sep}(?P<time>{_dt_format_re_time(am_pm=False)})"
                rf"{sep}[\u2013\u2014-]{sep}"
                rf"(?:(?P<sender>.+?):[ ]+(?P<body>.*)|(?P<sysbody>.+))$"
            ),
        ),
    ]


PATTERNS = _patterns()

# A body whose lines are only media placeholders (e.g. "<Media omitted>",
# "image omitted", "<attached: 20240912_1330.jpg>") is a media-not-in-text event,
# not chat content.
PLACEHOLDER_RE = re.compile(
    r"<attached:\s*(?P<file>[^>]+)>"
    r"|<\s*(?P<kind_wrapped>media|image|video|audio|document|sticker)\s+omitted\s*>"
    r"|\b(?P<kind_bare>media|image|video|audio|document|sticker)\s+omitted\b",
    re.IGNORECASE,
)

_KIND_FROM_PLACEHOLDER = {
    "media": MEDIA_TYPE,
    "image": "image",
    "video": "video",
    "audio": "audio",
    "document": "document",
    "sticker": "sticker",
}


class ExportParseError(ValueError):
    """Raised when an upload does not look like a WhatsApp chat export."""


@dataclass(frozen=True)
class ParsedMessage:
    """One exported message, mirroring the live webhook mapping where possible."""

    chat_id: str
    chat_name: str | None
    sender_name: str | None
    sender_id: str | None = None
    from_me: bool = False
    msg_type: str = "text"
    body: str | None = None
    timestamp: datetime | None = None  # UTC, aware
    media_omitted: bool = False
    media_kind: str | None = None
    content: bool = False  # True = embeddable chat text (not system / media-omitted)
    raw_line: str = ""
    # Local wall-clock time as written in the export (no tz). Used ONLY for the
    # synthetic dedup key so re-imports stay idempotent regardless of tz setting.
    local_timestamp: datetime | None = None


# --- date/time handling ---------------------------------------------------------------
_PART_DATE_RE = re.compile(r"^(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{2,4})$")
_PART_TIME_RE = re.compile(
    r"^(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?[\s\u202F\u00A0]*(?P<ampm>[AaPp][Mm])?$"
)


def _parse_local_datetime(date: str, time: str, date_order: str) -> datetime | None:
    dm = _PART_DATE_RE.match(date)
    tm = _PART_TIME_RE.match(time)
    if not dm or not tm:
        return None
    year = int(dm.group("y"))
    if year < 100:
        # Map 2-digit years: 00-49 → 2000-2049, 50-99 → 1950-1999.
        # Threshold of 50 (not 70) so e.g. 2026 is "26" → 2026, not 1926.
        year += 2000 if year < 50 else 1900
    a, b = int(dm.group("a")), int(dm.group("b"))
    month, day = (a, b) if date_order == "MDY" else (b, a)
    hour = int(tm.group("h"))
    minute = int(tm.group("m"))
    second = int(tm.group("s") or 0)
    ampm = (tm.group("ampm") or "").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def _detect_date_order(matches: list[tuple[str, str]]) -> str:
    """Count which date interpretation validates, from the DETECTED pattern's matches.

    "11/12/24" is valid as both MM/DD and DD/MM, so the tie-break (and the fallback
    when nothing disambiguates) is US-style MDY - matches the WAHA/docs examples.
    """
    mdy = dmy = 0
    for date, _time in matches:
        dm = _PART_DATE_RE.match(date)
        if not dm:
            continue
        a, b = int(dm.group("a")), int(dm.group("b"))
        if 1 <= a <= 12 and 1 <= b <= 31:
            mdy += 1
        if 1 <= b <= 12 and 1 <= a <= 31:
            dmy += 1
    if mdy == 0 and dmy > 0:
        return "DMY"
    return "MDY"  # winner or tie


def _detect_pattern(text: str) -> tuple[re.Pattern, str, str]:
    """Pick the line-start pattern + date order from the first ~50 content lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:_SAMPLE_LINES]
    best_name, best_regex, best_count = None, None, 0
    for name, regex in PATTERNS:
        count, matches = 0, []
        for ln in lines:
            m = regex.match(ln)
            if m:
                count += 1
                matches.append((m.group("date"), m.group("time")))
        if count > best_count:
            best_name, best_regex, best_count = name, regex, count
            best_matches = matches
    if best_count == 0 or best_regex is None:
        raise ExportParseError(
            "Could not detect a WhatsApp export timestamp pattern in the text. "
            "Expected lines like '[11/12/2024, 10:15:30 AM] Sender: message'."
        )
    return best_regex, best_name, _detect_date_order(best_matches)


def synthetic_message_id(chat_id: str, sender_name: str | None, local_ts: datetime | None, body: str | None) -> str:
    """Stable dedup key for exported messages: hash of (chat, sender, time, body).

    Timestamps in the export are stable per file, so the same file (or an updated
    export covering overlapping history) re-imports without duplicating rows.
    """
    stamp = local_ts.isoformat(timespec="seconds") if local_ts is not None else ""
    key = f"{chat_id}\x1f{sender_name or ''}\x1f{stamp}\x1f{body or ''}"
    return "exp_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]


def _classify_body(body: str | None) -> tuple[str | None, bool, str | None]:
    """Strip only the media-placeholder spans (e.g. "<Media omitted>").

    A message like "check out this <Media omitted>" keeps its real text for
    embedding while being flagged as having omitted media; a message that is
    ONLY placeholders becomes a pure media event (body=None, media kind tagged).
    """
    if not body or not body.strip():
        return None, False, None
    kinds: list[str] = []
    parts: list[str] = []
    pos = 0
    for m in PLACEHOLDER_RE.finditer(body):
        kinds.append(
            _KIND_FROM_PLACEHOLDER.get(
                (m.group("kind_wrapped") or m.group("kind_bare") or "media").lower()
                if not m.group("file")
                else "document",
                MEDIA_TYPE,
            )
        )
        parts.append(body[pos : m.start()])
        pos = m.end()
    parts.append(body[pos:])
    cleaned = "".join(parts).strip()
    if not cleaned:
        # The message is ONLY media placeholders -> a media event, no chat text.
        return None, True, (kinds[0] if kinds else MEDIA_TYPE)
    # Keep the text; the omitted media is metadata, not something to embed verbatim.
    return cleaned, bool(kinds), (kinds[0] if kinds else None)


def parse_export_lines(
    text: str,
    *,
    chat_id: str,
    chat_name: str | None = None,
    tz_name: str = "UTC",
    bot_name: str = "",
) -> list[ParsedMessage]:
    """Parse raw export text into ParsedMessage records."""
    regex, detected_name, date_order = _detect_pattern(text)
    tzinfo = ZoneInfo(tz_name) if tz_name else timezone.utc
    records: list[ParsedMessage] = []
    pending: ParsedMessage | None = None

    def flush() -> None:
        if pending is not None:
            records.append(pending)

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        m = regex.match(line)
        if m:
            flush()
            local_ts = _parse_local_datetime(m.group("date"), m.group("time"), date_order)
            sender = m.group("sender")
            sys_body = m.group("sysbody")
            if sys_body is not None:
                # Timestamped line with no "Sender: " prefix -> a WhatsApp system
                # message ("X added Y", "Messages and calls are...").
                pending = ParsedMessage(
                    chat_id=chat_id,
                    chat_name=chat_name,
                    sender_name=None,
                    msg_type=SYSTEM_TYPE,
                    body=sys_body.strip() or None,
                    timestamp=_to_utc(local_ts, tzinfo),
                    content=False,
                    raw_line=line,
                    local_timestamp=local_ts,
                )
                continue
            body_raw = m.group("body")
            body, media_omitted, media_kind = _classify_body(body_raw)
            # Text remains embeddable even when media was omitted alongside it;
            # only a body-less pure-media event gets the specific media type.
            msg_type = "text" if body else (media_kind if media_kind else "other")
            pending = ParsedMessage(
                chat_id=chat_id,
                chat_name=chat_name,
                sender_name=sender.strip() if sender else None,
                from_me=bool(bot_name) and sender is not None and sender.strip() == bot_name,
                msg_type=msg_type,
                body=body,
                timestamp=_to_utc(local_ts, tzinfo),
                media_omitted=media_omitted,
                media_kind=media_kind or ("text" if body else None),
                content=bool(body),
                raw_line=line,
                local_timestamp=local_ts,
            )
            continue
        # No timestamp prefix: a continuation line of the previous message.
        if pending is not None and pending.body is not None:
            # Strip trailing whitespace, keep leading so monospace lines survive.
            pending = _append_line(pending, raw_line.rstrip())
            continue
        # Preamble / signature lines before the first message ("Messages and calls
        # are end-to-end encrypted...") - ignored.

    flush()
    return records


def _append_line(msg: ParsedMessage, text_line: str) -> ParsedMessage:
    text_line = text_line.rstrip()
    body = ((msg.body or "") + "\n" + text_line).strip()
    return ParsedMessage(
        chat_id=msg.chat_id,
        chat_name=msg.chat_name,
        sender_name=msg.sender_name,
        sender_id=msg.sender_id,
        from_me=msg.from_me,
        msg_type=msg.msg_type,
        body=body,
        timestamp=msg.timestamp,
        media_omitted=msg.media_omitted,
        media_kind=msg.media_kind,
        content=bool(body) and msg.msg_type != SYSTEM_TYPE,
        raw_line=msg.raw_line,
        local_timestamp=msg.local_timestamp,
    )


def _to_utc(local_ts: datetime | None, tzinfo) -> datetime | None:
    if local_ts is None:
        return None
    return local_ts.replace(tzinfo=tzinfo).astimezone(timezone.utc)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ExportParseError("Could not decode export text (tried utf-8/utf-16/latin-1)")


def _extract_txt(data: bytes) -> str:
    """Accept raw .txt bytes or a WhatsApp .zip and return the decoded chat text."""
    if data[:4] == _ZIP_MAGIC:
        names: list[tuple[str, int]] = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    fn = info.filename
                    if "__MACOSX/" in fn or info.is_dir():
                        continue
                    if fn.lower().endswith(".txt") and "chat" in fn.lower():
                        names.append((fn, info.file_size))
        except zipfile.BadZipFile as exc:
            raise ExportParseError("Uploaded file is not a valid .zip archive") from exc
        if not names:
            raise ExportParseError("No .txt chat export found inside the .zip")
        best = max(names, key=lambda item: item[1])[0]
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return _decode(zf.read(best))
    return _decode(data)


def _chat_name_from_filename(filename: str) -> str | None:
    base = filename.rsplit("/", 1)[-1].removesuffix(".txt").removesuffix(".TXT")
    base = base.removeprefix("WhatsApp Chat with ").strip()
    if not base or base.lower().endswith(".zip"):
        return None
    return base


@dataclass
class ParsedExport:
    records: list[ParsedMessage]
    chat_name: str | None
    source: str  # human-readable: file name or zip member used


def parse_export(
    data: bytes,
    *,
    chat_id: str,
    chat_name: str | None = None,
    tz_name: str = "UTC",
    bot_name: str = "",
    source_name: str = "upload.txt",
) -> ParsedExport:
    """Top-level entry point: bytes -> parsed messages, idempotent on re-import."""
    text = _extract_txt(data)
    if not text.strip():
        raise ExportParseError("Export text is empty")
    detected_name = chat_name or _chat_name_from_filename(source_name)
    records = parse_export_lines(
        text,
        chat_id=chat_id,
        chat_name=detected_name,
        tz_name=tz_name,
        bot_name=bot_name,
    )
    if not records:
        raise ExportParseError("Parsed zero messages; is this really a chat export?")
    return ParsedExport(records=records, chat_name=detected_name, source=source_name)