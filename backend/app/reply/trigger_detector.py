"""Decide whether an incoming WAHA message is addressed to the bot.

Two trigger families, both gated on ``payload.fromMe``:

* Direct messages (1:1): the sender is talking to the bot, so any text message
  with a body is a trigger. No bot JID is needed to decide.
* Group @-mentions: only fire when the bot was explicitly @-mentioned. WAHA's
  incoming payload has no first-class ``mentions`` field (verified against the
  running NOWEB engine's ``toWAMessage``), so the mention JIDs are read from the
  raw engine data at ``payload._data.message.extendedTextMessage.contextInfo.mentionedJid``.
  A regex fallback scans the body for ``@<bot-digits>`` so detection still works
  before a session is paired / for engines that do not populate ``mentionedJid``.

The @-mention text is stripped from the question before retrieval. If nothing
remains after the strip (mention-only message), the caller replies with the short
"what's up?" clarification instead of running retrieval.

Own (``fromMe``) messages are always ignored: without this guard the bot would
answer its own replies forever.
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    """Result of trigger detection for one incoming message."""

    should_reply: bool
    chat_id: str | None  # where the reply should be sent (None when should_reply=False)
    question: str | None  # cleaned question (None when should_reply=False)
    clarify: bool  # True when it was addressed but had no real question
    is_group: bool
    reason: str  # human-readable, for logs


# ---------------------------------------------------------------- helpers

def _normalize_jid(value: Any) -> str | None:
    """NOWEB reports contacts as 123@s.whatsapp.net; canonicalize to @c.us."""
    if value is None:
        return None
    as_str = str(value).strip()
    if not as_str:
        return None
    return as_str.replace("@s.whatsapp.net", "@c.us")


def _digits(jid: str | None) -> str:
    """Digits-only form of a JID for matching (phone numbers stay unique)."""
    if not jid:
        return ""
    return re.sub(r"\D", "", jid)


_MENTION_BODY_RE = re.compile(r"@(?P<digits>\d{6,})\b")


def _is_group(payload: dict) -> bool:
    """WAHA sets ``participant`` only for group messages; JID suffix as backup."""
    participant = payload.get("participant")
    if participant is not None and str(participant).strip():
        return True
    chat = str(payload.get("from") or payload.get("to") or "")
    return chat.endswith("@g.us")


def extract_mentioned_jids(payload: dict) -> list[str]:
    """Mentioned contact JIDs from the raw engine message (_data).

    WAHA's NOWEB engine keeps Baileys' raw message at ``payload._data``; the
    full list of @-mentioned JIDs lives at
    ``message.extendedTextMessage.contextInfo.mentionedJid`` (each ``@s.whatsapp.net``,
    normalized here to ``@c.us``). Unwraps a single viewOnce layer defensively.
    """
    data = payload.get("_data") or {}
    message = data.get("message") or {}
    if isinstance(message, dict) and isinstance(message.get("viewOnceMessage"), dict):
        inner = message["viewOnceMessage"].get("message")
        if isinstance(inner, dict):
            message = inner

    ext = message.get("extendedTextMessage") or {}
    context_info = ext.get("contextInfo") or {}
    raw_jids = context_info.get("mentionedJid") or []
    return [jid for jid in (_normalize_jid(j) for j in raw_jids) if jid]


def strip_mention(body: str, bot_jid: str | None) -> str:
    """Remove @-mention tokens from a group message body.

    A group @-mention renders as ``@<digits>`` inside the body text; it would
    otherwise pollute the retrieval query. Anything that looks like an @mention
    is removed (not just the bot's) so leftover people-mentions do not become
    search keywords.
    """
    if not body:
        return ""
    without_mentions = re.sub(r"@\w+", " ", body)
    cleaned = re.sub(r"\s+", " ", without_mentions)
    return cleaned.strip()


# ---------------------------------------------------------------- detection

def detect_reply(payload: dict, bot_jid: str | None) -> Detection:
    """Classify an incoming WAHA message payload against the trigger rules.

    ``bot_jid`` is the bot's own WhatsApp JID (``1555...@c.us``) or None when it
    has not been configured / discovered yet - group mention detection silently
    degrades to no-op in that case.
    """
    if bool(payload.get("fromMe")):
        # Loop guard: never answer our own messages (each reply would otherwise
        # re-trigger "message" events forever).
        return Detection(False, None, None, False, False, "from_me (own message)")

    body = payload.get("body")
    body_text = str(body) if body is not None else ""
    is_group = _is_group(payload)
    # Reply goes to the chat that asked: `from` for inbound 1:1 (contact JID),
    # the group JID for group messages (`to`/`from` both carry it there).
    chat_id = _normalize_jid(payload.get("from") or payload.get("to"))

    if not is_group:
        if not chat_id or not chat_id.endswith(("@c.us", "@s.whatsapp.net")):
            # status broadcasts / newsletters are not real 1:1 chats
            return Detection(False, None, None, False, False, f"non-chat sender {chat_id!r} ignored")
        if not body_text.strip():
            # media-only DM with no caption: nothing to answer
            return Detection(False, None, None, False, False, "dm without text body")
        return Detection(True, chat_id, body_text, False, False, "dm")

    # ---- group ----
    bot_digits = _digits(bot_jid)
    if not bot_digits:
        return Detection(
            False,
            None,
            None,
            False,
            True,
            "bot_whatsapp_id unknown; cannot resolve @-mentions in group",
        )

    mentions = extract_mentioned_jids(payload)
    mentioned_bot = any(_digits(jid) == bot_digits for jid in mentions)
    if not mentioned_bot:
        # fallback: search the body for "@<bot-digits>" (works pre-pairing)
        mentioned_bot = bool(_MENTION_BODY_RE.search(body_text))

    if not mentioned_bot:
        return Detection(False, None, None, False, True, "group message without bot mention")

    question = strip_mention(body_text, bot_jid)
    if not question:
        return Detection(True, chat_id, None, True, True, "group mention without question")
    return Detection(True, chat_id, question, False, True, "group mention")