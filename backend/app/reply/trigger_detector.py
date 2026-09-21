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

Note on current NOWEB behavior (verified from a live paired session): group
@-mentions of the bot arrive with the LID in ``mentionedJid`` (``30727051714790@lid``)
and render in the body as ``@30727051714790`` - the phone number never appears.
Match against the lid as well as the phone number (``bot_aliases``), and only
count body mentions whose digits are actually ours.
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


def resolve_chat_id(payload: dict) -> str | None:
    """The chat/group JID a reply should go to, from a WAHA message payload.

    Inbound messages carry it in ``from`` (contact JID for 1:1, group JID for
    groups); outbound (bot's own) messages carry it in ``to``. Shared by the
    @-mention detection and the slash-command path so the two never disagree on
    where a reply belongs.
    """
    return _normalize_jid(payload.get("from") or payload.get("to"))


def resolve_sender_id(payload: dict) -> str | None:
    """The author's JID for a message, matching ``messages.sender_id``.

    Group messages name their author in ``participant``; 1:1 messages have no
    participant, so the author is the ``from`` JID. Used by the trusted-sender
    lockdown of the /unhinged_* commands (commands are never from_me, so this
    deliberately ignores the bot's own identity).
    """
    return _normalize_jid(payload.get("participant") or payload.get("from"))


# Slash commands recognized BEFORE any mention/DM gating. A bare command in a
# group - with no @-mention of the bot - must still work. Values are the
# canonical command names the reply worker switches on.
SLASH_COMMANDS = {
    "/unhinged_on": "unhinged_on",
    "/unhinged_off": "unhinged_off",
}


def detect_command(payload: dict) -> str | None:
    """Return the command name when the raw body is a known slash command.

    Matches the trimmed, case-insensitive body against the fixed command set
    ("/Unhinged_ON", " /unhinged_on " etc. all resolve to the same command);
    anything else - including a command buried inside a longer sentence - is
    None. The bot's own messages are ignored so a loop can never toggle itself.
    """
    if bool(payload.get("fromMe")):
        return None
    body = payload.get("body")
    text = str(body).strip() if body is not None else ""
    return SLASH_COMMANDS.get(text.lower())


_MENTION_DIGITS_RE = re.compile(r"@(\d{6,})")


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

def detect_reply(payload: dict, bot_jid: str | None, bot_aliases: set[str] | None = None) -> Detection:
    """Classify an incoming WAHA message payload against the trigger rules.

    ``bot_jid`` is the bot's own WhatsApp JID (``1555...@c.us``) or None when it
    has not been configured / discovered yet - group mention detection silently
    degrades to no-op in that case.

    ``bot_aliases`` is the set of digit-strings the bot can be mentioned under.
    As of NOWEB 2026.x, group @-mentions of an account surface the LID (e.g.
    ``30727051714790@lid``) rather than the phone JID in both ``mentionedJid``
    and the rendered ``@<digits>`` body text, so identity must cover both the
    phone number and the lid. Defaults to the phone digits when omitted.
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
    chat_id = resolve_chat_id(payload)

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
    aliases = set(bot_aliases or set())
    if bot_digits:
        aliases.add(bot_digits)
    if not aliases:
        return Detection(
            False,
            None,
            None,
            False,
            True,
            "bot_whatsapp_id unknown; cannot resolve @-mentions in group",
        )

    mentions = extract_mentioned_jids(payload)
    mentioned_bot = any(_digits(jid) in aliases for jid in mentions)
    if not mentioned_bot:
        # fallback: the rendered text carries "@<digits>" for the mentioned
        # contact. Only count it when the digits are actually ours (a generic
        # "@anyone" mention must not trigger the bot).
        mentioned_bot = any(d in aliases for d in _MENTION_DIGITS_RE.findall(body_text))

    if not mentioned_bot:
        return Detection(False, None, None, False, True, "group message without bot mention")

    question = strip_mention(body_text, bot_jid)
    if not question:
        return Detection(True, chat_id, None, True, True, "group mention without question")
    return Detection(True, chat_id, question, False, True, "group mention")