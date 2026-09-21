"""WAHA HTTP wrapper for the auto-reply path.

Only the two endpoints the reply worker needs:

* ``POST /api/sendText`` - send a text reply back to a chat.
* ``GET /api/sessions`` - discover the bot's own JID (``me.id``) for mention
  detection, cached in memory after the first successful read.

Shape verified against the running WAHA (2026.8.x): body
``{"session", "chatId", "text"}`` plus the ``X-Api-Key`` header. Text messages
are capped at 4096 chars by WhatsApp, so long answers are truncated here.
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

WAHA_SEND_TEXT_PATH = "/api/sendText"
WAHA_SESSIONS_PATH = "/api/sessions"


class ReplySenderError(RuntimeError):
    """WAHA refused or failed to send the reply."""


def truncate_text(text: str, max_chars: int | None = None) -> str:
    """Trim a reply to WhatsApp's message budget (4096) with an ellipsis."""
    limit = max_chars or settings.reply_max_chars
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].strip() + "\u2026"


def _headers() -> dict[str, str]:
    if settings.waha_api_key:
        return {"X-Api-Key": settings.waha_api_key}
    return {}


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=settings.waha_base_url,
        headers=_headers(),
        timeout=15.0,
    )


def send_text(
    chat_id: str,
    text: str,
    *,
    session: str | None = None,
    client: httpx.Client | None = None,
) -> dict | None:
    """Send ``text`` to ``chat_id`` and return WAHA's JSON payload (or None).

    ``client`` is injectable for tests (httpx.MockTransport); a real client on
    ``settings.waha_base_url`` is created when omitted. Raises ``ReplySenderError``
    on any non-2xx response so callers can fall back to an apology.
    """
    session_name = session or settings.waha_session
    body = {
        "session": session_name,
        "chatId": chat_id,
        "text": truncate_text(text),
    }

    owns_client = client is None
    if owns_client:
        client = _client()
    try:
        response = client.post(WAHA_SEND_TEXT_PATH, json=body)
        if response.status_code >= 400:
            raise ReplySenderError(
                f"WAHA sendText failed: HTTP {response.status_code} for chat_id={chat_id!r}"
            )
        try:
            return response.json()
        except ValueError:
            return None
    finally:
        if owns_client:
            client.close()


def fetch_session_me(
    *,
    session: str | None = None,
    client: httpx.Client | None = None,
) -> dict | None:
    """Return the WAHA ``me`` object for a session: {"id", "lid", "pushName", ...}.

    ``id`` is the phone JID (``251947711181@c.us``) and ``lid`` the account's
    linked-device identity (``30727051714790@lid``) - both are needed to match
    group @-mentions. Safe on missing/unpaired sessions: returns None instead of
    raising so the worker can fall back gracefully (group mentions become no-ops
    until the session is paired).
    """
    session_name = session or settings.waha_session
    owns_client = client is None
    if owns_client:
        client = _client()
    try:
        response = client.get(WAHA_SESSIONS_PATH)
        if response.status_code >= 400:
            logger.warning("Cannot resolve bot identity: WAHA /api/sessions HTTP %s", response.status_code)
            return None
        sessions = response.json()
        if not isinstance(sessions, list):
            return None
        for entry in sessions:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("name") or entry.get("id") or "") != session_name:
                continue
            me = entry.get("me") or {}
            if (me.get("id") or "").strip():
                return me
        return None
    except httpx.HTTPError as exc:
        logger.warning("Cannot resolve bot identity from WAHA sessions: %s", exc)
        return None
    finally:
        if owns_client:
            client.close()