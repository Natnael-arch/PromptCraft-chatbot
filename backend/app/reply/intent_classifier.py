"""Intent classification for the Phase 5 casual/banter mode.

Splits a triggered message into one of two buckets before ``reply_worker``
decides how to answer:

* ``"knowledge"`` - the message actually asks about the program, deadlines,
  announcements, the product/bot itself, or any project vocabulary. The existing
  retrieval pipeline answers it.
* ``"banter"``   - a joke, exclamation or low-content message with no knowledge
  signal. The casual personality handles it (see ``answer._banter_reply``).

The bias is deliberately KNOWLEDGE-first, mirroring ``route_question``'s
rule-based style: any program keyword OR an interrogative frame (a question word
or a trailing ``?``) classifies as knowledge, and with neither present we still
default to knowledge on ambiguity. Only messages that carry *no* knowledge signal
at all fall through to banter - a bot that jokes away a real question is worse
than a bot that try-hards through a joke. Every decision is logged at INFO so the
split is auditable in ``docker compose logs``.
"""

import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

# Short greetings / small-talk well-being check-ins that carry no informational intent:
# even if framed with interrogatives or trailing '?' ("how are you?", "what's up?"),
# these classify as banter rather than knowledge.
_SMALL_TALK_GREETING_RE = re.compile(
    r"^(?:"
    r"h[ea]llo+|hi+|hey+|sup|yo+|greetings|good\s+(?:morning|afternoon|evening|night|day)"
    r"|how\s+(?:are\s+you|are\s+u|is\s+it\s+going|'s\s+it\s+going|are\s+you\s+doing|u\s+doing|'s\s+up|do\s+you\s+do)"
    r"|what(?:\'s|\s+is)\s+up"
    r"|how\'s\s+everything"
    r")[\s!?.]*$",
    re.IGNORECASE,
)

# Interrogative frames that read as a real question even without program
# vocabulary: the classic wh- words, auxiliary-verb + subject openings
# ("do you respond...", "is there...", "can you tell me..."), and a trailing '?'
# (e.g. DMs like "lol?" still get treated as questions rather than joked away).
_INTERROGATIVE_RE = re.compile(
    r"\b(?:who|what|which|when|where|why|how|whom|whose)\b"
    r"|\b(?:is|are|was|were|do|does|did|can|could|would|should|will|shall|may|might)\b"
    r"\s+(?:my|your|our|the|this|that|there|it|i|we|you|they)"
    r"|\?"
)

KNOWLEDGE = "knowledge"
BANTER = "banter"


def classify_intent(question: str) -> str:
    """Return ``"knowledge"`` or ``"banter"`` for a triggered message.

    Rule order:
    1. Program/announcement vocabulary (``KNOWLEDGE_INTENT_KEYWORDS``) present
       as a case-insensitive substring -> knowledge.
    2. Pure greeting or small-talk check-in ("how are you?", "what's up?", "hi") -> banter.
    3. No vocabulary hit, but an interrogative frame (question word, an
       auxiliary-verb opening, or ``?``) -> knowledge (conservative: better to
       answer try-hard than joke away).
    4. No knowledge signal at all -> banter.
    """
    q = (question or "").strip().lower()

    for keyword in settings.knowledge_intent_keywords:
        if keyword and keyword in q:
            logger.info("Intent=knowledge (keyword=%r) question=%r", keyword, q[:200])
            return KNOWLEDGE

    if _SMALL_TALK_GREETING_RE.match(q):
        logger.info("Intent=banter (small-talk/greeting) question=%r", q[:200])
        return BANTER

    if _INTERROGATIVE_RE.search(q):
        logger.info("Intent=knowledge (interrogative) question=%r", q[:200])
        return KNOWLEDGE

    logger.info("Intent=banter question=%r", q[:200])
    return BANTER