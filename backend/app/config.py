from urllib.parse import quote_plus

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables / .env."""

    # --- Postgres (only used to build DATABASE_URL when it is not provided) ---
    postgres_user: str = "unipods"
    postgres_password: str = "unipods_dev_password"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "unipods"

    # --- Backend ---
    # Optional override. When unset, the connection string below is built from the
    # POSTGRES_* values above (single source of truth for the compose setup).
    database_url: str = ""

    # --- WAHA ---
    # Base URL as seen from INSIDE the compose network. Sent as the X-Api-Key header
    # on backend -> WAHA calls (health check). Empty means no auth header is sent.
    waha_base_url: str = "http://waha:3000"
    waha_api_key: str = ""
    waha_session: str = "default"

    # Pre-provisioned for the Phase 3 job queue; not used by Phase 1 code paths.
    redis_url: str = "redis://redis:6379/0"

    # --- Embeddings (Phase 2/3) ---
    # Provider name: "openai_compatible" (any OpenAI-style POST <url>/embeddings
    # server, e.g. a vLLM/Ollama gateway serving a 1024-dim model), "gemini"
    # (Google Gemini via the google-genai SDK), or "mock" (deterministic vectors,
    # DEV ONLY - lets the pipeline run with no API key).
    # Match EMBEDDING_DIMENSIONS to the provider's model output: the chunks table
    # is VECTOR(1024). Gemini truncates server-side via output_dimensionality, so
    # gemini-embedding-001 (native 3072-d) fits the schema at 1024-d.
    embedding_provider: str = "mock"
    embedding_api_url: str = ""  # e.g. http://localhost:11434/v1 (Ollama)
    embedding_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 64
    embedding_timeout: float = 120.0

    # --- Gemini (Phase 3) ---
    # API key for the Google AI Studio / Gemini API (used when
    # EMBEDDING_PROVIDER=gemini and for Gemini answer generation).
    gemini_api_key: str = ""

    # --- Answer generation (Phase 3) ---
    # Provider name: "gemini" (default gemini-2.5-flash) synthesizes the answer
    # over the retrieved chunks/citations; "extractive" keeps returning the
    # Phase-2 bullet list with no LLM call (offline / no API key).
    answer_provider: str = "extractive"
    answer_model: str = "gemini-2.5-flash"

    # --- Sessionization (Phase 2) ---
    # A gap of this many minutes with no activity starts a new session.
    session_gap_minutes: int = 30
    # A single continuously-active conversation is capped at this many messages per
    # session so one busy day does not become one giant un-embed-able blob.
    session_max_messages: int = 200
    # Chunks are split at message boundaries to stay under this many characters.
    session_max_chunk_chars: int = 8000

    # --- Retrieval (Phase 2) ---
    retrieval_top_k: int = 6
    # RRF k constant: merged_score = sum(1 / (k + rank)). k=60 is the classic value.
    retrieval_rrf_k: int = 60

    # --- Export import (Phase 2) ---
    # If set, messages from this sender name are flagged from_me (the bot's own
    # messages in the export).
    export_bot_name: str = ""

    # --- Voice transcription (Phase 3) ---
    # Gemini model used to transcribe audio. As of 2026 the gemini-2.5-flash tier
    # accepts audio directly and supports structured output, with a per-prompt
    # limit of roughly 8.4 hours (~1M tokens; 32 tokens/second) and a single audio
    # file per request. Supported MIME types include audio/ogg (covers WhatsApp's
    # Ogg-Opus voice notes), audio/mpeg, audio/mp3, audio/m4a, audio/wav,
    # audio/x-aac, audio/flac, audio/webm, audio/pcm. Re-verify these limits before
    # changing the model tier - they drift between releases.
    transcribe_model: str = "gemini-2.5-flash"
    # Recordings longer than this many seconds are transcribed in fixed windows
    # (each well under the ~8.4h single-request cap) so structured JSON output
    # stays reliable on very long tracks. Segment timestamps are kept on the
    # recording's own timeline and merged back into one continuous transcript.
    transcribe_window_seconds: int = 1800
    # Max multipart upload size for voice files. Gemini's own input limit is 500MB
    # and inline (non-Files-API) requests are capped at 20MB, above which
    # transcriber.py goes through the Files API - so this cap is a dev-sanity bound.
    voice_max_upload_bytes: int = 200 * 1024 * 1024
    # Max speaker turns packed into one voice chunk before flushing, mirroring
    # Phase 2's session_max_messages hard cap for text sessions (chunks also stay
    # under session_max_chunk_chars, the shared char budget).
    voice_segments_per_chunk: int = 50

    # --- Live replies (Phase 4) ---
    # The bot's own WhatsApp JID (e.g. 15551234567@c.us). When empty it is
    # discovered lazily from the WAHA session info and cached in memory.
    bot_whatsapp_id: str = ""
    # Per-chat in-memory cooldown between automatic replies, in seconds. Prevents
    # spam / feedback loops in busy groups. Not persisted (Phase 5 moves to Redis).
    reply_cooldown_seconds: int = 30
    # Automatic replies longer than this many characters are truncated before
    # sendText (WhatsApp text messages cap at 4096 chars).
    reply_max_chars: int = 4000

    # --- Live ingest (Phase 4) ---
    # Minimum seconds between automatic session/chunk rebuilds PER CHAT
    # (in-memory debounce). Bursts of messages in an active group collapse into
    # one rebuild; combined with the chat_ingest_state cursor this keeps indexing
    # cheap without re-embedding unchanged history.
    ingest_debounce_seconds: int = 45
    # Kill switch: when false, schedule_incremental_ingest is a no-op so indexing
    # is only ever driven manually via /ingest/export (useful for demos).
    ingest_auto_enabled: bool = True

    # --- Trusted senders (Phase 5) ---
    # Secret required in the X-Admin-Token header for the trusted-sender admin
    # routes (/admin/trusted-senders). Hackathon-grade stopgap: a shared static
    # token read from env, not a real auth system.
    admin_token: str = ""
    # When true, only senders present in the trusted_senders table may toggle a
    # chat's unhinged mode via /unhinged_on|off. A command from anyone else is
    # SILENTLY ignored (logged, no state change, no reply) so it doesn't look
    # broken to someone probing whether it's admin-only. False = anyone can
    # toggle, matching the default ungated behavior.
    unhinged_toggle_restricted_to_trusted: bool = False

    # --- Casual mode (Phase 5) ---
    # Kill switch for the casual/banter reply mode. When false, classify_intent is
    # never consulted and every detection takes the knowledge path exactly as in
    # Phase 4 - flipping this single env var is the full revert.
    banter_mode_enabled: bool = True
    # Vocabulary of program/announcement terms. If the user's question touches any
    # of these (case-insensitive substring), intent is classified as "knowledge".
    # Split on commas/pipes when set via env, e.g.
    # KNOWLEDGE_INTENT_KEYWORDS="deadline,submission,judging,recording"
    knowledge_intent_keywords: list[str] = [
        "deadline",
        "submission",
        "submit",
        "judging",
        "judge",
        "prize",
        "winners",
        "winning",
        "team",
        "teams",
        "hackathon",
        "program",
        "cohort",
        "schedule",
        "agenda",
        "recording",
        "meeting",
        "announcement",
        "announce",
        "mentor",
        "demo",
        "github",
        "repo",
        "repository",
        "api",
        "port",
        "server",
        "deploy",
        "instructions",
        "instruction",
        "requirement",
        "feature",
        "bug",
        "dates",
        "date",
        "bot",
        "afrobo",
        "different from one another",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Do not JSON-decode env values for complex-typed fields. Plain strings
        # are the only list source here (KNOWLEDGE_INTENT_KEYWORDS), and they are
        # split by _split_keyword_list below - without this flag pydantic-settings
        # would try json.loads() on the env value first and crash on a normal
        # comma-separated string.
        enable_decoding=False,
    )

    @field_validator("knowledge_intent_keywords", mode="before")
    @classmethod
    def _split_keyword_list(cls, v):
        # Env vars can't carry a JSON array, so accept a plain comma/pipe-delimited
        # string too: KNOWLEDGE_INTENT_KEYWORDS="deadline,submission,judging".
        # docker compose forwards ${KNOWLEDGE_INTENT_KEYWORDS:-} as an EMPTY string
        # when the var is unset - treat that as "use the declared default" instead
        # of wiping the vocabulary with an empty list.
        if isinstance(v, str):
            parts = v.replace("|", ",").split(",")
            cleaned = [p.strip() for p in parts if p.strip()]
            if not cleaned:
                return cls.model_fields["knowledge_intent_keywords"].default
            return cleaned
        return v

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()