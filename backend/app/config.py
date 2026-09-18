from urllib.parse import quote_plus

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

    # --- Embeddings (Phase 2) ---
    # Provider name: "openai_compatible" (any OpenAI-style POST <url>/embeddings
    # server, e.g. a vLLM/Ollama gateway serving a 1024-dim model) or "mock"
    # (deterministic vectors, DEV ONLY - lets the pipeline run with no API key).
    # Match EMBEDDING_DIMENSIONS to the provider's model output: the chunks table
    # is VECTOR(1024), so the default bge-m3 (1024 dims) fits the schema.
    embedding_provider: str = "mock"
    embedding_api_url: str = ""  # e.g. http://localhost:11434/v1 (Ollama)
    embedding_api_key: str = ""
    embedding_model: str = "bge-m3"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 64
    embedding_timeout: float = 120.0

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

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

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