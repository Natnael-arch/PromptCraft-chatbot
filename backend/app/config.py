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