import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.db import engine
from app.routes_admin import router as admin_router
from app.routes_ask import router as ask_router
from app.routes_debug import router as debug_router
from app.routes_ingest import router as ingest_router
from app.routes_voice import router as voice_router
from app.webhook import router as webhook_router

logger = logging.getLogger(__name__)

# Default root logger is WARNING-only, which hides the app's INFO diagnostics
# (webhook captures, auto-reply decisions in Phase 4). Wire stderr to INFO so
# those observability lines actually show in `docker compose logs`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The schema is managed by Alembic (the container entrypoint runs
    # `alembic upgrade head` before uvicorn starts), so startup only verifies the
    # connection instead of calling create_all - migrations are the single source
    # of truth as the schema evolves in later phases.
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection OK")
    except Exception:  # noqa: BLE001
        logger.exception("Database connection failed at startup")
    yield


app = FastAPI(title="unipods-bot", version="0.1.0", lifespan=lifespan)

app.include_router(webhook_router)
app.include_router(debug_router)
app.include_router(ingest_router)
app.include_router(ask_router)
app.include_router(voice_router)
app.include_router(admin_router)


@app.get("/")
def root() -> dict:
    return {"app": "unipods-bot", "docs": "/docs", "health": "/health"}