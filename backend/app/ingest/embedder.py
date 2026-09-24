"""Embedding providers + writing embeddings into the `chunks` table.

The provider is swappable via the EMBEDDING_PROVIDER env var, but there is a real,
working default behind every abstraction here:

* ``openai_compatible`` - POSTs batches to any OpenAI-style ``/embeddings`` endpoint
  (a local Ollama/vLLM gateway, an OpenAI-compatible cloud host, ...). The default
  model name is bge-m3, which outputs 1024-d vectors - exactly matching the
  ``chunks.embedding VECTOR(1024)`` schema. Set EMBEDDING_API_URL/API_KEY/MODEL to
  point it somewhere real.
* ``gemini`` - Google Gemini via the google-genai SDK. We request
  ``output_dimensionality=1024`` as a *request parameter* (Gemini's native MRL
  truncation - the API truncates server-side, no client-side slicing). The truncated
  vector is L2-renormalized to unit length, because the API does not renormalize
  truncated MRL output and cosine similarity needs unit vectors. Requires
  GEMINI_API_KEY.
* ``mock`` - deterministic pseudo-random unit-normalized vectors, DEV ONLY. It lets
  the whole pipeline (ingest -> sessions -> chunks) run end-to-end with no API key
  or network, so the code path is exercised before a real endpoint is wired up.

Ingestion asks for embeddings as few, large batched calls (not one call per text).
Anything returning vectors with the wrong dimensionality is a hard error: writing
them would silently corrupt the vector(1024) column.

Idempotency: re-embedding a session MUST not duplicate chunks - handled by
``replace_chunks_for_session`` (delete-and-replace on session_id).
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import settings
from app.models import Chunk

logger = __import__("logging").getLogger(__name__)

# Retry policy for Gemini 429 RESOURCE_EXHAUSTED (per-minute RPM limit).
# Only 429s are retried — auth failures, dimension errors, etc. fail fast.
# Delays are deliberately > 60s so retries clear the per-minute window:
#   attempt 1 fails → wait 30s
#   attempt 2 fails → wait 60s  (total ~90s since attempt 1 → past the minute)
#   attempt 3 fails → wait 120s (total ~210s → well past the minute)
#   attempt 4 fails → propagate as EmbeddingError
_GEMINI_RETRY_ATTEMPTS = 4   # total attempts (1 original + 3 retries)
_GEMINI_RETRY_BASE_S   = 30  # seconds before first retry (> half the RPM window)
_GEMINI_RETRY_FACTOR   = 2   # delay doubles each retry: 30s → 60s → 120s


class EmbeddingError(RuntimeError):
    """Base class for embedding failures."""


class EmbeddingRequestError(EmbeddingError):
    """The API could not be reached or rejected the request."""


class EmbeddingDimensionError(EmbeddingError):
    """Provider returned vectors that do not match the chunks.embedding column."""


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized-ish float vector of length ``dimensions`` per text."""
        ...


@dataclass
class OpenAICompatibleEmbedder:
    """Real provider: batched POST /embeddings to an OpenAI-compatible server."""

    name: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = settings.embedding_model
    dimensions: int = settings.embedding_dimensions
    batch_size: int = settings.embedding_batch_size
    timeout: float = settings.embedding_timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.base_url:
            raise EmbeddingRequestError(
                "EMBEDDING_API_URL is not set - point openai_compatible at an "
                "OpenAI-compatible /embeddings server (e.g. Ollama/vLLM) or switch "
                "EMBEDDING_PROVIDER=mock for offline development."
            )
        endpoint = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                payload = {"model": self.model, "input": batch}
                try:
                    response = client.post(endpoint, json=payload, headers=headers)
                except httpx.HTTPError as exc:
                    raise EmbeddingRequestError(
                        f"Embedding request failed for batch {start // self.batch_size + 1}: {exc}"
                    ) from exc
                if response.status_code != 200:
                    raise EmbeddingRequestError(
                        f"Embedding endpoint returned HTTP {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                vectors.extend(self._parse(response.json(), expected=len(batch), batch_no=start // self.batch_size + 1))
        return vectors

    def _parse(self, data: Any, *, expected: int, batch_no: int) -> list[list[float]]:
        entries = []
        if isinstance(data, dict):
            entries = data.get("data") or []
        elif isinstance(data, list):  # some gateways return a bare list
            entries = data
        if len(entries) != expected:
            raise EmbeddingRequestError(
                f"Embedding batch {batch_no}: expected {expected} vectors, got {len(entries)}"
            )
        vectors = []
        for i, entry in enumerate(entries):
            vec = entry.get("embedding") if isinstance(entry, dict) else entry
            if not isinstance(vec, list) or len(vec) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"Embedding batch {batch_no}, item {i}: expected {self.dimensions}-d "
                    f"vector (must match chunks.embedding VECTOR({self.dimensions})), "
                    f"got {len(vec) if isinstance(vec, list) else 'non-list'}. "
                    f"Set EMBEDDING_MODEL to a {self.dimensions}-dim model."
                )
            vectors.append([float(x) for x in vec])
        return vectors


@dataclass
class GeminiEmbedder:
    """Gemini embeddings via the google-genai SDK (``models.embed_content``).

    MRL truncation (Gemini's ``outputDimensionality``) is a NATIVE REQUEST
    PARAMETER - the API truncates server-side to the requested size. Per the
    Gemini docs we do NOT slice client-side; we always ask for
    ``output_dimensionality=self.dimensions`` (default 1024) to match the
    ``chunks.embedding VECTOR(1024)`` column exactly.

    One caveat per the docs: the truncated MRL output is NOT renormalized by the
    API, so we L2-renormalize the returned vector ourselves (cosine similarity
    in pgvector assumes unit vectors). This is a normalization step, not a
    truncation step.

    Batching: ``embed_content`` accepts a list of contents per request; Gemini
    caps a single embedding request at 100 inputs, so ``batch_size`` defaults to
    the app's ``EMBEDDING_BATCH_SIZE`` (64) which is under the limit.
    """

    name: str = "gemini"
    api_key: str = ""
    model: str = settings.embedding_model
    dimensions: int = settings.embedding_dimensions
    batch_size: int = settings.embedding_batch_size
    task_type: str = "RETRIEVAL_DOCUMENT"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or settings.gemini_api_key
        if self.api_key:
            self._client = self._make_client()
        else:
            self._client = None

    def _make_client(self):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - import guarded at runtime
            raise EmbeddingRequestError(
                "google-genai is not installed. Add 'google-genai' to "
                "requirements.txt and pip install it, or switch "
                "EMBEDDING_PROVIDER=mock for offline development."
            ) from exc
        return genai.Client(api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._client:
            raise EmbeddingRequestError(
                "GEMINI_API_KEY is not set - set it to use the gemini embedding "
                "provider, or switch EMBEDDING_PROVIDER=mock for offline development."
            )
        try:
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingRequestError(
                "google-genai is not installed. Add 'google-genai' to "
                "requirements.txt and pip install it."
            ) from exc

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_no = start // self.batch_size + 1
            response = self._embed_batch_with_retry(batch, batch_no, genai_types)
            embeddings = response.embeddings  # list[ContentEmbedding]
            if not embeddings or len(embeddings) != len(batch):
                raise EmbeddingRequestError(
                    f"Gemini embedding batch {batch_no}: "
                    f"expected {len(batch)} vectors, got {len(embeddings) if embeddings else 0}"
                )
            for i, embedding in enumerate(embeddings):
                vec = embedding.values
                if not isinstance(vec, list) or len(vec) != self.dimensions:
                    raise EmbeddingDimensionError(
                        f"Gemini embedding batch {batch_no}, "
                        f"item {i}: expected {self.dimensions}-d vector (must match "
                        f"chunks.embedding VECTOR({self.dimensions})), "
                        f"got {len(vec) if isinstance(vec, list) else 'non-list'}. "
                        f"output_dimensionality={self.dimensions} was requested, so "
                        f"this is a model/API mismatch."
                    )
                vectors.append(_l2_normalize([float(x) for x in vec]))
        return vectors

    def _embed_batch_with_retry(self, batch: list[str], batch_no: int, genai_types: Any) -> Any:
        """Call embed_content with exponential backoff on 429 RESOURCE_EXHAUSTED.

        Only 429s (per-minute RPM limit) are retried — other errors propagate
        immediately. Each wait is logged at WARNING level so rate-limit events
        are visible without digging through SDK tracebacks.

        Attempts: _GEMINI_RETRY_ATTEMPTS total (original + retries).
        Delays:   _GEMINI_RETRY_BASE_S * (_GEMINI_RETRY_FACTOR ** attempt).
        e.g. defaults → 5s, 10s, 20s before giving up.
        """
        try:
            from google.genai import errors as genai_errors
        except ImportError:  # pragma: no cover
            genai_errors = None  # type: ignore[assignment]

        delay = _GEMINI_RETRY_BASE_S
        for attempt in range(_GEMINI_RETRY_ATTEMPTS):
            try:
                return self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=genai_types.EmbedContentConfig(
                        output_dimensionality=self.dimensions,
                        task_type=self.task_type,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                is_rate_limit = (
                    "429" in str(exc)
                    or "RESOURCE_EXHAUSTED" in str(exc)
                    or (genai_errors and isinstance(exc, genai_errors.ClientError) and getattr(exc, "status_code", None) == 429)
                )
                is_last_attempt = attempt == _GEMINI_RETRY_ATTEMPTS - 1
                if not is_rate_limit or is_last_attempt:
                    raise  # non-429, or retries exhausted → propagate as-is
                logger.warning(
                    "Gemini embedding 429 RESOURCE_EXHAUSTED on batch %d "
                    "(attempt %d/%d) — backing off %.0fs then retrying.",
                    batch_no, attempt + 1, _GEMINI_RETRY_ATTEMPTS, delay,
                )
                time.sleep(delay)
                delay *= _GEMINI_RETRY_FACTOR
        # Unreachable — the loop always returns or raises, but satisfies mypy.
        raise EmbeddingRequestError("Gemini embed_batch_with_retry: exhausted retries")  # pragma: no cover


def _l2_normalize(vec: list[float]) -> list[float]:
    """Unit-normalize ``vec`` in place of truncation (MRL output is not renormalized).

    Gemini truncates server-side; the returned truncated vector is NOT unit length,
    and cosine similarity assumes unit vectors, so we normalize after embedding.
    """
    norm = (sum(x * x for x in vec)) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


@dataclass
class MockEmbedder:
    """DEV ONLY: deterministic unit vectors so the pipeline runs with zero config.

    Not for real retrieval: vectors are content-independent (hash-seeded), so hybrid
    search over mock chunks only exercises the plumbing, never real semantics.
    """

    name: str = "mock"
    dimensions: int = settings.embedding_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_mock_vector(t, self.dimensions) for t in texts]


def _mock_vector(text: str, dimensions: int) -> list[float]:
    rng = random.Random(text)
    v = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
    norm = (sum(x * x for x in v)) ** 0.5 or 1.0
    return [x / norm for x in v]


def get_embedder(**overrides: Any) -> EmbeddingProvider:
    """Factory: EMBEDDING_PROVIDER selects the concrete provider implementation."""
    provider_name = overrides.get("provider", settings.embedding_provider)
    if provider_name == "openai_compatible":
        return OpenAICompatibleEmbedder(
            base_url=overrides.get("base_url", settings.embedding_api_url),
            api_key=overrides.get("api_key", settings.embedding_api_key),
            model=overrides.get("model", settings.embedding_model),
            dimensions=overrides.get("dimensions", settings.embedding_dimensions),
            batch_size=overrides.get("batch_size", settings.embedding_batch_size),
            timeout=overrides.get("timeout", settings.embedding_timeout),
        )
    if provider_name == "gemini":
        return GeminiEmbedder(
            api_key=overrides.get("api_key", settings.gemini_api_key),
            model=overrides.get("model", settings.embedding_model),
            dimensions=overrides.get("dimensions", settings.embedding_dimensions),
            batch_size=overrides.get("batch_size", settings.embedding_batch_size),
        )
    if provider_name == "mock":
        return MockEmbedder(
            dimensions=overrides.get("dimensions", settings.embedding_dimensions)
        )
    raise EmbeddingRequestError(
        f"Unknown EMBEDDING_PROVIDER={provider_name!r} (expected openai_compatible | gemini | mock)"
    )


def embed_chunks(
    provider: EmbeddingProvider, contents: list[str], batch_size: int | None = None
) -> list[list[float]]:
    """Embed many chunk texts, batched by the provider (never one call per text)."""
    if batch_size is None:
        batch_size = getattr(provider, "batch_size", settings.embedding_batch_size)
    vectors: list[list[float]] = []
    for start in range(0, len(contents), batch_size):
        vectors.extend(
            provider.embed(contents[start : start + batch_size])
        )
    return vectors


def replace_chunks_for_session(
    db, session_id: str, chunks: list, vectors: list[list[float]]
):
    """Idempotent chunk writer: keyed on session_id via delete-and-replace.

    Deleting that session's chunks before inserting guarantees re-embedding a
    session never creates duplicates (e.g. when the same export is imported again).
    """
    from sqlalchemy import delete

    existing = db.execute(delete(Chunk).where(Chunk.session_id == session_id))
    for chunk, vector in zip(chunks, vectors):
        # Keep the Phase-1 message_id FK useful: backfill chunks covering exactly one
        # message point at it; multi-message chunks use only message_ids.
        db.add(
            Chunk(
                session_id=session_id,
                message_id=chunk.message_ids[0] if len(chunk.message_ids) == 1 else None,
                message_ids=chunk.message_ids,
                content=chunk.content,
                embedding=vector,
            )
        )
    return existing.rowcount