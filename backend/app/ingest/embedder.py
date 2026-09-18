"""Embedding providers + writing embeddings into the `chunks` table.

The provider is swappable via the EMBEDDING_PROVIDER env var, but there is a real,
working default behind every abstraction here:

* ``openai_compatible`` - POSTs batches to any OpenAI-style ``/embeddings`` endpoint
  (a local Ollama/vLLM gateway, an OpenAI-compatible cloud host, ...). This is the
  provider to use for production. The default model name is bge-m3, which outputs
  1024-d vectors - exactly matching the ``chunks.embedding VECTOR(1024)`` schema.
  Set EMBEDDING_API_URL/API_KEY/MODEL to point it somewhere real.
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
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import settings
from app.models import Chunk


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
    if provider_name == "mock":
        return MockEmbedder(
            dimensions=overrides.get("dimensions", settings.embedding_dimensions)
        )
    raise EmbeddingRequestError(
        f"Unknown EMBEDDING_PROVIDER={provider_name!r} (expected openai_compatible | mock)"
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