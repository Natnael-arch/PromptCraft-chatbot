"""Tests for the Phase 3 Gemini provider paths (offline / stubbed).

The google-genai SDK calls are replaced with fakes so these tests run with NO
real API key and NO network. Live/real-API tests are gated behind ``GEMINI_API_KEY``
being set in the environment (see ``test_live.py``).

The object under test is built via the same ``get_embedder()`` /
``answer_question()`` entry points the routes use, with the provider forced to
``gemini`` via overrides/config - exactly the code path ``EMBEDDING_PROVIDER=gemini``
and ``ANSWER_PROVIDER=gemini`` exercise.
"""

import sys
import unittest
from unittest import mock

sys.path.insert(0, "/home/nate/promptcraft_chatbot/unipods-bot/backend")

from app.config import settings  # noqa: E402
from app.ingest import embedder  # noqa: E402
from app.retrieval import answer  # noqa: E402


class _FakeContentEmbedding:
    """Mimics google.genai.types.ContentEmbedding (has .values)."""

    def __init__(self, values: list[float]):
        self.values = values


class _FakeModels:
    def __init__(self, text_eggs=("gemini-2.5-flash",)):
        self.text_eggs = text_eggs

    def embed_content(self, model, contents, config=None):
        # The native MRL truncation param must be present in the request.
        assert config is not None
        assert config.output_dimensionality == 1024
        return type(
            "EmbedContentResponse",
            (),
            {"embeddings": [_FakeContentEmbedding([3.0] * 1024) for _ in contents]},
        )()

    def generate_content(self, model, contents, config=None):
        return type("GenerateContentResponse", (), {"text": "stubbed gemini answer"})()


def _fake_client(*args, **kwargs):
    api_key = kwargs.get("api_key") or (args[0] if args else None) or getattr(
        settings, "gemini_api_key", "fake-key"
    )
    assert api_key  # never construct the client without a key
    return type("Client", (), {"models": _FakeModels()})()


class GeminiEmbedderTests(unittest.TestCase):
    def setUp(self):
        settings.embedding_provider = "gemini"
        settings.gemini_api_key = "fake-key"
        settings.embedding_model = "gemini-embedding-001"
        settings.embedding_dimensions = 1024

    def tearDown(self):
        settings.embedding_provider = "mock"
        settings.gemini_api_key = ""
        settings.answer_provider = "extractive"

    def test_factory_returns_gemini(self):
        provider = embedder.get_embedder(provider="gemini", api_key="fake-key")
        self.assertEqual(provider.name, "gemini")
        self.assertIsInstance(provider, embedder.GeminiEmbedder)

    @mock.patch.object(embedder.GeminiEmbedder, "_make_client", side_effect=_fake_client)
    def test_embed_requests_output_dimensionality_and_renormalizes(self, _mk):
        provider = embedder.get_embedder(provider="gemini")
        out = provider.embed(["hello", "world"])

        self.assertEqual(len(out), 2)                       # one vector per input
        for vec in out:
            self.assertEqual(len(vec), 1024)                # request-param truncation
            total = sum(x * x for x in vec)
            self.assertAlmostEqual(total, 1.0, places=6)    # L2 renormalized (MRL)

    @mock.patch.object(embedder.GeminiEmbedder, "_make_client", side_effect=_fake_client)
    def test_missing_key_raises(self, _mk):
        settings.gemini_api_key = ""
        provider = embedder.GeminiEmbedder(api_key="")
        with self.assertRaises(embedder.EmbeddingRequestError):
            provider.embed(["hello"])


class MockEmbedderStillWorksTests(unittest.TestCase):
    def test_mock_offline_1024d(self):
        settings.embedding_provider = "mock"
        provider = embedder.get_embedder(provider="mock")
        out = provider.embed(["hello"])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1024)
        self.assertAlmostEqual(sum(x * x for x in out[0]), 1.0, places=6)


class GeminiAnswerTests(unittest.TestCase):
    def setUp(self):
        settings.gemini_api_key = "fake-key"
        settings.answer_provider = "gemini"
        settings.answer_model = "gemini-2.5-flash"

    def tearDown(self):
        settings.gemini_api_key = ""
        settings.answer_provider = "extractive"

    @mock.patch("google.genai.Client", side_effect=_fake_client)
    def test_gemini_synthesize_returns_gemini_text(self, _client):
        text = answer._gemini_synthesize("question?", "context bullets here")
        self.assertEqual(text, "stubbed gemini answer")

    @mock.patch("google.genai.Client", side_effect=_fake_client)
    def test_gemini_synthesize_falls_back_without_key(self, _client):
        settings.gemini_api_key = ""
        text = answer._gemini_synthesize("question?", "context bullets here")
        self.assertEqual(text, "context bullets here")  # extractive fallback

    def test_extractive_provider_returns_input_unchanged(self):
        settings.answer_provider = "extractive"
        self.assertEqual(
            answer._apply_answer_provider("q?", "extractive bullets"),
            "extractive bullets",
        )


if __name__ == "__main__":
    unittest.main()