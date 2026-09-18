"""Live Gemini API tests - run ONLY when GEMINI_API_KEY is set.

These exercise the real google-genai SDK against the Google API, so they are
skipped (not failed) when no key is present. Local/CI runs that define no
``GEMINI_API_KEY`` stay fully offline.
"""

import os
import sys
import unittest

sys.path.insert(0, "/home/nate/promptcraft_chatbot/unipods-bot/backend")

from app.config import settings  # noqa: E402
from app.ingest import embedder  # noqa: E402
from app.retrieval import answer  # noqa: E402

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

_live = unittest.skipUnless(
    GEMINI_API_KEY,
    "GEMINI_API_KEY not set; skipping live Gemini API tests",
)


@_live
class LiveGeminiEmbedderTests(unittest.TestCase):
    def setUp(self):
        settings.embedding_provider = "gemini"
        settings.gemini_api_key = GEMINI_API_KEY
        settings.embedding_model = "gemini-embedding-001"
        settings.embedding_dimensions = 1024

    def tearDown(self):
        settings.embedding_provider = "mock"
        settings.gemini_api_key = ""

    def test_live_embedding_is_1024d(self):
        provider = embedder.get_embedder(provider="gemini")
        out = provider.embed(
            ["What happened in the group yesterday?", "Design discussion about pgvector schemas."]
        )
        self.assertEqual(len(out), 2)
        for vec in out:
            self.assertEqual(len(vec), 1024)
            self.assertAlmostEqual(sum(x * x for x in vec), 1.0, places=4)


@_live
class LiveGeminiAnswerTests(unittest.TestCase):
    def setUp(self):
        settings.gemini_api_key = GEMINI_API_KEY
        settings.answer_provider = "gemini"
        settings.answer_model = "gemini-2.5-flash"

    def tearDown(self):
        settings.gemini_api_key = ""
        settings.answer_provider = "extractive"

    def test_live_gemini_answer_is_nonempty_text(self):
        context = (
            "• **Design session** — 23 messages.\n"
            "   – *\"We chose pgvector with a 1024-dim embedding column\"* "
            "(Nate, 2026-09-01)\n"
            "   – *\"Gemini supports MRL output_dimensionality truncation natively\"* "
            "(Alex, 2026-09-01)\n"
        )
        text = answer._gemini_synthesize("What did we decide about embeddings?", context)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text.strip()), 0)
        self.assertNotEqual(text, context)  # synthesized, not the raw bullets


if __name__ == "__main__":
    unittest.main()