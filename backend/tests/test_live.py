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
from app.voice import transcriber  # noqa: E402

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


def _tts_clip(phrase: str) -> tuple[bytes, str]:
    """Synthesize a short spoken clip with Gemini TTS for the voice round-trip."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=f"Say exactly this, clearly and at a normal pace: {phrase}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
        ),
    )
    part = resp.candidates[0].content.parts[0]
    mime = part.inline_data.mime_type or "audio/pcm"
    if "l16" in mime.lower() or "pcm" in mime.lower():
        mime = "audio/pcm"  # L16;rate=24000 -> the canonical PCM type we accept
    return part.inline_data.data, mime


@_live
class LiveGeminiVoiceTranscriberTests(unittest.TestCase):
    """End-to-end exercise of the real transcription path (TTS -> transcribe).

    Uses Gemini TTS to synthesize speech so the test is self-contained (no audio
    fixture checked into the repo). If the TTS model is unavailable in this
    environment the test skips rather than fails.
    """

    def setUp(self):
        settings.gemini_api_key = GEMINI_API_KEY
        settings.transcribe_model = "gemini-2.5-flash"

    def tearDown(self):
        settings.gemini_api_key = ""

    def test_live_transcription_returns_diarized_segments(self):
        try:
            audio, mime = _tts_clip("the deployment is on friday")
        except Exception as exc:  # noqa: BLE001 - TTS model availability varies
            self.skipTest(f"Gemini TTS unavailable: {exc}")

        provider = transcriber.get_transcriber(
            api_key=GEMINI_API_KEY, window_seconds=600
        )
        result = provider.transcribe(audio, mime, filename="tts_sample")

        self.assertTrue(result.segments, "expected at least one transcript segment")
        self.assertTrue(result.raw_response.get("segments"))
        for seg in result.segments:
            self.assertTrue(seg.speaker)
            self.assertLessEqual(seg.start, seg.end)
            self.assertTrue(seg.text.strip())

        words = " ".join(s.text.lower() for s in result.segments)
        self.assertIn("friday", words)


if __name__ == "__main__":
    unittest.main()