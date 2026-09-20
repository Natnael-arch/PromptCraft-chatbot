"""Tests for the Phase 3 voice path (offline / stubbed).

Mirrors test_phase3_gemini.py: the google-genai SDK is replaced with fakes and
no database is needed, so these run with NO API key and NO network. Real
transcription against the Gemini API is gated in test_live.py.

Coverage:
* upload mime validation (accept WhatsApp ogg/opus + common formats, reject junk)
* MM:SS timestamp parsing + malformed-response handling (raw preserved)
* duration probing (mutagen) from a synthesized WAV
* single-shot transcription, and windowed transcription merged onto one timeline
* voice sessionization: context header + chunking + segment refs for citations
* hybrid search scope actually includes voice chunks (compiled SQL assertion)
* voice citation building ("Speaker 2, 04:12-04:38")
"""

import hashlib
import io
import json
import re
import struct
import sys
import unittest
import wave
from unittest import mock

sys.path.insert(0, "/home/nate/promptcraft_chatbot/unipods-bot/backend")

from app.models import Chunk  # noqa: E402
from app.retrieval import answer, search  # noqa: E402
from app.voice import transcriber, voice_sessionizer  # noqa: E402


def make_wav(seconds: float, rate: int = 8000) -> bytes:
    """Build a silent mono PCM WAV of a known duration (mutagen probes it)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# transcriber: mime validation
# ---------------------------------------------------------------------------

class MimeValidationTests(unittest.TestCase):
    def test_whatsapp_ogg_and_opus_accepted(self):
        self.assertEqual(transcriber.normalize_mime_type("audio/ogg", "note.ogg"), "audio/ogg")
        self.assertEqual(transcriber.normalize_mime_type("audio/opus", "note.opus"), "audio/opus")

    def test_common_formats_accepted(self):
        for mime, name in [
            ("audio/mpeg", "a.mp3"),
            ("audio/mp4", "a.m4a"),
            ("audio/wav", "a.wav"),
            ("audio/webm", "a.webm"),
            ("audio/x-aac", "a.aac"),
            ("audio/flac", "a.flac"),
        ]:
            self.assertEqual(transcriber.normalize_mime_type(mime, name), mime)

    def test_generic_type_falls_back_to_extension(self):
        # WhatsApp clients sometimes send application/octet-stream for .opus files.
        self.assertEqual(
            transcriber.normalize_mime_type("application/octet-stream", "v.opus"),
            "audio/ogg",
        )
        self.assertEqual(
            transcriber.normalize_mime_type("application/ogg", "v.ogg"), "audio/ogg"
        )
        self.assertEqual(
            transcriber.normalize_mime_type("", "song.mp3"), "audio/mpeg"
        )

    def test_unsupported_types_rejected(self):
        for mime, name in [("text/plain", "notes.txt"), ("video/mp4", "clip.mp4"), ("image/png", "x.png")]:
            with self.assertRaises(transcriber.AudioFormatError):
                transcriber.normalize_mime_type(mime, name)


# ---------------------------------------------------------------------------
# transcriber: timestamps + response validation
# ---------------------------------------------------------------------------

class TimestampAndValidationTests(unittest.TestCase):
    def test_parse_mm_ss_and_hh_mm_ss(self):
        self.assertEqual(transcriber._parse_timestamp("04:12"), 252.0)
        self.assertEqual(transcriber._parse_timestamp("1:02:33"), 3753.0)
        self.assertEqual(transcriber._parse_timestamp(12.5), 12.5)

    def test_parse_bad_timestamp_raises(self):
        with self.assertRaises(transcriber.TranscriptionError):
            transcriber._parse_timestamp("not-a-time")

    def test_parse_segments_sorts_and_types(self):
        raw = {
            "segments": [
                {"speaker": "Speaker 2", "start": "00:10", "end": "00:12", "text": "second"},
                {"speaker": "Speaker 1", "start": "00:01", "end": "00:05", "text": "first"},
            ]
        }
        segs = transcriber._parse_segments(raw)
        self.assertEqual([s.text for s in segs], ["first", "second"])
        self.assertEqual(segs[0].start, 1.0)

    def test_malformed_shape_preserves_raw(self):
        for bad in [{"nope": 1}, {"segments": []}, {"segments": [{"speaker": "S1"}]}]:
            with self.assertRaises(transcriber.TranscriptionError) as ctx:
                transcriber._parse_segments(bad)
            self.assertEqual(ctx.exception.raw, bad)  # raw kept for debugging


# ---------------------------------------------------------------------------
# transcriber: duration + single/windowed transcription via a fake SDK client
# ---------------------------------------------------------------------------

class _FakeModels:
    def __init__(self, mode="single"):
        self.mode = mode
        self.calls: list[str] = []

    def generate_content(self, model, contents, config=None):
        prompt = contents[0]
        self.calls.append(prompt)
        if self.mode == "malformed":
            return type("R", (), {"text": "this is not json"})()
        match = re.search(r"from (\d+:\d+) to (\d+:\d+)", prompt)
        n = len(self.calls)
        if match:
            # Deliberately window-relative timestamps to exercise offset merging.
            seg = {"speaker": f"Speaker {n}", "start": "00:00", "end": "00:01", "text": f"window {n}"}
        else:
            seg = {"speaker": "Speaker 1", "start": "00:03", "end": "00:10", "text": "whole file"}
        return type("R", (), {"text": json.dumps({"segments": [seg]})})()


class _FakeClient:
    def __init__(self, mode="single"):
        self.models = _FakeModels(mode)
        self.files = type("Files", (), {"upload": lambda self, path: None})()


class TranscriptionTests(unittest.TestCase):
    def _transcriber(self, mode="single", **kw):
        fake = _FakeClient(mode)
        with mock.patch.object(transcriber.GeminiTranscriber, "_make_client", return_value=fake):
            t = transcriber.GeminiTranscriber(api_key="fake-key", **kw)
        return t, fake

    def test_duration_probe(self):
        wav = make_wav(1.5)
        self.assertAlmostEqual(transcriber.read_duration(wav), 1.5, delta=0.05)
        self.assertIsNone(transcriber.read_duration(b"not audio"))

    def test_single_shot_transcription(self):
        t, fake = self._transcriber()
        result = t.transcribe(make_wav(1.0), "audio/wav", filename="note.wav")
        self.assertFalse(result.windowed)
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].speaker, "Speaker 1")
        self.assertTrue(result.raw_response.get("segments"))
        self.assertEqual(len(fake.models.calls), 1)

    def test_windowed_transcription_merges_timeline(self):
        # 2.5s recording, 1s windows -> 3 windows (the final partial one included).
        t, fake = self._transcriber(window_seconds=1)
        result = t.transcribe(make_wav(2.5), "audio/wav", filename="call.wav")
        self.assertTrue(result.windowed)
        self.assertEqual(len(fake.models.calls), 3)
        self.assertEqual(len(result.segments), 3)
        starts = [s.start for s in result.segments]
        self.assertEqual(starts, [0.0, 1.0, 2.0])  # relative->absolute offset applied
        speakers = [s.speaker for s in result.segments]
        self.assertEqual(speakers, ["Speaker 1", "Speaker 2", "Speaker 3"])

    def test_malformed_response_raises_with_raw(self):
        t, fake = self._transcriber(mode="malformed")
        with self.assertRaises(transcriber.TranscriptionError) as ctx:
            t.transcribe(make_wav(1.0), "audio/wav")
        self.assertEqual(ctx.exception.raw, {"raw_text": "this is not json"})

    def test_missing_key_raises(self):
        t = transcriber.GeminiTranscriber(api_key="")
        with self.assertRaises(transcriber.TranscriptionRequestError):
            t.transcribe(make_wav(1.0), "audio/wav")

    def test_sha256_is_the_idempotency_key(self):
        self.assertEqual(
            transcriber._sha256(b"same-bytes"), hashlib.sha256(b"same-bytes").hexdigest()
        )
        self.assertNotEqual(transcriber._sha256(b"a"), transcriber._sha256(b"b"))


# ---------------------------------------------------------------------------
# voice_sessionizer
# ---------------------------------------------------------------------------

def _segments(n: int, text_len: int = 20):
    return [
        {
            "speaker": f"Speaker {i % 2 + 1}",
            "start": float(i * 5),
            "end": float(i * 5 + 4),
            "text": f"segment {i} " + "x" * text_len,
        }
        for i in range(n)
    ]


class VoiceSessionizerTests(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(voice_sessionizer.format_timestamp(0), "00:00")
        self.assertEqual(voice_sessionizer.format_timestamp(252), "04:12")
        self.assertEqual(voice_sessionizer.format_timestamp(3753), "1:02:33")

    def test_header_and_parse_roundtrip(self):
        header = voice_sessionizer.build_voice_header(
            _segments(4), duration_seconds=252
        )
        self.assertIn("Voice call", header)
        self.assertIn("Speaker 1, Speaker 2", header)
        parsed = voice_sessionizer.parse_chunk_header(header)
        self.assertIn("Speaker 1, Speaker 2", parsed["speakers"])
        self.assertEqual(parsed["duration"], "04:12")

    def test_chunks_respect_char_budget_and_keep_segment_refs(self):
        header = voice_sessionizer.build_voice_header(_segments(10), duration_seconds=100)
        chunks = voice_sessionizer.build_voice_chunks(
            _segments(10), header=header, max_chunk_chars=400, segments_per_chunk=100
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.content.startswith(header))
            self.assertLessEqual(len(chunk.content), 400 + len(header) + 2)
            self.assertTrue(chunk.segments)
            # every chunk carries the exact segments it was built from (citations)
            for seg in chunk.segments:
                self.assertIn("speaker", seg)

    def test_segments_per_chunk_cap(self):
        header = voice_sessionizer.build_voice_header(_segments(10))
        chunks = voice_sessionizer.build_voice_chunks(
            _segments(10), header=header, max_chunk_chars=100000, segments_per_chunk=4
        )
        self.assertEqual([len(c.segments) for c in chunks], [4, 4, 2])

    def test_parse_non_voice_content_is_safe(self):
        self.assertEqual(
            voice_sessionizer.parse_chunk_header("Session · 2026-01-01 · 3 messages")["speakers"],
            "",
        )


# ---------------------------------------------------------------------------
# retrieval integration: hybrid search must include voice chunks
# ---------------------------------------------------------------------------

class _CaptureDB:
    """Fake Session that captures the compiled-in-progress statement."""

    def __init__(self):
        self.stmts = []

    def execute(self, stmt):
        self.stmts.append(stmt)
        return type("Result", (), {"all": lambda self: [], "scalars": lambda self: self})()


class SearchScopeTests(unittest.TestCase):
    def _compiled(self, fn, **kw):
        db = _CaptureDB()
        fn(db, "group@g.us", *kw["args"], **kw["kwargs"])
        from sqlalchemy.dialects import postgresql

        return str(db.stmts[0].compile(dialect=postgresql.dialect()))

    def test_vector_search_query_covers_text_and_voice(self):
        sql = self._compiled(
            search.vector_search, args=([0.0] * 1024,), kwargs={"k": 5}
        )
        self.assertIn("LEFT OUTER JOIN sessions", sql)
        self.assertIn("LEFT OUTER JOIN recordings", sql)
        self.assertIn("recordings.chat_id", sql)
        self.assertIn("sessions.chat_id", sql)

    def test_keyword_search_query_covers_text_and_voice(self):
        sql = self._compiled(
            search.keyword_search, args=("launch date",), kwargs={"k": 5}
        )
        self.assertIn("LEFT OUTER JOIN recordings", sql)
        self.assertIn("recordings.chat_id", sql)
        self.assertIn("to_tsvector", sql)


# ---------------------------------------------------------------------------
# answer: voice citations
# ---------------------------------------------------------------------------

class VoiceCitationTests(unittest.TestCase):
    def test_voice_chunk_source_renders_speaker_and_window(self):
        chunk = Chunk(
            source_type="voice",
            voice_segments=[
                {"speaker": "Speaker 2", "start": 252.0, "end": 278.0, "text": "ship friday"},
                {"speaker": "Speaker 1", "start": 280.0, "end": 300.0, "text": "agreed"},
            ],
        )
        info = answer._voice_chunk_source(chunk)
        self.assertEqual(info["speaker"], "Speaker 2")
        self.assertEqual(info["segment_start"], 252.0)
        self.assertEqual(info["segment_end"], 300.0)
        self.assertEqual(info["segment"], "Speaker 2, 04:12\u201305:00")

    def test_voice_chunk_source_is_safe_without_segments(self):
        info = answer._voice_chunk_source(Chunk(source_type="voice", voice_segments=None))
        self.assertIsNone(info["speaker"])
        self.assertIsNone(info["segment"])


if __name__ == "__main__":
    unittest.main()
