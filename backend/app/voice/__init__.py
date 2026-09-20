"""Phase 3 voice ingestion: transcriber -> voice_sessionizer -> embedder.

A voice note is uploaded, transcribed by Gemini (diarized + timestamped),
sessionized into chunk-ready segments (context-header treatment, same char
budget as text sessions), and embedded through the SAME embedder.py used by the
Phase 2 text path - voice chunks land in `chunks` with source_type='voice' and a
recording_id instead of a session_id.
"""