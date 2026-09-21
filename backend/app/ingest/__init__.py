"""Phase 2 chat-export backfill pipeline: parser, sessionizer, embedder.

``rebuild_sessions_and_chunks`` is the shared session->embed->replace pipeline
used by both the manual export import and the live-ingest background task.
"""

from app.ingest.rebuild import rebuild_sessions_and_chunks  # noqa: F401