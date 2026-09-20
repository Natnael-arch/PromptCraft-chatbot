"""Phase 4: live auto-replies over WAHA.

Flow (all logic lives here, webhook.py only schedules the worker):

    webhook.py captures the message (shared with /ask) and then hands the
    original payload to ``reply_worker.reply_to_captured`` as a FastAPI
    BackgroundTask - the HTTP response to WAHA stays instant.

    Worker pipeline: resolve the bot's own JID -> decide whether this message is
    addressed to us (DM always / group @-mention only) with trigger_detector ->
    per-chat in-memory cooldown -> answer via the same answer_question pipeline
    /ask uses -> send the reply back through WAHA's sendText endpoint.

No self-HTTP round-trip: the worker imports answer_question directly instead of
calling our own /ask endpoint.
"""