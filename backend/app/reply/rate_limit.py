"""In-memory per-chat reply cooldown.

Phase 4 deliberately keeps this process-local and naive (a dict + lock). It
neither survives restarts nor scales across replicas; the documented Phase 5
move replaces it with a Redis-backed rate limiter. The lock makes it safe to
share across FastAPI BackgroundTask threads.
"""

import threading
import time


class InMemoryCooldown:
    """Per-key cooldown window. ``check_and_spend`` is atomic and re-arms on hit."""

    def __init__(self, window_seconds: float, now: callable = time.monotonic):
        self.window_seconds = float(window_seconds)
        self._now = now
        self._last = {}
        self._lock = threading.Lock()

    def allowed(self, key: str, *, now: float | None = None) -> bool:
        """Return True if ``key`` may fire now; consumes a slot when True."""
        ts = now if now is not None else self._now()
        with self._lock:
            last = self._last.get(key)
            if last is not None and ts - last < self.window_seconds:
                return False
            self._last[key] = ts
            return True

    def clear(self) -> None:
        with self._lock:
            self._last.clear()