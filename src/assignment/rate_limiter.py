"""Per-user sliding-window rate limiter for availability and cost control."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Allow at most ``max_requests`` per user inside ``window_seconds``."""

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: int = 60,
        *,
        clock: Callable[[], float] | None = None,
    ):
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be at least 1")
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque[float]] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        self._clock = clock or time.monotonic
        self._lock = asyncio.Lock()
        self.last_decision = {
            "blocked": False,
            "user_id": None,
            "retry_after_seconds": 0.0,
        }

    @staticmethod
    def _block_response(message: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part.from_text(text=message)])

    def _evict_expired(self, window: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

    def check(self, user_id: str = "anonymous") -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)`` and record allowed hits."""
        now = self._clock()
        key = user_id or "anonymous"
        window = self.user_windows[key]
        self._evict_expired(window, now)
        if len(window) >= self.max_requests:
            retry_after = max(0.0, self.window_seconds - (now - window[0]))
            return False, retry_after
        window.append(now)
        return True, 0.0

    async def on_user_message_callback(self, *, invocation_context, user_message):
        del user_message
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        async with self._lock:
            self.total_count += 1
            allowed, retry_after = self.check(user_id)
            blocked = not allowed
            if blocked:
                self.blocked_count += 1
            self.last_decision = {
                "blocked": blocked,
                "user_id": user_id,
                "retry_after_seconds": retry_after,
            }
        if blocked:
            wait = max(1, int(retry_after + 0.999))
            return self._block_response(f"Rate limit exceeded. Try again in {wait}s.")
        return None

    def reset(self, user_id: str | None = None) -> None:
        if user_id is None:
            self.user_windows.clear()
        else:
            self.user_windows.pop(user_id, None)
