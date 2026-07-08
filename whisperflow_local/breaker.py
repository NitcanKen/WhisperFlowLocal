"""Shared circuit breaker for the remote-primary routers.

Both the LLM router (`router.LLMRouter`) and the ASR router
(`asr_router.ASRRouter`) follow the same policy: try the remote first, and after
`threshold` consecutive failures stop attempting it for `cooldown` seconds, then
re-probe once. This class owns that state so neither router duplicates it.

Thread-safe (dictations run on worker threads) and driven by an injectable
`clock` (`time.monotonic` in production) so the cooldown is deterministically
testable without sleeping.
"""
import threading
import time


class CircuitBreaker:
    def __init__(self, *, threshold=3, cooldown=300.0, clock=time.monotonic):
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive = 0
        self._temporary = False   # tripped: skipping the remote while cooling
        self._retry_at = 0.0

    def allow_remote(self) -> bool:
        """True unless the breaker is tripped and still within its cooldown.
        Once the cooldown has elapsed a single re-probe is allowed through."""
        with self._lock:
            if not self._temporary:
                return True
            return self._clock() >= self._retry_at

    def record_success(self) -> bool:
        """Reset the breaker. Returns True if it had been tripped (the caller
        uses that to fire a 'reconnected' notice)."""
        with self._lock:
            was_temporary = self._temporary
            self._consecutive = 0
            self._temporary = False
            self._retry_at = 0.0
        return was_temporary

    def record_failure(self) -> bool:
        """Count a remote failure. Returns True only on the call that first
        trips the breaker (the caller uses that to fire a 'fallback' notice).
        A failure while already tripped re-arms the cooldown."""
        with self._lock:
            self._consecutive += 1
            tripped_now = (not self._temporary
                           and self._consecutive >= self.threshold)
            if self._temporary or tripped_now:
                self._temporary = True
                self._retry_at = self._clock() + self.cooldown
        return tripped_now

    def reset(self) -> None:
        """Clear all state — used when the user manually switches backend."""
        with self._lock:
            self._consecutive = 0
            self._temporary = False
            self._retry_at = 0.0
