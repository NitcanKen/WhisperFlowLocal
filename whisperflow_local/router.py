"""LLM backend router with remote-only and legacy fallback modes.

Exposes the same interface as a single backend (`format_text`, `propose_edits`,
`run_command`, `ping`, `.model`) so callers are unchanged.

Modes (config `llm_backend`):
- "remote": every call goes to vLLM. Failures propagate to the app's
  deterministic cleanup path; Ollama is never contacted.
- "local": every call goes to the local Ollama backend; the remote is never
  contacted.
- "auto": every call goes to the remote first. On any `LLMUnavailable` (TTFT
  deadline, connection error, HTTP/stream error) the SAME call is retried on the
  local backend. After `threshold` consecutive fallbacks the router trips to a
  temporary local-only state (no remote attempts, no per-request stall); after
  `cooldown` seconds it re-probes the remote on the next call and, on success,
  switches back.

Only `LLMUnavailable` triggers fallback — a `ValueError` (unknown AI command)
or other bug propagates unchanged.

The remote-primary + cooldown state lives in a shared `CircuitBreaker`
(`breaker.py`), which `ASRRouter` also composes.
"""
import time

from .breaker import CircuitBreaker
from .llm import LLMUnavailable


class LLMRouter:
    def __init__(self, *, local, remote, backend="auto", threshold=3,
                 cooldown=300.0, clock=time.monotonic, notify=None):
        self.local = local
        self.remote = remote
        self.backend = backend            # "remote" | "auto" | "local"
        self._breaker = CircuitBreaker(threshold=threshold, cooldown=cooldown,
                                       clock=clock)
        self._notify = notify             # notify(event, model): "fallback"|"reconnected"
        self._last_backend = remote if backend == "remote" else local

    # --- public interface (mirrors a backend) ------------------------------
    def format_text(self, text, profile, vocab=None):
        return self._dispatch("format_text", text, profile, vocab=vocab)

    def propose_edits(self, text, vocab=None):
        return self._dispatch("propose_edits", text, vocab=vocab)

    def run_command(self, command, text):
        return self._dispatch("run_command", command, text)

    def ping(self):
        if self.backend == "remote":
            return self.remote.ping()
        if self.backend == "local":
            return self.local.ping()
        return self.remote.ping() or self.local.ping()

    @property
    def model(self):
        """Model that served the most recent call (for logging)."""
        return self._last_backend.model

    # --- runtime configuration ---------------------------------------------
    def set_backend(self, backend):
        """Switch selection. Resets the breaker so a fresh choice starts clean."""
        self.backend = backend
        self._breaker.reset()

    def set_local_model(self, model):
        if self.local is not None:
            self.local.model = model

    def set_remote(self, url, model):
        self.remote.base_url = url.rstrip("/")
        self.remote.model = model

    # --- dispatch + breaker ------------------------------------------------
    def _dispatch(self, method, *args, **kwargs):
        if self.backend == "remote":
            self._last_backend = self.remote
            return getattr(self.remote, method)(*args, **kwargs)
        if self.backend == "local" or not self._breaker.allow_remote():
            self._last_backend = self.local
            return getattr(self.local, method)(*args, **kwargs)
        try:
            result = getattr(self.remote, method)(*args, **kwargs)
        except LLMUnavailable:
            if self._breaker.record_failure() and self._notify:
                self._notify("fallback", self.local.model)
            self._last_backend = self.local
            return getattr(self.local, method)(*args, **kwargs)
        if self._breaker.record_success() and self._notify:
            self._notify("reconnected", self.remote.model)
        self._last_backend = self.remote
        return result
