"""LLM backend router: remote vLLM primary + local Ollama fallback + breaker.

Exposes the same interface as a single backend (`format_text`, `propose_edits`,
`run_command`, `ping`, `.model`) so callers are unchanged.

Modes (config `llm_backend`):
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
"""
import threading
import time

from .llm import LLMUnavailable


class LLMRouter:
    def __init__(self, *, local, remote, backend="auto", threshold=3,
                 cooldown=300.0, clock=time.monotonic, notify=None):
        self.local = local
        self.remote = remote
        self.backend = backend            # "auto" | "local"
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self._notify = notify             # notify(event, model): "fallback"|"reconnected"
        self._lock = threading.Lock()
        self._consecutive = 0
        self._temporary = False           # breaker tripped (auto mode only)
        self._retry_at = 0.0
        self._last_backend = local        # backs the .model property (logging)

    # --- public interface (mirrors a backend) ------------------------------
    def format_text(self, text, profile, vocab=None):
        return self._dispatch("format_text", text, profile, vocab=vocab)

    def propose_edits(self, text, vocab=None):
        return self._dispatch("propose_edits", text, vocab=vocab)

    def run_command(self, command, text):
        return self._dispatch("run_command", command, text)

    def ping(self):
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
        with self._lock:
            self.backend = backend
            self._temporary = False
            self._consecutive = 0
            self._retry_at = 0.0

    def set_local_model(self, model):
        self.local.model = model

    def set_remote(self, url, model):
        self.remote.base_url = url.rstrip("/")
        self.remote.model = model

    # --- dispatch + breaker ------------------------------------------------
    def _dispatch(self, method, *args, **kwargs):
        if self.backend == "local" or not self._should_try_remote():
            self._last_backend = self.local
            return getattr(self.local, method)(*args, **kwargs)
        try:
            result = getattr(self.remote, method)(*args, **kwargs)
        except LLMUnavailable:
            self._on_remote_failure()
            self._last_backend = self.local
            return getattr(self.local, method)(*args, **kwargs)
        self._on_remote_success()
        self._last_backend = self.remote
        return result

    def _should_try_remote(self):
        """auto mode: try the remote unless the breaker is tripped and still
        cooling down. A probe is allowed once the cooldown has elapsed."""
        with self._lock:
            if not self._temporary:
                return True
            return self._clock() >= self._retry_at

    def _on_remote_success(self):
        with self._lock:
            was_temporary = self._temporary
            self._consecutive = 0
            self._temporary = False
            self._retry_at = 0.0
        if was_temporary and self._notify:
            self._notify("reconnected", self.remote.model)

    def _on_remote_failure(self):
        with self._lock:
            self._consecutive += 1
            tripped_now = (not self._temporary
                           and self._consecutive >= self.threshold)
            if self._temporary or tripped_now:
                # Either a probe just failed, or we just tripped: (re)arm cooldown.
                self._temporary = True
                self._retry_at = self._clock() + self.cooldown
        if tripped_now and self._notify:
            self._notify("fallback", self.local.model)
