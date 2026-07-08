"""ASR router: remote Qwen3-ASR primary + local SenseVoice fallback + breaker.

Mirrors `router.LLMRouter` for the ASR layer and exposes the exact surface
`app.py` already calls on `self.asr` (`transcribe`, `ensure_loaded`,
`engine_name`, `set_engine`), so the pipeline is unchanged.

Modes (config `asr_engine`):
- "sensevoice": every utterance is transcribed locally by SenseVoice; the remote
  is never contacted.
- "qwen3": the remote Qwen3-ASR is primary. On any `ASRUnavailable` (unreachable
  box, timeout, malformed response) the SAME utterance is transcribed by the local
  SenseVoice instead. After `threshold` consecutive fallbacks the shared breaker
  trips to local-only (no per-utterance stall); after `cooldown` seconds the next
  utterance re-probes the remote and, on success, switches back.

The local SenseVoice engine is the resident fallback and is always loaded, so a
fallback is instant; the remote holds no local model memory (no MPS pressure).
"""
import time

from .asr import ASRUnavailable
from .breaker import CircuitBreaker


class ASRRouter:
    def __init__(self, *, local, remote, backend="auto", threshold=3,
                 cooldown=300.0, clock=time.monotonic, notify=None):
        self.local = local               # SenseVoiceEngine (resident fallback)
        self.remote = remote             # RemoteQwenASRBackend
        self.backend = backend           # "auto" | "local"
        self._breaker = CircuitBreaker(threshold=threshold, cooldown=cooldown,
                                       clock=clock)
        self._notify = notify            # notify(event, engine): "fallback"|"reconnected"
        self._last_engine = local        # backs engine_name (who served last)

    # --- surface app.py already uses on self.asr ---------------------------
    @property
    def engine_name(self) -> str:
        """Engine that served the most recent transcription (so the log line
        reflects a mid-call fallback)."""
        return self._last_engine.name

    def ensure_loaded(self, progress_cb=None) -> None:
        """Warm the local SenseVoice engine — the resident fallback in both
        modes. The remote needs no local load."""
        self.local.ensure_loaded(progress_cb)

    def transcribe(self, wav_path: str, language: str = "auto",
                   context: list = None) -> str:
        if self.backend == "local" or not self._breaker.allow_remote():
            self._last_engine = self.local
            return self.local.transcribe(wav_path, language, context)
        try:
            result = self.remote.transcribe(wav_path, language, context)
        except ASRUnavailable:
            if self._breaker.record_failure() and self._notify:
                self._notify("fallback", self.local.name)
            self._last_engine = self.local
            return self.local.transcribe(wav_path, language, context)
        if self._breaker.record_success() and self._notify:
            self._notify("reconnected", self.remote.name)
        self._last_engine = self.remote
        return result

    # --- runtime configuration ---------------------------------------------
    def set_engine(self, code: str) -> None:
        """Menu selection: "qwen3" -> remote-primary auto, else local-only.
        Resets the breaker so a fresh choice starts clean."""
        self.backend = "auto" if code == "qwen3" else "local"
        self._breaker.reset()

    def set_remote(self, url: str, model: str) -> None:
        self.remote.base_url = url.rstrip("/")
        self.remote.model = model

    def ping(self) -> bool:
        if self.backend == "local":
            return True  # SenseVoice is local; always reachable
        return self.remote.ping()
