"""ASR engines: SenseVoiceSmall (funasr) and Qwen3-ASR (native hotword bias).

Both download on first use and are cached (modelscope / huggingface hubs),
so later runs are offline. The ASREngine facade routes to the configured
engine and silently falls back to SenseVoice if Qwen3-ASR cannot load —
dictation must keep working even when the bigger model is missing.

Memory: only ONE engine is resident at a time — switching engines unloads
the other (SenseVoice ~1 GB, Qwen3-ASR ~3.5-4 GB on MPS; the target machine
has 16 GB and Ollama may hold another ~3.4 GB).
"""
import gc
import threading

from .applog import log
from .textproc import strip_sensevoice_tags

# SPEC language modes -> SenseVoice language argument.
# SenseVoice handles intra-sentence code-switching natively in auto mode,
# so Mixed maps to auto; yue/en force a primary language.
LANGUAGE_MAP = {
    "auto": "auto",
    "mixed": "auto",
    "yue": "yue",
    "en": "en",
}

# SPEC language modes -> Qwen3-ASR language names (None = auto-detect;
# Qwen3-ASR code-switches natively, so Mixed also maps to auto).
QWEN_LANGUAGE_MAP = {
    "auto": None,
    "mixed": None,
    "yue": "Cantonese",
    "en": "English",
}

ENGINES = ("sensevoice", "qwen3")


def _release_memory() -> None:
    """Return freed model memory to the OS (MPS keeps a private pool)."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class SenseVoiceEngine:
    """Lazy-loading wrapper around funasr AutoModel(SenseVoiceSmall)."""

    name = "sensevoice"

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def ensure_loaded(self, progress_cb=None) -> None:
        with self._lock:
            if self._model is not None:
                return
            if progress_cb:
                progress_cb("Loading SenseVoiceSmall (downloads on first run)…")
            from funasr import AutoModel

            self._model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cpu",
                disable_update=True,
            )

    def unload(self) -> None:
        with self._lock:
            self._model = None
        _release_memory()

    def transcribe(self, wav_path: str, language: str = "auto",
                   context: list = None) -> str:
        """context is accepted for interface parity; SenseVoice has no
        contextual biasing, so it is ignored here."""
        self.ensure_loaded()
        result = self._model.generate(
            input=wav_path,
            cache={},
            language=LANGUAGE_MAP.get(language, "auto"),
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        if not result:
            return ""
        raw = " ".join(seg.get("text", "") for seg in result)
        return strip_sensevoice_tags(raw)


class Qwen3ASREngine:
    """Qwen3-ASR-1.7B via the qwen-asr package: better Cantonese accuracy
    and native context biasing (the vocabulary list goes straight into the
    recognizer)."""

    name = "qwen3"
    MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def ensure_loaded(self, progress_cb=None) -> None:
        with self._lock:
            if self._model is not None:
                return
            if progress_cb:
                progress_cb("Loading Qwen3-ASR-1.7B (downloads on first run)…")
            import torch
            from qwen_asr import Qwen3ASRModel

            kwargs = {}
            if torch.backends.mps.is_available():
                kwargs = {"dtype": torch.bfloat16, "device_map": "mps"}
            try:
                self._model = Qwen3ASRModel.from_pretrained(
                    self.MODEL_ID, max_inference_batch_size=1, **kwargs
                )
            except Exception:
                if not kwargs:
                    raise
                log("asr", "Qwen3-ASR MPS load failed — retrying on CPU")
                self._model = Qwen3ASRModel.from_pretrained(
                    self.MODEL_ID, max_inference_batch_size=1
                )

    def unload(self) -> None:
        with self._lock:
            self._model = None
        _release_memory()

    def transcribe(self, wav_path: str, language: str = "auto",
                   context: list = None) -> str:
        self.ensure_loaded()
        results = self._model.transcribe(
            wav_path,
            context=", ".join(context) if context else "",
            language=QWEN_LANGUAGE_MAP.get(language),
        )
        if not results:
            return ""
        return (results[0].text or "").strip()


class ASREngine:
    """Facade: routes to the configured engine, falls back to SenseVoice."""

    def __init__(self, engine: str = "sensevoice"):
        self._engines = {"sensevoice": SenseVoiceEngine(),
                         "qwen3": Qwen3ASREngine()}
        self._name = engine if engine in self._engines else "sensevoice"
        self.loading = False

    @property
    def engine_name(self) -> str:
        return self._name

    def set_engine(self, name: str) -> None:
        """Select an engine and unload the others — only one model stays
        resident (16 GB machines cannot hold both plus Ollama)."""
        if name not in self._engines:
            return
        self._name = name
        for other_name, other in self._engines.items():
            if other_name != name:
                other.unload()

    def _active(self, progress_cb=None):
        """Load the selected engine, degrading to SenseVoice on failure."""
        engine = self._engines[self._name]
        try:
            engine.ensure_loaded(progress_cb)
            return engine
        except Exception as exc:
            if self._name == "sensevoice":
                raise
            log("asr", f"{self._name} load FAILED ({exc}) — "
                       "falling back to SenseVoice")
            self._name = "sensevoice"
            fallback = self._engines["sensevoice"]
            fallback.ensure_loaded(progress_cb)
            return fallback

    def ensure_loaded(self, progress_cb=None) -> None:
        self.loading = True
        try:
            self._active(progress_cb)
        finally:
            self.loading = False

    def transcribe(self, wav_path: str, language: str = "auto",
                   context: list = None) -> str:
        return self._active().transcribe(wav_path, language, context)
