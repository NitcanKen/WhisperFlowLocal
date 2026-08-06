"""Persistent JSON configuration: hotkeys, language, LLM, dictionary, app rules."""
import json
import os
import threading

from . import paths

DEFAULTS = {
    # Push-to-talk: hold this key to record (pynput key name, e.g. alt_r, cmd_r, f18)
    "ptt_key": "alt_r",
    # Hands-free toggle combo (pynput GlobalHotKeys syntax)
    "toggle_hotkey": "<cmd>+<shift>+d",
    # ASR language mode: auto | yue | en | mixed
    "language": "auto",
    # ASR engine selection (mirrors llm_backend):
    #   "qwen3"      = remote Qwen3-ASR (vLLM /v1/audio/transcriptions) primary +
    #                  local SenseVoice fallback + circuit breaker (see asr_router.py).
    #   "sensevoice" = SenseVoice only; the remote is never contacted.
    "asr_engine": "sensevoice",
    # Optional Qwen3-ASR endpoint (vLLM OpenAI audio API). Local-only by
    # default; change this in config.json when opting into a trusted remote.
    "qwen_asr_url": "http://127.0.0.1:8001/v1",
    "qwen_asr_model": "Qwen/Qwen3-ASR-1.7B",
    # Optional bearer token. WHISPERFLOW_QWEN_ASR_API_KEY takes precedence.
    "qwen_asr_api_key": "",
    # Fast-fail an unreachable box on connect; generous read cap for one-shot
    # transcription so a legitimately-working request is never cut off.
    "qwen_asr_connect_timeout": 1.0,
    "qwen_asr_total_timeout": 30.0,
    "llm_enabled": True,
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen3.5:4b",
    # LLM backend selection:
    #   "auto"  = remote vLLM (OpenAI /v1) primary + local Ollama fallback +
    #             circuit breaker (see router.py).
    #   "local" = Ollama only; the remote is never contacted.
    "llm_backend": "local",
    "vllm_url": "http://127.0.0.1:8000/v1",
    "vllm_model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    # Optional bearer token. WHISPERFLOW_VLLM_API_KEY takes precedence.
    "vllm_api_key": "",
    # Fall back to local if the remote can't connect within this many seconds,
    # or sends no first token within the TTFT deadline (streaming). A generous
    # total cap only guards a stream that dribbles forever.
    "vllm_connect_timeout": 1.0,
    "vllm_ttft_timeout": 1.0,
    "vllm_total_timeout": 30.0,
    # After this many consecutive fallbacks, pin to local; re-probe the remote
    # after fallback_cooldown seconds (auto-retry).
    "fallback_threshold": 3,
    "fallback_cooldown": 300.0,
    # Default formatting profile: Raw | Clean | Email | Message | Notes
    "profile": "Clean",
    "copy_only": False,
    "sounds": True,
    # Output punctuation on/off (applies after ASR + LLM formatting)
    "punctuation": True,
    # Convert output to Hong Kong traditional characters (OpenCC s2hk)
    "traditional_hk": True,
    # UI language: auto (follow macOS) | en | zh-HK
    "ui_language": "auto",
    # Custom dictionary: replacements applied to transcripts before formatting
    "dictionary": [
        {"from": "whisper flow", "to": "WhisperFlow"},
    ],
    # Bare hotword terms (no replacement): bias the LLM prompt and, for ASR
    # engines with context biasing, the recognizer itself
    "hotwords": [],
    # Frontmost-app name (substring, case-insensitive) -> profile
    "app_rules": {
        "Mail": "Email",
        "Messages": "Message",
        "Slack": "Message",
        "Discord": "Message",
        "WhatsApp": "Message",
        "Terminal": "Raw",
        "iTerm": "Raw",
        "Code": "Raw",
        "Notes": "Notes",
    },
    "onboarded": False,
}

_lock = threading.Lock()


class Config:
    """Thread-safe config store backed by a JSON file."""

    def __init__(self, path: str = paths.CONFIG_PATH):
        self.path = path
        self.data = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        with _lock:
            if os.path.exists(self.path):
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        stored = json.load(f)
                    merged = dict(DEFAULTS)
                    merged.update(stored)
                    self.data = merged
                except (json.JSONDecodeError, OSError):
                    # Corrupt config: keep defaults, do not crash the app.
                    self.data = dict(DEFAULTS)

    def save(self) -> None:
        with _lock:
            paths.ensure_dirs()
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                os.chmod(tmp, 0o600)
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)

    def get(self, key: str):
        return self.data.get(key, DEFAULTS.get(key))

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()
