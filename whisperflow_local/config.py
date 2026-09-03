"""Persistent JSON configuration: hotkeys, language, LLM, dictionary, app rules."""
import json
import os
import threading

from . import paths

# Old five-profile installs map onto the two surviving modes.
_PROFILE_MIGRATION = {
    "Raw": "Verbatim", "Clean": "Verbatim", "Message": "Verbatim",
    "Email": "Structured", "Notes": "Structured",
}

DEFAULTS = {
    # Push-to-talk: hold this key to record (pynput key name, e.g. alt_r, cmd_r, f18)
    "ptt_key": "alt_r",
    # Hands-free toggle combo (pynput GlobalHotKeys syntax)
    "toggle_hotkey": "<cmd>+<shift>+d",
    # HOLD this instead of the bare ptt_key to dictate a content-generation
    # request (the LLM writes the text rather than transcribing it). Same
    # grammar as toggle_hotkey, but held, and its trigger may itself be a
    # modifier. Empty string disables generation.
    "generate_hotkey": "<shift>+<alt_r>",
    # ASR language mode: auto | yue | en | mixed
    "language": "auto",
    # ASR engine selection (mirrors llm_backend):
    #   "qwen3"      = remote Qwen3-ASR (vLLM /v1/audio/transcriptions) primary +
    #                  local SenseVoice fallback + circuit breaker (see asr_router.py).
    #   "sensevoice" = SenseVoice only; the remote is never contacted.
    "asr_engine": "qwen3",
    # Private Tailscale endpoint on the GB10. SenseVoice remains the ASR-only
    # fallback if the remote recognizer is unavailable.
    "qwen_asr_url": "http://100.71.138.54:8800/v1",
    "qwen_asr_model": "Qwen3-ASR-0.6B",
    # Optional bearer token. WHISPERFLOW_QWEN_ASR_API_KEY takes precedence.
    "qwen_asr_api_key": "",
    # Fast-fail an unreachable box on connect; generous read cap for one-shot
    # transcription so a legitimately-working request is never cut off.
    "qwen_asr_connect_timeout": 1.0,
    "qwen_asr_total_timeout": 30.0,
    "llm_enabled": True,
    # LLM backend selection is fixed to the private GB10. A remote failure
    # degrades to deterministic cleanup; the app never constructs Ollama.
    "llm_backend": "remote",
    "vllm_url": "http://100.71.138.54:8090/v1",
    "vllm_model": "Qwen3.6-35B-A3B",
    # Optional bearer token. WHISPERFLOW_VLLM_API_KEY takes precedence.
    "vllm_api_key": "",
    # Fall back to local if the remote can't connect within this many seconds,
    # or sends no first token within the TTFT deadline (streaming). A generous
    # total cap only guards a stream that dribbles forever.
    "vllm_connect_timeout": 3.0,
    "vllm_ttft_timeout": 20.0,
    "vllm_total_timeout": 60.0,
    # Reasoning-effort budget for reasoning-capable remotes (e.g. DeepSeek-V4).
    # "none" disables the chain-of-thought — essential for dictation latency:
    # with reasoning on, an utterance can take 6-60 s and the answer arrives in
    # the reasoning channel. Empty string omits the field for servers that
    # reject it. enable_thinking (Qwen) is sent separately and is ignored here.
    "vllm_reasoning_effort": "none",
    # After this many consecutive fallbacks, pin to local; re-probe the remote
    # after fallback_cooldown seconds (auto-retry).
    "fallback_threshold": 3,
    "fallback_cooldown": 300.0,
    # Formatting profile, chosen from the menu and never overridden:
    #   "Verbatim"   = keep the spoken wording; the LLM only repunctuates,
    #                  drops fillers/stutters and fixes ASR homophones.
    #   "Structured" = understand the utterance and re-emit it as structured
    #                  written Chinese.
    "profile": "Verbatim",
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
                    # Migrate legacy local/auto installs to the GB10-only route.
                    merged["llm_backend"] = "remote"
                    merged.pop("ollama_url", None)
                    merged.pop("ollama_model", None)
                    # Five profiles collapsed to two. An un-migrated legacy
                    # value would match no menu item: every checkmark would go
                    # dark and format_text would silently pass text through.
                    merged["profile"] = _PROFILE_MIGRATION.get(
                        merged.get("profile"), merged.get("profile"))
                    if merged.get("profile") not in ("Verbatim", "Structured"):
                        merged["profile"] = DEFAULTS["profile"]
                    # Per-app rules are gone: the menu choice is authoritative.
                    merged.pop("app_rules", None)
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
