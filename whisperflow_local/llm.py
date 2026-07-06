"""Local LLM layer: qwen3.5 via the localhost Ollama API.

Qwen3-family models are hybrid reasoning models. Thinking is disabled with
the API-level `think: false` flag; if the server/model rejects that flag the
request is retried without it and any <think>...</think> block is stripped
from the output defensively.
"""
import re

import requests

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_COMMON_RULES = (
    "You are a dictation post-processor. The user dictated text that may mix "
    "Cantonese and English in the same sentence (code-switching). "
    "NEVER translate between languages; keep every word in the language it was "
    "spoken. Preserve English words embedded in Cantonese sentences exactly. "
    "Return ONLY the processed text with no explanations, no quotes, no labels."
)

PROFILES = {
    # Raw is handled upstream (LLM bypassed entirely).
    "Clean": _COMMON_RULES + (
        " Task: fix punctuation and capitalization, remove filler words "
        "(um, uh, er, 呃, 嗯, 即係 when used as filler), and fix obvious "
        "transcription slips. Do NOT rephrase, expand, or summarize."
    ),
    "Email": _COMMON_RULES + (
        " Task: format the dictation as polished email body text: proper "
        "sentences, paragraphs, punctuation, professional but natural tone. "
        "Remove fillers. Keep the original meaning and language mix. Do not "
        "invent a subject line, greeting, or signature unless dictated."
    ),
    "Message": _COMMON_RULES + (
        " Task: format as a casual chat message: light punctuation, keep it "
        "short and natural, remove fillers. Keep the original language mix."
    ),
    "Notes": _COMMON_RULES + (
        " Task: format the dictation as concise notes using bullet points "
        "(one idea per line, '-' bullets). Remove fillers. Keep the original "
        "language mix."
    ),
}

AI_COMMANDS = {
    "Formalize": _COMMON_RULES + (
        " Task: rewrite the text to be more formal and professional while "
        "keeping its meaning and its language mix."
    ),
    "Summarize": _COMMON_RULES + (
        " Task: summarize the text concisely in the same language(s) it is "
        "written in."
    ),
    "Translate to English": (
        "You are a precise translator. Translate the user's text to natural "
        "English. Return ONLY the translation."
    ),
    "Translate to Cantonese": (
        "You are a precise translator. Translate the user's text to natural "
        "written Cantonese (廣東話口語，用香港常用字). Return ONLY the translation."
    ),
}


class LLMUnavailable(Exception):
    """Raised when Ollama cannot be reached or the model is missing."""


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,  # disable Qwen3 hybrid reasoning
            "options": {"temperature": 0.2},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            if resp.status_code == 400 and "think" in resp.text.lower():
                # Model/server rejects the think flag: retry without it.
                payload.pop("think", None)
                resp = requests.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
                )
            if resp.status_code == 404:
                raise LLMUnavailable(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
        except requests.RequestException as exc:
            raise LLMUnavailable(
                f"Ollama unreachable at {self.base_url}: {exc.__class__.__name__}"
            ) from exc
        content = THINK_RE.sub("", content)
        return content.strip().strip('"').strip()

    def format_text(self, text: str, profile: str, vocab: list = None) -> str:
        """Apply a formatting profile. Raw or unknown profiles pass through.

        vocab: user's preferred terms/spellings — soft bias so the model
        keeps names like 'WhisperFlow' in their canonical form."""
        system = PROFILES.get(profile)
        if not system or not text.strip():
            return text
        if vocab:
            system += (
                " The user's preferred vocabulary (when the dictation refers "
                "to one of these, use this exact spelling): "
                + ", ".join(vocab) + "."
            )
        out = self._chat(system, text)
        return out if out else text

    def run_command(self, command: str, text: str) -> str:
        system = AI_COMMANDS.get(command)
        if not system:
            raise ValueError(f"Unknown AI command: {command}")
        return self._chat(system, text)

    def ping(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/version", timeout=3).raise_for_status()
            return True
        except requests.RequestException:
            return False
