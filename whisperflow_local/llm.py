"""Local LLM layer: qwen3.5 via the localhost Ollama API.

Qwen3-family models are hybrid reasoning models. Thinking is disabled with
the API-level `think: false` flag (measured 60-180 s/utterance with thinking
on — unusable for dictation); if the server/model rejects that flag the
request is retried without it and any <think>...</think> block is stripped
from the output defensively.

The Clean profile does NOT rewrite the transcript with the LLM. Verified
against real qwen3.5:4b output: full-sentence rewriting corrupts Cantonese
(converts to written Chinese, reorders, swaps random words). Instead the
model proposes an edit list ({"from","to"} pairs) and textproc.apply_edits
applies only the edits that pass deterministic safety guards, so a
hallucinated edit degrades to a no-op instead of corrupted text.
"""
import json
import re

import requests

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_COMMON_RULES = (
    "You are a dictation post-processor. The user dictated text that may mix "
    "Cantonese and English in the same sentence (code-switching). "
    "The text comes from automatic speech recognition, so it may contain "
    "recognition errors: a word that sounds like what was said but is wrong "
    "in context — Cantonese homophones or near-homophones, or an English "
    "term transcribed as similar-sounding Chinese characters or as a "
    "different English word. When the sentence context makes the intended "
    "word clear, silently correct it to what the speaker actually said; if "
    "unsure, keep the original word. "
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


# Edit-list prompt for the Clean profile. Written in Cantonese: qwen3.5:4b
# follows Cantonese instructions for Cantonese text far better than English
# ones (verified empirically against the live model).
EDIT_SYSTEM = """你係廣東話聽寫校正器，只負責搵出 ASR（語音識別）寫錯咗嘅字。輸入可能廣東話中英夾雜。

只搵兩類問題：
1. 同音字／近音字錯誤：ASR 將講者講嘅字寫成粵語同音或近音嘅另一個字，令上下文明顯唔通順。
2. 填充詞：呃、嗯、um、uh（淨係呢啲）。

輸出 JSON：{"edits": [{"from": "錯嘅片段", "to": "正確片段"}]}
規則：
- 「from」必須一字不差咁出現喺輸入入面，而且至少兩個字（要帶埋上下文，例如「中影夾雜」而唔係「影」）。
- 「to」同「from」讀音要相近（粵拼同音或近音），唔准換做讀音完全唔同嘅字。
- 大部分句子係啱嘅：冇明顯錯就輸出 {"edits": []}。你唔係好有把握講者原本想講咩字，就唔好改。寧願唔改，都唔好改錯。
- 唔准改異體字寫法（例如個→箇）、唔准翻譯、唔准轉書面語、唔准改語序。
- 廣東話口語字（嘅、咁、係、俾、唔、而家、依家）係正常，唔係錯。

例子：
輸入：我想食意大利份，同埋一杯凍檸茶。
輸出：{"edits": [{"from": "食意大利份", "to": "食意大利粉"}]}
輸入：幫我 book 三點嘅會議室。
輸出：{"edits": []}
輸入：呃佢話個 server 好似 down 咗。
輸出：{"edits": [{"from": "呃", "to": ""}]}"""


def parse_edits(raw: str) -> list:
    """Parse the model's edit-list JSON. Malformed output → no edits,
    never an exception: a broken response must degrade to a no-op."""
    try:
        data = json.loads(raw)
        edits = data.get("edits")
        if not isinstance(edits, list):
            return []
        return [
            {"from": str(e["from"]), "to": str(e["to"])}
            for e in edits
            if isinstance(e, dict) and "from" in e and "to" in e
        ]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


class LLMUnavailable(Exception):
    """Raised when Ollama cannot be reached or the model is missing."""


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, system: str, user: str, force_json: bool = False) -> str:
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
        if force_json:
            payload["format"] = "json"
            payload["options"] = {"temperature": 0}
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

    def propose_edits(self, text: str, vocab: list = None) -> list:
        """Ask the model for ASR-error corrections as an edit list.

        Returns [{"from","to"}, ...]; the caller applies them through
        textproc.apply_edits so unsafe edits are dropped. Raises
        LLMUnavailable when Ollama is down (same contract as format_text)."""
        if not text.strip():
            return []
        system = EDIT_SYSTEM
        if vocab:
            system += (
                "\n\n用戶常用詞（如果原文有片段係呢啲詞嘅近音誤寫，"
                "改返做呢個寫法）：" + "、".join(vocab)
            )
        return parse_edits(self._chat(system, text, force_json=True))

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
