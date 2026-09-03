"""Vocabulary (hotwords): input parsing, bias-list building, LLM prompt."""
from whisperflow_local.llm import LLMClient
from whisperflow_local.textproc import parse_vocab_entry, vocab_terms


def test_parse_vocab_entry_pair_both_arrows():
    assert parse_vocab_entry("維斯帕 → WhisperFlow") == ("pair", "維斯帕", "WhisperFlow")
    assert parse_vocab_entry("wrong -> right") == ("pair", "wrong", "right")


def test_parse_vocab_entry_bare_term():
    assert parse_vocab_entry("  SenseVoice  ") == ("term", "SenseVoice")


def test_parse_vocab_entry_invalid():
    assert parse_vocab_entry("") is None
    assert parse_vocab_entry("   ") is None
    assert parse_vocab_entry("a → ") is None
    assert parse_vocab_entry("→ b") is None


def test_vocab_terms_merges_and_dedupes():
    dictionary = [
        {"from": "whisper flow", "to": "WhisperFlow"},
        {"from": "維斯帕", "to": "WhisperFlow"},  # duplicate target
        {"from": "x", "to": ""},  # empty target dropped
    ]
    hotwords = ["Ollama", "WhisperFlow"]
    assert vocab_terms(dictionary, hotwords) == ["Ollama", "WhisperFlow"]


def test_vocab_terms_empty_inputs():
    assert vocab_terms(None, None) == []
    assert vocab_terms([], []) == []


def test_format_text_injects_vocab_into_system_prompt():
    client = LLMClient("http://127.0.0.1:1", "test-model")
    captured = {}

    def fake_chat(system, user):
        captured["system"] = system
        captured["user"] = user
        return "ok"

    client._chat = fake_chat
    client.format_text("hello", "Structured", vocab=["WhisperFlow", "Ollama"])
    assert "WhisperFlow, Ollama" in captured["system"]
    assert captured["user"] == "hello"


def test_format_text_without_vocab_keeps_prompt_clean():
    client = LLMClient("http://127.0.0.1:1", "test-model")
    captured = {}
    client._chat = lambda s, u: captured.setdefault("system", s) and "ok" or "ok"
    client.format_text("hello", "Structured")
    assert "preferred vocabulary" not in captured["system"]
