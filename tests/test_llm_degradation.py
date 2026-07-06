"""Graceful-degradation tests (H2): a REAL connection attempt to a dead local
port must raise LLMUnavailable, and callers fall back to the raw transcript.
This exercises the actual network error path against a real closed socket."""
import pytest

from whisperflow_local.llm import THINK_RE, LLMClient, LLMUnavailable


def test_ollama_down_raises_lmm_unavailable():
    client = LLMClient("http://127.0.0.1:1", "qwen3.5:4b", timeout=2)
    with pytest.raises(LLMUnavailable):
        client.format_text("hello world", "Clean")


def test_ping_false_when_down():
    client = LLMClient("http://127.0.0.1:1", "qwen3.5:4b", timeout=2)
    assert client.ping() is False


def test_raw_profile_bypasses_llm_entirely():
    # Raw must never touch the network — works even with Ollama down.
    client = LLMClient("http://127.0.0.1:1", "qwen3.5:4b", timeout=2)
    assert client.format_text("keep as is", "Raw") == "keep as is"


def test_think_block_stripping():
    s = "<think>internal chain of thought</think>Final answer."
    assert THINK_RE.sub("", s) == "Final answer."
