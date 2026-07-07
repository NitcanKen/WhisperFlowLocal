"""VLLMBackend: SSE accumulation, think-strip, and real-socket fallback.

The streaming parse is unit-tested with an in-test fake response; the failure
path uses a REAL closed port (mirrors test_llm_degradation) so no product code
is mocked.
"""
import json

import pytest

from whisperflow_local.llm import THINK_RE, LLMUnavailable, VLLMBackend


class FakeResp:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        for ln in self._lines:
            yield ln

    def close(self):
        self.closed = True


def _sse(pieces):
    lines = ['data: {"choices":[{"delta":{"role":"assistant"}}]}']
    for p in pieces:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}))
    lines.append("data: [DONE]")
    return lines


def test_read_stream_accumulates_deltas():
    b = VLLMBackend("http://x/v1", "m")
    resp = FakeResp(_sse(["Hello", " ", "world"]))
    assert b._read_stream(resp) == "Hello world"
    assert resp.closed  # connection released even on the happy path


def test_read_stream_ignores_malformed_lines():
    b = VLLMBackend("http://x/v1", "m")
    lines = ["", ": keep-alive comment", "data: not-json",
             'data: {"choices":[{"delta":{"content":"ok"}}]}',
             "data: [DONE]"]
    assert b._read_stream(FakeResp(lines)) == "ok"


def test_think_block_stripped_from_stream():
    b = VLLMBackend("http://x/v1", "m")
    raw = b._read_stream(FakeResp(_sse(["<think>", "reasoning", "</think>", "Answer"])))
    assert THINK_RE.sub("", raw).strip() == "Answer"


def test_connect_failure_raises_unavailable():
    b = VLLMBackend("http://127.0.0.1:1/v1", "m",
                    connect_timeout=0.5, ttft_timeout=0.5)
    with pytest.raises(LLMUnavailable):
        b.format_text("hello world", "Clean")


def test_ping_false_when_down():
    b = VLLMBackend("http://127.0.0.1:1/v1", "m", connect_timeout=0.5)
    assert b.ping() is False


def test_raw_profile_bypasses_network():
    # Unknown/Raw profile never touches the network — safe even with no server.
    b = VLLMBackend("http://127.0.0.1:1/v1", "m", connect_timeout=0.5)
    assert b.format_text("keep as is", "Raw") == "keep as is"
