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


def _sse(pieces, field="content"):
    lines = ['data: {"choices":[{"delta":{"role":"assistant"}}]}']
    for p in pieces:
        lines.append("data: " + json.dumps({"choices": [{"delta": {field: p}}]}))
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


def test_reasoning_channel_salvaged_when_content_empty():
    # DeepSeek-V4-Flash-style deployment: the final answer streams through the
    # reasoning channel and `content` never arrives. The backend must salvage it
    # instead of returning "" (which silently no-ops the whole LLM layer).
    b = VLLMBackend("http://x/v1", "m")
    resp = FakeResp(_sse(['{"edits": [', '{"from":"a","to":"b"}', "]}"],
                          field="reasoning"))
    assert b._read_stream(resp) == '{"edits": [{"from":"a","to":"b"}]}'


def test_content_channel_wins_over_reasoning():
    # When both channels stream, content is the real answer; reasoning is CoT.
    b = VLLMBackend("http://x/v1", "m")
    lines = ['data: {"choices":[{"delta":{"role":"assistant"}}]}',
             'data: {"choices":[{"delta":{"reasoning":"thinking..."}}]}',
             'data: {"choices":[{"delta":{"content":"Answer"}}]}',
             "data: [DONE]"]
    assert b._read_stream(FakeResp(lines)) == "Answer"


def test_think_block_stripped_from_stream():
    b = VLLMBackend("http://x/v1", "m")
    raw = b._read_stream(FakeResp(_sse(["<think>", "reasoning", "</think>", "Answer"])))
    assert THINK_RE.sub("", raw).strip() == "Answer"


def test_bearer_auth_headers_are_optional():
    base_headers = {
        "Accept": "text/event-stream",
        "ngrok-skip-browser-warning": "1",
    }
    assert VLLMBackend("http://x/v1", "m")._auth_headers() == base_headers
    secured = VLLMBackend(
        "http://x/v1", "m", api_key="unit-test-key"  # pragma: allowlist secret
    )
    assert secured._auth_headers() == {
        **base_headers, "Authorization": "Bearer unit-test-key",
    }


def test_chat_requests_final_content_without_reasoning(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResp(_sse(["OK"]))

    monkeypatch.setattr("whisperflow_local.llm.requests.post", fake_post)
    backend = VLLMBackend(
        "https://example.ngrok-free.dev/v1", "DeepSeek-V4-Flash-0731",
        api_key="unit-test-key",  # pragma: allowlist secret
    )

    assert backend._chat("Be concise", "Reply OK") == "OK"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["json"]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
    # Reasoning must stay included: deployments that ignore enable_thinking put
    # the final answer there, and _read_stream salvages it when content is empty.
    assert captured["json"]["include_reasoning"] is True
    # DeepSeek-V4's real reasoning kill switch — defaults to "none".
    assert captured["json"]["reasoning_effort"] == "none"
    assert captured["headers"]["ngrok-skip-browser-warning"] == "1"


def test_reasoning_effort_omitted_when_blank(monkeypatch):
    # Plain non-reasoning vLLM servers may reject an unknown field; a blank
    # setting must drop it from the payload entirely.
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResp(_sse(["OK"]))

    monkeypatch.setattr("whisperflow_local.llm.requests.post", fake_post)
    backend = VLLMBackend("http://x/v1", "m", reasoning_effort="")
    backend._chat("s", "u")
    assert "reasoning_effort" not in captured["json"]


def test_connect_failure_raises_unavailable():
    b = VLLMBackend("http://127.0.0.1:1/v1", "m",
                    connect_timeout=0.5, ttft_timeout=0.5)
    with pytest.raises(LLMUnavailable):
        b.format_text("hello world", "Structured")


def test_ping_false_when_down():
    b = VLLMBackend("http://127.0.0.1:1/v1", "m", connect_timeout=0.5)
    assert b.ping() is False


def test_profile_without_a_prompt_bypasses_network():
    # Verbatim has no PROFILES entry (it goes through propose_cleanup), so
    # format_text short-circuits — safe even with no server.
    b = VLLMBackend("http://127.0.0.1:1/v1", "m", connect_timeout=0.5)
    assert b.format_text("keep as is", "Verbatim") == "keep as is"
