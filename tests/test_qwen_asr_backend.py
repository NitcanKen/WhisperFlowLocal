"""RemoteQwenASRBackend: form-field building, response parsing, and real-socket
failure.

The parse is unit-tested with an in-test fake response object; the failure path
uses a REAL closed port (mirrors test_llm_degradation / test_vllm_backend) so no
product code is mocked.
"""
import pytest

from whisperflow_local.asr import ASRUnavailable, RemoteQwenASRBackend


class FakeResp:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_form_data_omits_language_for_auto():
    b = RemoteQwenASRBackend("http://x/v1", "m")
    data = b._form_data("auto", None)
    assert data["model"] == "m"
    assert data["response_format"] == "json"
    assert "language" not in data   # auto-detect
    assert "prompt" not in data


def test_form_data_maps_language_and_hotword_prompt():
    b = RemoteQwenASRBackend("http://x/v1", "m")
    data = b._form_data("yue", ["WhisperFlow", "中英夾雜"])
    assert data["language"] == "Cantonese"          # QWEN_LANGUAGE_MAP
    assert data["prompt"] == "WhisperFlow, 中英夾雜"  # native decode-time biasing


def test_form_data_english_and_blank_context():
    b = RemoteQwenASRBackend("http://x/v1", "m")
    assert b._form_data("en", ["", "  "])["language"] == "English"
    assert "prompt" not in b._form_data("en", ["", "  "])  # nothing real to bias


def test_parse_response_extracts_text():
    b = RemoteQwenASRBackend("http://x/v1", "m")
    assert b._parse_response(FakeResp({"text": "  hello world  "})) == "hello world"


def test_parse_response_empty_text_ok():
    # A silent utterance legitimately returns empty text — not an error.
    b = RemoteQwenASRBackend("http://x/v1", "m")
    assert b._parse_response(FakeResp({"text": ""})) == ""


def test_parse_response_bad_status_raises():
    b = RemoteQwenASRBackend("http://x/v1", "m")
    with pytest.raises(ASRUnavailable):
        b._parse_response(FakeResp({"error": "boom"}, status_code=400))


def test_parse_response_missing_text_raises():
    b = RemoteQwenASRBackend("http://x/v1", "m")
    with pytest.raises(ASRUnavailable):
        b._parse_response(FakeResp({"not_text": "x"}))


def test_parse_response_malformed_json_raises():
    b = RemoteQwenASRBackend("http://x/v1", "m")
    with pytest.raises(ASRUnavailable):
        b._parse_response(FakeResp(ValueError("no json")))


def test_transcribe_connect_failure_raises_unavailable(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....WAVEfake")  # real bytes; the box is what's unreachable
    b = RemoteQwenASRBackend("http://127.0.0.1:1/v1", "m",
                             connect_timeout=0.5, total_timeout=0.5)
    with pytest.raises(ASRUnavailable):
        b.transcribe(str(wav), "auto", ["WhisperFlow"])


def test_ping_false_when_down():
    b = RemoteQwenASRBackend("http://127.0.0.1:1/v1", "m", connect_timeout=0.5)
    assert b.ping() is False
