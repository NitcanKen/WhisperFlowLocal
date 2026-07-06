"""ASREngine facade: engine routing, live switching, and fallback."""
import pytest

from whisperflow_local.asr import ASREngine


class FakeEngine:
    def __init__(self, name, text="", fail=False):
        self.name = name
        self.text = text
        self.fail = fail
        self.loads = 0
        self.calls = []

    def ensure_loaded(self, progress_cb=None):
        if self.fail:
            raise RuntimeError(f"{self.name} cannot load")
        self.loads += 1
        self.loaded = True

    def unload(self):
        self.loaded = False

    def transcribe(self, wav_path, language="auto", context=None):
        self.calls.append((wav_path, language, context))
        return self.text


def make_facade(qwen_fail=False):
    facade = ASREngine("qwen3")
    facade._engines = {
        "sensevoice": FakeEngine("sensevoice", "sv text"),
        "qwen3": FakeEngine("qwen3", "qw text", fail=qwen_fail),
    }
    return facade


def test_routes_to_selected_engine_with_context():
    facade = make_facade()
    out = facade.transcribe("a.wav", "yue", context=["WhisperFlow"])
    assert out == "qw text"
    assert facade._engines["qwen3"].calls == [("a.wav", "yue", ["WhisperFlow"])]
    assert facade._engines["sensevoice"].calls == []


def test_falls_back_to_sensevoice_when_selected_engine_fails():
    facade = make_facade(qwen_fail=True)
    out = facade.transcribe("a.wav", "auto", context=None)
    assert out == "sv text"
    assert facade.engine_name == "sensevoice"  # sticky after fallback


def test_sensevoice_failure_is_fatal():
    facade = ASREngine("sensevoice")
    facade._engines = {
        "sensevoice": FakeEngine("sensevoice", fail=True),
        "qwen3": FakeEngine("qwen3"),
    }
    with pytest.raises(RuntimeError):
        facade.transcribe("a.wav")


def test_set_engine_switches_live_and_ignores_unknown():
    facade = make_facade()
    facade.set_engine("sensevoice")
    assert facade.transcribe("b.wav") == "sv text"
    facade.set_engine("nonsense")
    assert facade.engine_name == "sensevoice"


def test_set_engine_unloads_the_other_engine():
    facade = make_facade()
    facade.transcribe("a.wav")  # qwen3 now loaded
    assert facade._engines["qwen3"].loaded
    facade.set_engine("sensevoice")
    assert facade._engines["qwen3"].loaded is False  # freed for 16 GB RAM
    facade.transcribe("b.wav")
    facade.set_engine("qwen3")
    assert facade._engines["sensevoice"].loaded is False
