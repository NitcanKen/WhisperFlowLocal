"""Config persistence tests (A3/D3) against real temp files."""
import stat

from whisperflow_local.config import DEFAULTS, Config


def test_defaults_and_roundtrip(tmp_path):
    path = str(tmp_path / "config.json")
    c = Config(path)
    assert c.get("ptt_key") == DEFAULTS["ptt_key"]
    c.set("ptt_key", "f19")
    c.set("dictionary", [{"from": "abc", "to": "ABC Corp"}])
    assert stat.S_IMODE((tmp_path / "config.json").stat().st_mode) == 0o600
    # A fresh instance reads persisted values back (survives restart).
    c2 = Config(path)
    assert c2.get("ptt_key") == "f19"
    assert c2.get("dictionary") == [{"from": "abc", "to": "ABC Corp"}]
    # Unset keys still resolve to defaults.
    assert c2.get("profile") == DEFAULTS["profile"]


def test_corrupt_config_falls_back_to_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not valid json", encoding="utf-8")
    c = Config(str(path))
    assert c.get("language") == DEFAULTS["language"]


def test_existing_config_permissions_are_hardened(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"language": "en"}', encoding="utf-8")
    path.chmod(0o644)
    Config(str(path))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_public_defaults_are_local_and_credential_free():
    assert DEFAULTS["asr_engine"] == "sensevoice"
    assert DEFAULTS["llm_backend"] == "local"
    assert DEFAULTS["qwen_asr_url"].startswith("http://127.0.0.1:")
    assert DEFAULTS["vllm_url"].startswith("http://127.0.0.1:")
    assert DEFAULTS["qwen_asr_api_key"] == ""
    assert DEFAULTS["vllm_api_key"] == ""
