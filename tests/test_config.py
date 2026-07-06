"""Config persistence tests (A3/D3) against real temp files."""
from whisperflow_local.config import DEFAULTS, Config


def test_defaults_and_roundtrip(tmp_path):
    path = str(tmp_path / "config.json")
    c = Config(path)
    assert c.get("ptt_key") == DEFAULTS["ptt_key"]
    c.set("ptt_key", "f19")
    c.set("dictionary", [{"from": "abc", "to": "ABC Corp"}])
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
