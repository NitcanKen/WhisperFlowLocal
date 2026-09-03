"""History storage tests against a real temporary SQLite database (F1)."""
from whisperflow_local.history import History


def test_add_and_recent(tmp_path):
    h = History(str(tmp_path / "h.sqlite3"))
    h.add("raw one", "formatted one", "Mail", "Structured")
    h.add("raw two", "formatted two", "Slack", "Verbatim")
    rows = h.recent(10)
    assert len(rows) == 2
    assert rows[0]["formatted"] == "formatted two"  # newest first
    assert rows[0]["app"] == "Slack" and rows[0]["profile"] == "Verbatim"
    assert rows[1]["raw"] == "raw one"
    assert h.count() == 2
    h.close()


def test_recent_limit_and_clear(tmp_path):
    h = History(str(tmp_path / "h.sqlite3"))
    for i in range(15):
        h.add(f"r{i}", f"f{i}", "App", "Verbatim")
    assert len(h.recent(10)) == 10
    h.clear()
    assert h.count() == 0
    assert h.recent(5) == []
    h.close()


def test_cjk_roundtrip(tmp_path):
    h = History(str(tmp_path / "h.sqlite3"))
    h.add("你好呀 hello", "你好呀，hello!", "Structured", "Verbatim")
    row = h.recent(1)[0]
    assert row["raw"] == "你好呀 hello"
    assert row["formatted"] == "你好呀，hello!"
    h.close()
