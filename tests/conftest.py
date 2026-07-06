"""Shared test isolation.

Tests must never write to the production app-support log: stub errors like
'qwen3 cannot load' landing in the real app.log read as field failures and
derail debugging of the live app.
"""
import pytest

import whisperflow_local.applog as applog


@pytest.fixture(autouse=True)
def isolate_applog(tmp_path, monkeypatch):
    monkeypatch.setattr(applog, "LOG_PATH", str(tmp_path / "app.log"))
