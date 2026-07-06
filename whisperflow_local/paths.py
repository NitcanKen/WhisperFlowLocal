"""Filesystem locations for config, history, and cached models."""
import os

APP_SUPPORT = os.path.expanduser("~/Library/Application Support/WhisperFlow-Local")
CONFIG_PATH = os.path.join(APP_SUPPORT, "config.json")
HISTORY_DB = os.path.join(APP_SUPPORT, "history.sqlite3")
AUDIO_TMP = os.path.join(APP_SUPPORT, "last_recording.wav")
LAUNCH_AGENT = os.path.expanduser(
    "~/Library/LaunchAgents/com.whisperflow.local.plist"
)


def ensure_dirs() -> None:
    os.makedirs(APP_SUPPORT, exist_ok=True)
