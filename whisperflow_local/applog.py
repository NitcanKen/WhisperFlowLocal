"""Append-only pipeline log so every dictation stage is diagnosable:
~/Library/Application Support/WhisperFlow-Local/app.log"""
import datetime
import os

from . import paths

LOG_PATH = os.path.join(paths.APP_SUPPORT, "app.log")


def log(stage: str, detail: str) -> None:
    paths.ensure_dirs()
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"{ts} [{stage}] {detail}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)
