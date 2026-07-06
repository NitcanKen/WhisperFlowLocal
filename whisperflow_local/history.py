"""SQLite-backed transcription history."""
import sqlite3
import threading
import time

from . import paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    raw TEXT NOT NULL,
    formatted TEXT NOT NULL,
    app TEXT,
    profile TEXT
);
"""


class History:
    def __init__(self, db_path: str = paths.HISTORY_DB):
        paths.ensure_dirs()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def add(self, raw: str, formatted: str, app: str, profile: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO transcriptions (ts, raw, formatted, app, profile) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), raw, formatted, app, profile),
            )
            self._conn.commit()
            return cur.lastrowid

    def recent(self, limit: int = 10) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, raw, formatted, app, profile FROM transcriptions "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0], "ts": r[1], "raw": r[2],
                "formatted": r[3], "app": r[4], "profile": r[5],
            }
            for r in rows
        ]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM transcriptions"
            ).fetchone()[0]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM transcriptions")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
