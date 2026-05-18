"""Thread-safe SQLite cache for baseline and projection values."""
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any

from app.config import CACHE_DB, CACHE_TTL_DAYS

_lock = threading.Lock()
_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        _local.conn.commit()
    return _local.conn


def get(key: str) -> Any | None:
    with _lock:
        row = _conn().execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    value, expires_at = row
    if datetime.fromisoformat(expires_at) < datetime.utcnow():
        return None
    return json.loads(value)


def set(key: str, value: Any, ttl_days: int = CACHE_TTL_DAYS) -> None:
    expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
    with _lock:
        _conn().execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), expires_at),
        )
        _conn().commit()
