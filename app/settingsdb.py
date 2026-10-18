"""Отдельная БД настроек приложения.

Все настраиваемые значения (включая системные и служебные ключи кабинета)
хранятся в собственном файле SQLite, независимом от app.db. При первом запуске
значения разово переносятся из таблицы settings старой базы app.db.
"""

import os
import sqlite3
import threading
from datetime import datetime

import config

_LOCK = threading.RLock()
_INIT_DONE = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

MIGRATION_KEY = "settings_migrated_from_appdb"


def _path() -> str:
    return getattr(config, "SETTINGS_DB", "/data/settings/settings.db")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(_path())
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(_path(), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_schema() -> None:
    global _INIT_DONE
    with _LOCK:
        if _INIT_DONE:
            return
        with connect() as conn:
            conn.executescript(SCHEMA)
        _migrate()
        _INIT_DONE = True


def _migrated() -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (MIGRATION_KEY,)
        ).fetchone()
    return row is not None


def _read_legacy() -> list[tuple[str, str, str]]:
    """Значения из таблицы settings старой базы app.db."""
    path = getattr(config, "APP_DB", "")
    if not path or not os.path.exists(path):
        return []
    conn = None
    try:
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT key, value, updated_at FROM settings").fetchall()
        return [
            (row["key"], row["value"], row["updated_at"] or now()) for row in rows
        ]
    except sqlite3.Error:
        return []
    finally:
        if conn is not None:
            conn.close()


def _migrate() -> None:
    if _migrated():
        return
    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM settings").fetchone()["c"]
    if not count:
        rows = _read_legacy()
        if rows:
            with _LOCK, connect() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO settings (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    rows,
                )
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (MIGRATION_KEY, now()),
        )


def get(key: str, default: str = "") -> str:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set(key: str, value: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (key, value, now()),
        )


def delete(key: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM settings WHERE key = ?", (key,))


def all() -> dict[str, str]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: row["value"] for row in rows}


def info() -> dict:
    return {"path": _path(), "migrated": _migrated(), "total": len(all())}
