"""Отдельная схема настроек приложения.

Все настраиваемые значения (включая системные и служебные ключи кабинета)
хранятся в собственной схеме PostgreSQL, независимой от app. При первом
запуске значения разово переносятся из таблицы settings схемы app.
"""

import threading
from datetime import datetime

import db

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


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect():
    return db.connect("settings")


def ensure_schema() -> None:
    global _INIT_DONE
    with _LOCK:
        if _INIT_DONE:
            return
        with connect() as conn:
            db.exec_script(conn, SCHEMA)
        _migrate()
        _INIT_DONE = True


def _migrated() -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = %s", (MIGRATION_KEY,)
        ).fetchone()
    return row is not None


def _read_legacy() -> list[tuple[str, str, str]]:
    """Значения из таблицы settings схемы app."""
    try:
        with db.connect("app") as conn:
            rows = conn.execute(
                "SELECT key, value, updated_at FROM settings"
            ).fetchall()
            return [
                (row["key"], row["value"], row["updated_at"] or now())
                for row in rows
            ]
    except Exception:
        return []


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
                    "INSERT INTO settings (key, value, updated_at) "
                    "VALUES (%s, %s, %s) ON CONFLICT (key) DO NOTHING",
                    rows,
                )
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (MIGRATION_KEY, now()),
        )


def get(key: str, default: str = "") -> str:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = %s", (key,)
        ).fetchone()
    return row["value"] if row else default


def set(key: str, value: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (%s, %s, %s) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (key, value, now()),
        )


def delete(key: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM settings WHERE key = %s", (key,))


def all() -> dict[str, str]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: row["value"] for row in rows}


def info() -> dict:
    return {"schema": "settings", "migrated": _migrated(), "total": len(all())}
