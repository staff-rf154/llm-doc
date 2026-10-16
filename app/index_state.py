import sqlite3
import threading
from datetime import datetime

import config

_LOCK = threading.Lock()
_INIT_DONE = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    collection TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS file_progress (
    collection TEXT NOT NULL,
    source_key TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    chunks INTEGER NOT NULL,
    done INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (collection, source_key)
);
"""

OPEN_STATUSES = ("running", "paused", "interrupted")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.INDEX_STATE_DB, timeout=30, check_same_thread=False)
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
        _INIT_DONE = True


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def start_run(collection: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (collection, status, started_at, finished_at, error)
            VALUES (?, 'running', ?, NULL, NULL)
            ON CONFLICT(collection) DO UPDATE SET
                status = 'running',
                finished_at = NULL,
                error = NULL
            """,
            (collection, _now()),
        )


def mark_run(collection: str, status: str, error: str | None = None) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE collection = ?",
            (status, error, _now() if status in ("done", "error") else None, collection),
        )


def get_run(collection: str) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT collection, status, started_at, finished_at, error FROM runs WHERE collection = ?",
            (collection,),
        ).fetchone()
    return dict(row) if row else None


def get_open_run() -> dict | None:
    """Последний незавершённый run: подходит для продолжения после падения."""
    ensure_schema()
    placeholders = ",".join("?" * len(OPEN_STATUSES))
    with connect() as conn:
        row = conn.execute(
            f"""
            SELECT collection, status, started_at, finished_at, error
            FROM runs WHERE status IN ({placeholders})
            ORDER BY started_at DESC LIMIT 1
            """,
            OPEN_STATUSES,
        ).fetchone()
    return dict(row) if row else None


def forget_run(collection: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM file_progress WHERE collection = ?", (collection,))
        conn.execute("DELETE FROM runs WHERE collection = ?", (collection,))


def forget_source(collection: str, source_key: str) -> None:
    """Забыть прогресс по конкретному файлу, чтобы он индексировался заново."""
    with _LOCK, connect() as conn:
        conn.execute(
            "DELETE FROM file_progress WHERE collection = ? AND source_key = ?",
            (collection, source_key),
        )


def get_file_progress(collection: str, source_key: str) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT size, mtime, chunks, done, updated_at
            FROM file_progress WHERE collection = ? AND source_key = ?
            """,
            (collection, source_key),
        ).fetchone()
    return dict(row) if row else None


def save_file_progress(
    collection: str,
    source_key: str,
    size: int,
    mtime: float,
    chunks: int,
    done: int,
) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            """
            INSERT INTO file_progress
                (collection, source_key, size, mtime, chunks, done, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collection, source_key) DO UPDATE SET
                size = excluded.size,
                mtime = excluded.mtime,
                chunks = excluded.chunks,
                done = excluded.done,
                updated_at = excluded.updated_at
            """,
            (collection, source_key, size, mtime, chunks, done, _now()),
        )


def run_stats(collection: str) -> dict:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS files, COALESCE(SUM(done), 0) AS done,
                   COALESCE(SUM(chunks), 0) AS chunks
            FROM file_progress WHERE collection = ?
            """,
            (collection,),
        ).fetchone()
    return {
        "files": int(row["files"]),
        "done": int(row["done"]),
        "chunks": int(row["chunks"]),
    }