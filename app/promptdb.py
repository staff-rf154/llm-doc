import sqlite3
import threading
from datetime import datetime

import config

_LOCK = threading.Lock()
_INIT_DONE = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    product TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompts_active ON prompts(active);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.PROMPTS_DB, timeout=30, check_same_thread=False)
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


def count_prompts() -> int:
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM prompts").fetchone()
    return int(row["n"])


def list_prompts(active_only: bool = True) -> list[dict]:
    ensure_schema()
    query = "SELECT text, product, keywords, source_url FROM prompts"
    if active_only:
        query += " WHERE active = 1"
    query += " ORDER BY id"
    with connect() as conn:
        rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def upsert_prompts(items: list[dict], replace: bool = False) -> int:
    ensure_schema()
    with _LOCK, connect() as conn:
        if replace:
            conn.execute("DELETE FROM prompts")
        now = datetime.now().isoformat(timespec="seconds")
        for item in items:
            conn.execute(
                """
                INSERT INTO prompts (text, product, keywords, source_url, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(text) DO UPDATE SET
                    product = excluded.product,
                    keywords = excluded.keywords,
                    source_url = excluded.source_url,
                    active = 1
                """,
                (
                    item["text"],
                    item.get("product", ""),
                    " ".join(item.get("keywords", [])),
                    item.get("source_url", ""),
                    now,
                ),
            )
    return count_prompts()


def set_active(texts: list[str], active: bool) -> int:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.executemany(
            "UPDATE prompts SET active = ? WHERE text = ?",
            [(1 if active else 0, text) for text in texts],
        )
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0