"""Пользователи, сессии, журнал запросов, оценки и исправления ответов."""

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta

import config

_LOCK = threading.RLock()
_INIT_DONE = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    username TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    grounded INTEGER NOT NULL DEFAULT 0,
    corrected INTEGER NOT NULL DEFAULT 0,
    sources INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_user ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
CREATE TABLE IF NOT EXISTS ratings (
    request_id INTEGER PRIMARY KEY,
    vote TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    author TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_status ON corrections(status);
"""

SESSION_DAYS = 14


def connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(config.APP_DB)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(config.APP_DB, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_schema() -> None:
    global _INIT_DONE
    with _LOCK:
        if _INIT_DONE:
            return
        with connect() as conn:
            conn.executescript(SCHEMA)
        _INIT_DONE = True


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 200_000
    ).hex()
    return salt, digest


def verify_password(password: str, salt: str, expected: str) -> bool:
    _, digest = hash_password(password, salt)
    return hmac.compare_digest(digest, expected)


def create_user(username: str, password: str, role: str = "user") -> dict:
    ensure_schema()
    salt, digest = hash_password(password)
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (username, salt, password_hash, role, active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (username.strip(), salt, digest, role, now()),
        )
        user_id = cur.lastrowid
    return get_user(user_id) or {}


def get_user(user_id: int) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, username, role, active, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_name(username: str) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, username, salt, password_hash, role, active, created_at
            FROM users WHERE username = ?
            """,
            (username.strip(),),
        ).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.username, u.role, u.active, u.created_at,
                   (SELECT COUNT(*) FROM requests r WHERE r.user_id = u.id) AS requests,
                   (SELECT COUNT(*) FROM ratings t JOIN requests r2 ON r2.id = t.request_id
                     WHERE r2.user_id = u.id AND t.vote = 'up') AS likes,
                   (SELECT COUNT(*) FROM ratings t JOIN requests r2 ON r2.id = t.request_id
                     WHERE r2.user_id = u.id AND t.vote = 'down') AS dislikes,
                   (SELECT MAX(r3.created_at) FROM requests r3 WHERE r3.user_id = u.id) AS last_request
            FROM users u ORDER BY u.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def set_password(user_id: int, password: str) -> None:
    ensure_schema()
    salt, digest = hash_password(password)
    with _LOCK, connect() as conn:
        conn.execute(
            "UPDATE users SET salt = ?, password_hash = ? WHERE id = ?",
            (salt, digest, user_id),
        )


def set_active(user_id: int, active: bool) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            "UPDATE users SET active = ? WHERE id = ?",
            (1 if active else 0, user_id),
        )
        if not active:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


def set_role(user_id: int, role: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))


def delete_user(user_id: int) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def login(username: str, password: str) -> str | None:
    user = get_user_by_name(username)
    if not user or not user["active"]:
        return None
    if not verify_password(password, user["salt"], user["password_hash"]):
        return None
    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(days=SESSION_DAYS)).isoformat(timespec="seconds")
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user["id"], now(), expires),
        )
    return token


def logout(token: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def user_by_token(token: str | None) -> dict | None:
    if not token:
        return None
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.username, u.role, u.active
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token = ? AND s.expires_at > ?
            """,
            (token, now()),
        ).fetchone()
    return dict(row) if row and row["active"] else None


def ensure_admin() -> str:
    """Создать первого администратора из переменных окружения, если его ещё нет."""
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
    if int(row["n"]):
        return ""
    username = config.ADMIN_USERNAME
    password = config.ADMIN_PASSWORD
    if not username or not password:
        return ""
    create_user(username, password, role="admin")
    return f"создан администратор {username}"


def log_request(
    user: dict | None,
    question: str,
    answer: str,
    grounded: bool,
    corrected: bool,
    sources: int,
    latency_ms: int,
) -> int:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO requests
                (user_id, username, question, answer, grounded, corrected, sources, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (user or {}).get("id"),
                (user or {}).get("username", ""),
                question,
                answer,
                1 if grounded else 0,
                1 if corrected else 0,
                sources,
                latency_ms,
                now(),
            ),
        )
        return int(cur.lastrowid)


def list_requests(
    limit: int = 100,
    offset: int = 0,
    username: str | None = None,
    vote: str | None = None,
    search: str | None = None,
) -> list[dict]:
    ensure_schema()
    query = """
        SELECT r.id, r.username, r.user_id, r.question, r.answer, r.grounded,
               r.corrected, r.sources, r.latency_ms, r.created_at,
               t.vote, t.comment AS rating_comment, t.updated_at AS rated_at
        FROM requests r LEFT JOIN ratings t ON t.request_id = r.id
        WHERE 1 = 1
    """
    params: list = []
    if username:
        query += " AND r.username = ?"
        params.append(username)
    if vote == "none":
        query += " AND t.vote IS NULL"
    elif vote in ("up", "down"):
        query += " AND t.vote = ?"
        params.append(vote)
    if search:
        query += " AND (r.question LIKE ? OR r.answer LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    query += " ORDER BY r.id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def count_requests(vote: str | None = None) -> int:
    ensure_schema()
    query = """
        SELECT COUNT(*) AS n FROM requests r
        LEFT JOIN ratings t ON t.request_id = r.id WHERE 1 = 1
    """
    params: list = []
    if vote == "none":
        query += " AND t.vote IS NULL"
    elif vote in ("up", "down"):
        query += " AND t.vote = ?"
        params.append(vote)
    with connect() as conn:
        row = conn.execute(query, params).fetchone()
    return int(row["n"])


def set_rating(request_id: int, vote: str, comment: str = "") -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            """
            INSERT INTO ratings (request_id, vote, comment, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                vote = excluded.vote,
                comment = excluded.comment,
                updated_at = excluded.updated_at
            """,
            (request_id, vote, comment, now(), now()),
        )


def create_correction(question: str, answer: str, author: str) -> dict:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO corrections (question, answer, status, author, created_at, updated_at)
            VALUES (?, ?, 'active', ?, ?, ?)
            """,
            (question, answer, author, now(), now()),
        )
        correction_id = int(cur.lastrowid)
    return get_correction(correction_id) or {}


def get_correction(correction_id: int) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM corrections WHERE id = ?", (correction_id,)
        ).fetchone()
    return dict(row) if row else None


def list_corrections(status: str | None = None) -> list[dict]:
    ensure_schema()
    query = "SELECT * FROM corrections"
    params: list = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY id DESC"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def update_correction(
    correction_id: int,
    question: str | None = None,
    answer: str | None = None,
    status: str | None = None,
) -> None:
    ensure_schema()
    fields = []
    params: list = []
    if question is not None:
        fields.append("question = ?")
        params.append(question)
    if answer is not None:
        fields.append("answer = ?")
        params.append(answer)
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if not fields:
        return
    fields.append("updated_at = ?")
    params.append(now())
    params.append(correction_id)
    with _LOCK, connect() as conn:
        conn.execute(f"UPDATE corrections SET {', '.join(fields)} WHERE id = ?", params)


def delete_correction(correction_id: int) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM corrections WHERE id = ?", (correction_id,))


def stats_by_user() -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT u.id AS user_id, u.username, u.role, u.active, u.created_at,
                   COUNT(r.id) AS requests,
                   COALESCE(SUM(t.vote = 'up'), 0) AS likes,
                   COALESCE(SUM(t.vote = 'down'), 0) AS dislikes,
                   MAX(r.created_at) AS last_request
            FROM users u
            LEFT JOIN requests r ON r.user_id = u.id
            LEFT JOIN ratings t ON t.request_id = r.id
            GROUP BY u.id ORDER BY requests DESC, u.username
            """
        ).fetchall()
    return [dict(row) for row in rows]


def stats_by_day(days: int = 30) -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   COUNT(*) AS requests,
                   COUNT(DISTINCT username) AS users
            FROM requests
            WHERE created_at >= datetime('now', ?)
            GROUP BY day ORDER BY day
            """,
            (f"-{int(days)} days",),
        ).fetchall()
    return [dict(row) for row in rows]


def top_questions(limit: int = 20) -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT question, COUNT(*) AS n,
                   SUM(CASE WHEN vote = 'up' THEN 1 ELSE 0 END) AS likes,
                   SUM(CASE WHEN vote = 'down' THEN 1 ELSE 0 END) AS dislikes
            FROM requests LEFT JOIN ratings ON ratings.request_id = requests.id
            GROUP BY question ORDER BY n DESC, question LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def summary() -> dict:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM users) AS users,
                (SELECT COUNT(*) FROM users WHERE active = 1) AS active_users,
                (SELECT COUNT(*) FROM requests) AS requests,
                (SELECT COUNT(*) FROM ratings WHERE vote = 'up') AS likes,
                (SELECT COUNT(*) FROM ratings WHERE vote = 'down') AS dislikes,
                (SELECT COUNT(*) FROM corrections) AS corrections,
                (SELECT COUNT(*) FROM corrections WHERE status = 'active') AS active_corrections
            """
        ).fetchone()
    return dict(row)
