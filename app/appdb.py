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
    doc_name TEXT NOT NULL DEFAULT '',
    doc_pages TEXT NOT NULL DEFAULT '',
    doc_ref TEXT NOT NULL DEFAULT '',
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
    doc_name TEXT NOT NULL DEFAULT '',
    doc_pages TEXT NOT NULL DEFAULT '',
    doc_ref TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    author TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exclusions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'document',
    document TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    source_name TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'admin',
    author TEXT NOT NULL DEFAULT '',
    comment TEXT NOT NULL DEFAULT '',
    request_id INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exclusions_pattern ON exclusions(pattern);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    question TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    focus TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    uses INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_status ON lessons(status);
CREATE TABLE IF NOT EXISTS training_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL DEFAULT 'correction',
    correction_id INTEGER,
    lesson_id INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    note TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_log_created ON training_log(created_at);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_status ON corrections(status);
"""

EXCLUSION_COLUMNS = ("scope", "document", "question", "page")
CORRECTION_COLUMNS = ("doc_name", "doc_pages", "doc_ref")
CHECK_COLUMNS = ("check_status", "check_comment", "checked_at", "check_collection")
REQUEST_COLUMNS = ("doc_name", "doc_pages", "doc_ref")

CHECK_STATUSES = ("unchecked", "verified", "review")


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in (
        ("exclusions", EXCLUSION_COLUMNS),
        ("corrections", CORRECTION_COLUMNS),
        ("corrections", CHECK_COLUMNS),
        ("requests", REQUEST_COLUMNS),
    ):
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column in columns:
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                )

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
            _migrate(conn)
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
    reference: dict | None = None,
) -> int:
    ensure_schema()
    reference = reference or {}
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO requests
                (user_id, username, question, answer, grounded, corrected,
                 doc_name, doc_pages, doc_ref, sources, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (user or {}).get("id"),
                (user or {}).get("username", ""),
                question,
                answer,
                1 if grounded else 0,
                1 if corrected else 0,
                reference.get("doc_name", ""),
                reference.get("doc_pages", ""),
                reference.get("doc_ref", ""),
                sources,
                latency_ms,
                now(),
            ),
        )
        return int(cur.lastrowid)


SORT_COLUMNS = {
    "date": "r.id",
    "vote": "CASE t.vote WHEN 'down' THEN 0 WHEN 'up' THEN 2 ELSE 1 END",
    "latency": "r.latency_ms",
    "username": "r.username",
}


def list_requests(
    limit: int = 100,
    offset: int = 0,
    username: str | None = None,
    vote: str | None = None,
    search: str | None = None,
    sort: str = "date",
    order: str = "desc",
) -> list[dict]:
    ensure_schema()
    query = """
        SELECT r.id, r.username, r.user_id, r.question, r.answer, r.grounded,
               r.corrected, r.doc_name, r.doc_pages, r.doc_ref,
               r.sources, r.latency_ms, r.created_at,
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
    column = SORT_COLUMNS.get(sort, SORT_COLUMNS["date"])
    direction = "ASC" if str(order).lower() == "asc" else "DESC"
    query += f" ORDER BY {column} {direction}, r.id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def add_exclusion(
    pattern: str,
    origin: str = "admin",
    author: str = "",
    comment: str = "",
    request_id: int | None = None,
    scope: str = "document",
    document: str = "",
    question: str = "",
    page: int | None = None,
) -> dict | None:
    ensure_schema()
    pattern = pattern.strip()
    if scope == "question" and question.strip():
        pattern = question.strip()
    if not pattern:
        return None
    with _LOCK, connect() as conn:
        conn.execute(
            """
            INSERT INTO exclusions
                (pattern, scope, document, question, page, source_name, origin, author,
                 comment, request_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern.lower(),
                scope,
                document.strip().lower(),
                question.strip().lower(),
                page if page is not None else 0,
                find_source_name(pattern) if scope in ("document", "section", "image") else "",
                origin,
                author,
                comment[:500],
                request_id,
                now(),
            ),
        )
    return get_exclusion_by_pattern(pattern)


def get_exclusion_by_pattern(pattern: str) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM exclusions WHERE pattern = ?", (pattern.strip().lower(),)
        ).fetchone()
    return dict(row) if row else None


def list_exclusions() -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.pattern, e.scope, e.document, e.question, e.page, e.source_name,
                   e.origin, e.author, e.comment, e.request_id, e.created_at,
                   (SELECT COUNT(*) FROM requests r
                     WHERE r.doc_name = e.source_name OR r.doc_ref LIKE '%' || e.source_name || '%')
                     AS mentions
            FROM exclusions e ORDER BY e.id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_exclusion(exclusion_id: int) -> bool:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute("DELETE FROM exclusions WHERE id = ?", (exclusion_id,))
    return bool(cur.rowcount)


def find_source_name(pattern: str) -> str:
    """Совпадает ли шаблон с именем документа из индекса; вернём найденное имя."""
    pattern = pattern.strip().lower()
    try:
        import vectorstore as vs

        names = {name.lower() for name in vs.known_sources()}
    except Exception:
        names = set()
    if not names:
        return ""
    if pattern in names:
        return pattern
    for name in names:
        if name.endswith(pattern) or pattern in name:
            return name
    return ""


def create_lesson(
    topic: str,
    question: str,
    answer: str,
    focus: str,
    author: str,
) -> dict:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO lessons (topic, question, answer, focus, author, status, uses,
                                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'active', 0, ?, ?)
            """,
            (topic, question, answer, focus, author, now(), now()),
        )
        lesson_id = int(cur.lastrowid)
    return get_lesson(lesson_id) or {}


def get_lesson(lesson_id: int) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
    return dict(row) if row else None


def list_lessons(status: str | None = None) -> list[dict]:
    ensure_schema()
    query = "SELECT * FROM lessons"
    params: list = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY id DESC"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def update_lesson(lesson_id: int, **fields) -> None:
    ensure_schema()
    allowed = ("topic", "question", "answer", "focus", "status")
    sets = [f"{key} = ?" for key, value in fields.items() if key in allowed and value is not None]
    if not sets:
        return
    params = [value for key, value in fields.items() if key in allowed and value is not None]
    sets.append("updated_at = ?")
    params.extend([now(), lesson_id])
    with _LOCK, connect() as conn:
        conn.execute(f"UPDATE lessons SET {', '.join(sets)} WHERE id = ?", params)


def delete_lesson(lesson_id: int) -> bool:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))
    return bool(cur.rowcount)


def count_uses(lesson_id: int) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("UPDATE lessons SET uses = uses + 1 WHERE id = ?", (lesson_id,))


def add_training_log(
    source: str,
    correction_id: int | None,
    lesson_id: int | None,
    status: str,
    note: str = "",
    author: str = "",
) -> int:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO training_log
                (source, correction_id, lesson_id, status, note, author, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source, correction_id, lesson_id, status, note[:300], author, now()),
        )
        return int(cur.lastrowid)


def update_training_log(log_id: int, status: str, lesson_id: int | None = None, note: str = "") -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            "UPDATE training_log SET status = ?, lesson_id = COALESCE(?, lesson_id), note = ? WHERE id = ?",
            (status, lesson_id, note[:300], log_id),
        )


def list_training_log(limit: int = 30) -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT t.*, c.doc_ref, c.question AS correction_question
            FROM training_log t LEFT JOIN corrections c ON c.id = t.correction_id
            ORDER BY t.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_setting(key: str, default: str = "") -> str:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> str:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, now()),
        )
    return value


def correction_ids_by_question() -> dict[str, int]:
    """Карта «вопрос в нижнем регистре» -> id исправления."""
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, question FROM corrections WHERE status = 'active'"
        ).fetchall()
    return {row["question"].strip().lower(): int(row["id"]) for row in rows if row["question"]}


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


def create_correction(
    question: str,
    answer: str,
    author: str,
    doc_name: str = "",
    doc_pages: str = "",
    doc_ref: str = "",
) -> dict:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO corrections
                (question, answer, doc_name, doc_pages, doc_ref, status, author, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (question, answer, doc_name, doc_pages, doc_ref, author, now(), now()),
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
    doc_name: str | None = None,
    doc_pages: str | None = None,
    doc_ref: str | None = None,
) -> None:
    ensure_schema()
    fields = []
    params: list = []
    for column, value in (
        ("question", question),
        ("answer", answer),
        ("status", status),
        ("doc_name", doc_name),
        ("doc_pages", doc_pages),
        ("doc_ref", doc_ref),
    ):
        if value is not None:
            fields.append(f"{column} = ?")
            params.append(value)
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


def set_check(
    correction_id: int,
    status: str,
    comment: str,
    collection: str = "",
) -> None:
    ensure_schema()
    if status not in CHECK_STATUSES:
        status = "review"
    with _LOCK, connect() as conn:
        conn.execute(
            """
            UPDATE corrections
            SET check_status = ?, check_comment = ?, checked_at = ?, check_collection = ?
            WHERE id = ?
            """,
            (status, comment.strip()[:500], now(), collection, correction_id),
        )


def reset_checks() -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            """
            UPDATE corrections
            SET check_status = '', check_comment = '', checked_at = '', check_collection = ''
            """
        )


def check_summary() -> dict:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN check_status = 'verified' THEN 1 ELSE 0 END) AS verified,
                   SUM(CASE WHEN check_status = 'review' THEN 1 ELSE 0 END) AS review,
                   SUM(CASE WHEN check_status = '' THEN 1 ELSE 0 END) AS unchecked
            FROM corrections WHERE status = 'active'
            """
        ).fetchone()
    return {
        "total": int(row["total"]),
        "verified": int(row["verified"] or 0),
        "review": int(row["review"] or 0),
        "unchecked": int(row["unchecked"] or 0),
    }


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
                (SELECT COUNT(*) FROM lessons WHERE status = 'active') AS lessons,
                (SELECT COUNT(*) FROM corrections) AS corrections,
                (SELECT COUNT(*) FROM corrections WHERE status = 'active') AS active_corrections
            """
        ).fetchone()
    data = dict(row)
    data.update(check_summary())
    return data
