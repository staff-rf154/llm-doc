"""Пользователи, сессии, журнал запросов, оценки и исправления ответов."""

import hashlib
import hmac
from pathlib import Path
import json
import os
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta

import config
import settingsdb

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
    hidden INTEGER NOT NULL DEFAULT 0,
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
    origin TEXT NOT NULL DEFAULT 'web',
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
CREATE TABLE IF NOT EXISTS request_evals (
    request_id INTEGER PRIMARY KEY,
    verdict TEXT NOT NULL DEFAULT 'unknown',
    score REAL NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT '',
    method TEXT NOT NULL DEFAULT 'heuristic',
    suggested_answer TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);
CREATE INDEX IF NOT EXISTS idx_request_evals_verdict ON request_evals(verdict);
CREATE TABLE IF NOT EXISTS answer_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    topic_id INTEGER,
    lesson_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answer_issues_status ON answer_issues(status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_answer_issues_req_kind ON answer_issues(request_id, kind);
CREATE TABLE IF NOT EXISTS question_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    representative TEXT NOT NULL DEFAULT '',
    total INTEGER NOT NULL DEFAULT 0,
    problematic INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS topic_requests (
    topic_id INTEGER NOT NULL,
    request_id INTEGER NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (topic_id, request_id)
);
CREATE TABLE IF NOT EXISTS generated_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    norm TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'docs',
    doc_source TEXT NOT NULL DEFAULT '',
    doc_chunk INTEGER,
    variant INTEGER NOT NULL DEFAULT 0,
    request_id INTEGER,
    verdict TEXT NOT NULL DEFAULT '',
    cluster_id INTEGER,
    duplicate_of INTEGER,
    similarity REAL NOT NULL DEFAULT 0,
    vector BLOB,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_genq_norm ON generated_questions(norm);
CREATE INDEX IF NOT EXISTS idx_genq_cluster ON generated_questions(cluster_id);
CREATE INDEX IF NOT EXISTS idx_genq_source ON generated_questions(source);
CREATE TABLE IF NOT EXISTS domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    capabilities TEXT NOT NULL DEFAULT '[]',
    categories TEXT NOT NULL DEFAULT '[]',
    terminology TEXT NOT NULL DEFAULT '[]',
    sample_questions TEXT NOT NULL DEFAULT '[]',
    temperature REAL,
    allow_inference INTEGER NOT NULL DEFAULT 0,
    settings TEXT NOT NULL DEFAULT '{}',
    synonyms TEXT NOT NULL DEFAULT '[]',
    is_builtin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS doc_exclusions (
    root TEXT NOT NULL,
    source TEXT NOT NULL,
    excluded_at TEXT NOT NULL,
    PRIMARY KEY (root, source)
);
"""

CORRECTION_COLUMNS = ("doc_name", "doc_pages", "doc_ref")
CHECK_COLUMNS = ("check_status", "check_comment", "checked_at", "check_collection")
REQUEST_COLUMNS = ("doc_name", "doc_pages", "doc_ref", "origin")

CHECK_STATUSES = ("unchecked", "verified", "review")


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in (
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
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    if "hidden" not in existing:
        conn.execute(
            "ALTER TABLE users ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0"
        )
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(domains)")}
    if "temperature" not in existing:
        conn.execute("ALTER TABLE domains ADD COLUMN temperature REAL")
    if "allow_inference" not in existing:
        conn.execute(
            "ALTER TABLE domains ADD COLUMN allow_inference INTEGER NOT NULL DEFAULT 0"
        )
    if "settings" not in existing:
        conn.execute(
            "ALTER TABLE domains ADD COLUMN settings TEXT NOT NULL DEFAULT '{}'"
        )
    if "synonyms" not in existing:
        conn.execute(
            "ALTER TABLE domains ADD COLUMN synonyms TEXT NOT NULL DEFAULT '[]'"
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


def create_user(username: str, password: str, role: str = "user", hidden: bool = False) -> dict:
    ensure_schema()
    salt, digest = hash_password(password)
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (username, salt, password_hash, role, active, hidden, created_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (username.strip(), salt, digest, role, 1 if hidden else 0, now()),
        )
        user_id = cur.lastrowid
    return get_user(user_id) or {}


def get_user(user_id: int) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, username, role, active, hidden, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_name(username: str) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, username, salt, password_hash, role, active, hidden, created_at
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
            FROM users u WHERE u.hidden = 0 ORDER BY u.id
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
            SELECT u.id, u.username, u.role, u.active, u.hidden
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


def ensure_system() -> dict:
    """Создать скрытого системного пользователя с ролью `system`.

    Пароль берётся из SYSTEM_PASSWORD, а при пустом значении генерируется
    случайный и сохраняется в файл рядом с базой приложения.
    """
    ensure_schema()
    username = config.SYSTEM_USERNAME or "system"
    password = (config.SYSTEM_PASSWORD or "").strip()
    generated = False
    if not password:
        password = secrets.token_urlsafe(12)
        generated = True
    existing = get_user_by_name(username)
    if existing:
        if not existing.get("hidden") or existing.get("role") != "system":
            with _LOCK, connect() as conn:
                conn.execute(
                    "UPDATE users SET hidden = 1, role = 'system', active = 1 WHERE id = ?",
                    (existing["id"],),
                )
            return {"created": True, "username": username, "password": password, "generated": generated}
        return {}
    create_user(username, password, role="system", hidden=True)
    return {"created": True, "username": username, "password": password, "generated": generated}


def save_system_password(password: str) -> str:
    """Сохранить пароль системного пользователя в файл, вернуть путь."""
    path = Path(config.APP_DB).parent / "system_password.txt"
    path.write_text(password + "\n", encoding="utf-8")
    return str(path)


def log_request(
    user: dict | None,
    question: str,
    answer: str,
    grounded: bool,
    corrected: bool,
    sources: int,
    latency_ms: int,
    reference: dict | None = None,
    origin: str = "web",
) -> int:
    ensure_schema()
    reference = reference or {}
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO requests
                (user_id, username, question, answer, grounded, corrected,
                 doc_name, doc_pages, doc_ref, sources, latency_ms, origin, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                origin,
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
    origin: str | None = None,
    sort: str = "date",
    order: str = "desc",
) -> list[dict]:
    ensure_schema()
    query = """
        SELECT r.id, r.username, r.user_id, r.question, r.answer, r.grounded,
               r.corrected, r.doc_name, r.doc_pages, r.doc_ref,
               r.sources, r.latency_ms, r.origin, r.created_at,
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
    if origin == "web":
        query += " AND (r.origin = '' OR r.origin = 'web')"
    elif origin:
        query += " AND r.origin = ?"
        params.append(origin)
    column = SORT_COLUMNS.get(sort, SORT_COLUMNS["date"])
    direction = "ASC" if str(order).lower() == "asc" else "DESC"
    query += f" ORDER BY {column} {direction}, r.id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def create_lesson(
    topic: str,
    question: str,
    answer: str,
    focus: str,
    author: str,
    status: str = "active",
) -> dict:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO lessons (topic, question, answer, focus, author, status, uses,
                                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (topic, question, answer, focus, author, status, now(), now()),
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


def training_lessons_for_correction(correction_id: int) -> list[int]:
    """Уроки, созданные автообучением из указанного исправления."""
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT lesson_id FROM training_log
            WHERE correction_id = ? AND lesson_id IS NOT NULL
            """,
            (correction_id,),
        ).fetchall()
    return [int(row["lesson_id"]) for row in rows]


def delete_training_log_for_correction(correction_id: int) -> int:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute(
            "DELETE FROM training_log WHERE correction_id = ?", (correction_id,)
        )
    return int(cur.rowcount)


def get_setting(key: str, default: str = "") -> str:
    """Значение из отдельной БД настроек (settings.db)."""
    return settingsdb.get(key, default)


def set_setting(key: str, value: str) -> str:
    """Записать значение в отдельную БД настроек (settings.db)."""
    settingsdb.set(key, value)
    return value


DOMAIN_JSON_FIELDS = ("capabilities", "categories", "terminology", "sample_questions")


def _decode_domain(row: sqlite3.Row) -> dict:
    item = dict(row)
    item.pop("log_analysis", None)
    for field in DOMAIN_JSON_FIELDS:
        try:
            value = json.loads(item.get(field) or "[]")
            item[field] = value if isinstance(value, list) else []
        except json.JSONDecodeError:
            item[field] = []
    try:
        value = json.loads(item.get("settings") or "{}")
        item["settings"] = value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        item["settings"] = {}
    try:
        value = json.loads(item.get("synonyms") or "{}")
        item["synonyms"] = _decode_synonyms(value)
    except json.JSONDecodeError:
        item["synonyms"] = {}
    try:
        item["temperature"] = (
            float(item["temperature"]) if item.get("temperature") is not None else None
        )
    except (TypeError, ValueError):
        item["temperature"] = None
    if item["temperature"] is not None and item["temperature"] < 0:
        item["temperature"] = None
    item["allow_inference"] = bool(item.get("allow_inference"))
    return item


def _decode_synonyms(value) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = []
        for entry in value:
            if isinstance(entry, dict):
                items.append((entry.get("term", ""), entry.get("synonyms", [])))
    else:
        return result
    for term, synonyms in items:
        key = str(term).strip()
        if not key:
            continue
        if isinstance(synonyms, str):
            synonyms = [part.strip() for part in synonyms.split(",")]
        if not isinstance(synonyms, list):
            continue
        clean = [str(s).strip() for s in synonyms if str(s).strip()]
        if clean:
            result[key] = clean
    return result


def list_domains() -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM domains ORDER BY is_builtin DESC, id"
        ).fetchall()
    return [_decode_domain(row) for row in rows]


def get_domain_by_slug(slug: str) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT * FROM domains WHERE slug = ?", (slug,)).fetchone()
    return _decode_domain(row) if row else None


def count_domains() -> int:
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM domains").fetchone()
    return int(row["n"])


def insert_domain(
    slug: str,
    title: str,
    description: str = "",
    system_prompt: str = "",
    capabilities: list | None = None,
    categories: list | None = None,
    terminology: list | None = None,
    sample_questions: list | None = None,
    temperature: float | None = None,
    allow_inference: bool = False,
    settings: dict | None = None,
    synonyms: dict | None = None,
    is_builtin: bool = False,
) -> dict:
    ensure_schema()
    stamp = now()
    with _LOCK, connect() as conn:
        conn.execute(
            """
            INSERT INTO domains
                (slug, title, description, system_prompt, capabilities, categories,
                 terminology, sample_questions, temperature, allow_inference,
                 settings, synonyms, is_builtin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                slug,
                title,
                description,
                system_prompt,
                json.dumps(capabilities or [], ensure_ascii=False),
                json.dumps(categories or [], ensure_ascii=False),
                json.dumps(terminology or [], ensure_ascii=False),
                json.dumps(sample_questions or [], ensure_ascii=False),
                temperature,
                1 if allow_inference else 0,
                json.dumps(settings or {}, ensure_ascii=False),
                json.dumps(synonyms or {}, ensure_ascii=False),
                1 if is_builtin else 0,
                stamp,
                stamp,
            ),
        )
    return get_domain_by_slug(slug) or {}


def update_domain(slug: str, **fields) -> dict | None:
    ensure_schema()
    allowed = {
        "slug", "title", "description", "system_prompt",
        "capabilities", "categories", "terminology", "sample_questions",
        "temperature", "allow_inference", "is_builtin",
        "settings", "synonyms",
    }
    sets: list[str] = []
    params: list = []
    for key, value in fields.items():
        if key not in allowed:
            continue
        if key in DOMAIN_JSON_FIELDS:
            value = json.dumps(value or [], ensure_ascii=False)
        elif key == "settings":
            value = json.dumps(value or {}, ensure_ascii=False)
        elif key == "synonyms":
            value = json.dumps(value or {}, ensure_ascii=False)
        elif key in ("allow_inference", "is_builtin"):
            value = 1 if value else 0
        sets.append(f"{key} = ?")
        params.append(value)
    if not sets:
        return get_domain_by_slug(slug)
    sets.append("updated_at = ?")
    params.append(now())
    params.append(slug)
    with _LOCK, connect() as conn:
        conn.execute(f"UPDATE domains SET {', '.join(sets)} WHERE slug = ?", params)
    return get_domain_by_slug(fields.get("slug", slug))


def delete_domain(slug: str) -> bool:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute("DELETE FROM domains WHERE slug = ?", (slug,))
        return cur.rowcount > 0


def excluded_keys() -> set[tuple[str, str]]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute("SELECT root, source FROM doc_exclusions").fetchall()
    return {(row["root"], row["source"]) for row in rows}


def remove_exclusion(root: str, source: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            "DELETE FROM doc_exclusions WHERE root = ? AND source = ?",
            (root, source),
        )


def correction_ids_by_question() -> dict[str, int]:
    """Карта «вопрос в нижнем регистре» -> id исправления."""
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, question FROM corrections WHERE status = 'active'"
        ).fetchall()
    return {row["question"].strip().lower(): int(row["id"]) for row in rows if row["question"]}


def count_requests(
    vote: str | None = None,
    username: str | None = None,
    search: str | None = None,
    origin: str | None = None,
) -> int:
    ensure_schema()
    query = """
        SELECT COUNT(*) AS n FROM requests r
        LEFT JOIN ratings t ON t.request_id = r.id WHERE 1 = 1
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
    if origin == "web":
        query += " AND (r.origin = '' OR r.origin = 'web')"
    elif origin:
        query += " AND r.origin = ?"
        params.append(origin)
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
            WHERE u.hidden = 0
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
                (SELECT COUNT(*) FROM users WHERE hidden = 0) AS users,
                (SELECT COUNT(*) FROM users WHERE active = 1 AND hidden = 0) AS active_users,
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


def get_request(request_id: int) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
    return dict(row) if row else None


def get_request_full(request_id: int) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT r.*, t.vote, t.comment AS rating_comment
            FROM requests r LEFT JOIN ratings t ON t.request_id = r.id
            WHERE r.id = ?
            """,
            (request_id,),
        ).fetchone()
    return dict(row) if row else None


def recent_requests(limit: int = 500, days: int | None = None) -> list[dict]:
    ensure_schema()
    query = """
        SELECT r.id, r.question, r.answer, r.grounded, r.corrected, r.sources,
               r.latency_ms, r.origin, r.created_at, t.vote
        FROM requests r LEFT JOIN ratings t ON t.request_id = r.id
    """
    params: list = []
    if days:
        query += " WHERE r.created_at >= datetime('now', ?)"
        params.append(f"-{int(days)} days")
    query += " ORDER BY r.id DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def upsert_request_eval(
    request_id: int,
    verdict: str,
    score: float,
    reason: str,
    method: str = "heuristic",
    suggested_answer: str = "",
) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            """
            INSERT INTO request_evals
                (request_id, verdict, score, reason, method, suggested_answer, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                verdict = excluded.verdict,
                score = excluded.score,
                reason = excluded.reason,
                method = excluded.method,
                suggested_answer = excluded.suggested_answer,
                updated_at = excluded.updated_at
            """,
            (request_id, verdict, float(score), reason[:500], method, suggested_answer, now(), now()),
        )


def get_request_eval(request_id: int) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM request_evals WHERE request_id = ?", (request_id,)
        ).fetchone()
    return dict(row) if row else None


def list_request_evals(
    limit: int = 100,
    offset: int = 0,
    verdict: str | None = None,
    only_problems: bool = False,
    origin: str | None = None,
) -> list[dict]:
    ensure_schema()
    query = """
        SELECT e.request_id, e.verdict, e.score, e.reason, e.method, e.suggested_answer,
               e.updated_at, r.question, r.answer, r.username, r.grounded, r.corrected,
               r.origin, r.sources, r.created_at AS requested_at, t.vote
        FROM request_evals e
        JOIN requests r ON r.id = e.request_id
        LEFT JOIN ratings t ON t.request_id = r.id
        WHERE 1 = 1
    """
    params: list = []
    if verdict:
        query += " AND e.verdict = ?"
        params.append(verdict)
    if only_problems:
        query += " AND e.verdict IN ('partial', 'wrong', 'refused_wrongly')"
    if origin:
        query += " AND r.origin = ?"
        params.append(origin)
    query += " ORDER BY e.request_id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def count_request_evals(
    verdict: str | None = None,
    only_problems: bool = False,
    origin: str | None = None,
) -> int:
    ensure_schema()
    query = """
        SELECT COUNT(*) AS n FROM request_evals e
        JOIN requests r ON r.id = e.request_id
        WHERE 1 = 1
    """
    params: list = []
    if verdict:
        query += " AND e.verdict = ?"
        params.append(verdict)
    if only_problems:
        query += " AND e.verdict IN ('partial', 'wrong', 'refused_wrongly')"
    if origin:
        query += " AND r.origin = ?"
        params.append(origin)
    with connect() as conn:
        row = conn.execute(query, params).fetchone()
    return int(row["n"])


def eval_summary() -> dict:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM request_evals GROUP BY verdict"
        ).fetchall()
    counts = {row["verdict"]: int(row["n"]) for row in rows}
    total = sum(counts.values())
    problems = sum(counts.get(key, 0) for key in ("partial", "wrong", "refused_wrongly"))
    return {"total": total, "problems": problems, "by_verdict": counts}


def add_issue(request_id: int, kind: str, detail: str = "") -> int | None:
    ensure_schema()
    with _LOCK, connect() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO answer_issues
                    (request_id, kind, detail, status, created_at, updated_at)
                VALUES (?, ?, ?, 'open', ?, ?)
                """,
                (request_id, kind, detail[:500], now(), now()),
            )
        except sqlite3.IntegrityError:
            return None
        return int(cur.lastrowid)


def list_issues(status: str | None = "open", limit: int = 200, offset: int = 0) -> list[dict]:
    ensure_schema()
    query = """
        SELECT i.id, i.request_id, i.kind, i.detail, i.status, i.lesson_id, i.topic_id, i.created_at,
               r.question, r.answer, r.username, r.created_at AS requested_at, t.vote
        FROM answer_issues i
        JOIN requests r ON r.id = i.request_id
        LEFT JOIN ratings t ON t.request_id = i.request_id
        WHERE 1 = 1
    """
    params: list = []
    if status:
        query += " AND i.status = ?"
        params.append(status)
    query += " ORDER BY i.id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def count_issues(status: str | None = "open") -> int:
    ensure_schema()
    query = "SELECT COUNT(*) AS n FROM answer_issues WHERE 1 = 1"
    params: list = []
    if status:
        query += " AND status = ?"
        params.append(status)
    with connect() as conn:
        row = conn.execute(query, params).fetchone()
    return int(row["n"])


def issues_for_request(request_id: int) -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, kind, detail, status FROM answer_issues WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_issue(
    issue_id: int,
    status: str | None = None,
    lesson_id: int | None = None,
    detail: str | None = None,
) -> None:
    ensure_schema()
    fields: list = []
    params: list = []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if lesson_id is not None:
        fields.append("lesson_id = ?")
        params.append(lesson_id)
    if detail is not None:
        fields.append("detail = ?")
        params.append(detail[:500])
    if not fields:
        return
    fields.append("updated_at = ?")
    params.append(now())
    params.append(issue_id)
    with _LOCK, connect() as conn:
        conn.execute(f"UPDATE answer_issues SET {', '.join(fields)} WHERE id = ?", params)


def close_issues(request_id: int, kinds: tuple[str, ...]) -> None:
    if not kinds:
        return
    ensure_schema()
    placeholders = ", ".join("?" for _ in kinds)
    with _LOCK, connect() as conn:
        conn.execute(
            f"UPDATE answer_issues SET status = 'resolved', updated_at = ? "
            f"WHERE request_id = ? AND status = 'open' AND kind IN ({placeholders})",
            (now(), request_id, *kinds),
        )


def problem_request_ids() -> set[int]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT request_id FROM request_evals WHERE verdict IN ('partial', 'wrong', 'refused_wrongly')"
        ).fetchall()
    return {int(row["request_id"]) for row in rows}


def requests_for_llm(limit: int = 20) -> list[int]:
    """Запросы, ещё не оценённые моделью: сначала проблемные, потом свежие."""
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT r.id AS request_id,
                   CASE WHEN e.verdict IN ('partial', 'wrong', 'refused_wrongly') THEN 0 ELSE 1 END AS priority
            FROM requests r
            LEFT JOIN request_evals e ON e.request_id = r.id
            WHERE e.request_id IS NULL OR e.method != 'llm'
            ORDER BY priority, r.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [int(row["request_id"]) for row in rows]


def issue_summary() -> dict:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM answer_issues GROUP BY status"
        ).fetchall()
        kinds = conn.execute(
            "SELECT kind, COUNT(*) AS n FROM answer_issues WHERE status = 'open' GROUP BY kind"
        ).fetchall()
    counts = {row["status"]: int(row["n"]) for row in rows}
    return {
        "total": sum(counts.values()),
        "open": counts.get("open", 0),
        "resolved": counts.get("resolved", 0),
        "dismissed": counts.get("dismissed", 0),
        "by_kind": {row["kind"]: int(row["n"]) for row in kinds},
    }


def replace_topics(items: list[dict]) -> int:
    """Пересобрать ассоциации вопросов: темы и их состав."""
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("UPDATE answer_issues SET topic_id = NULL")
        conn.execute("DELETE FROM topic_requests")
        conn.execute("DELETE FROM question_topics")
        for item in items:
            cur = conn.execute(
                """
                INSERT INTO question_topics
                    (title, representative, total, problematic, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.get("title", "")[:200],
                    item.get("representative", ""),
                    int(item.get("total", 0)),
                    int(item.get("problematic", 0)),
                    now(),
                    now(),
                ),
            )
            topic_id = int(cur.lastrowid)
            for request_id, score in item.get("members", []):
                conn.execute(
                    "INSERT OR REPLACE INTO topic_requests (topic_id, request_id, score) VALUES (?, ?, ?)",
                    (topic_id, int(request_id), float(score)),
                )
                conn.execute(
                    "UPDATE answer_issues SET topic_id = ? WHERE request_id = ?",
                    (topic_id, int(request_id)),
                )
    return len(items)


def list_topics(limit: int = 80) -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM question_topics ORDER BY problematic DESC, total DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def topic_requests(topic_id: int, limit: int = 50) -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT tr.request_id, tr.score, r.question, r.answer, r.created_at,
                   e.verdict
            FROM topic_requests tr
            JOIN requests r ON r.id = tr.request_id
            LEFT JOIN request_evals e ON e.request_id = tr.request_id
            WHERE tr.topic_id = ?
            ORDER BY tr.score DESC LIMIT ?
            """,
            (topic_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def normalize_question(text: str) -> str:
    """Нормализованная форма вопроса для точного сопоставления."""
    value = re.sub(r"[^\w\s]", " ", (text or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def save_generated_question(
    question: str,
    source: str = "docs",
    doc_source: str = "",
    doc_chunk: int | None = None,
    variant: int = 0,
) -> dict | None:
    """Запомнить сгенерированный вопрос. Точный дубль возвращает существующую запись."""
    ensure_schema()
    text = (question or "").strip()
    if len(text) < 4:
        return None
    norm = normalize_question(text)
    if not norm:
        return None
    with _LOCK, connect() as conn:
        row = conn.execute(
            "SELECT * FROM generated_questions WHERE norm = ?", (norm,)
        ).fetchone()
        if row:
            return dict(row)
        cur = conn.execute(
            """
            INSERT INTO generated_questions
                (question, norm, source, doc_source, doc_chunk, variant, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                norm,
                source,
                doc_source,
                int(doc_chunk) if doc_chunk is not None else None,
                int(variant),
                now(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM generated_questions WHERE id = ?", (int(cur.lastrowid),)
        ).fetchone()
    return dict(row) if row else None


def existing_generated_norms() -> set[str]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute("SELECT norm FROM generated_questions").fetchall()
    return {row["norm"] for row in rows}


def update_generated_result(qid: int, request_id: int, verdict: str) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            "UPDATE generated_questions SET request_id = ?, verdict = ? WHERE id = ?",
            (int(request_id), verdict or "", int(qid)),
        )


def store_generated_vector(qid: int, blob: bytes) -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute(
            "UPDATE generated_questions SET vector = ? WHERE id = ?", (blob, int(qid))
        )


def list_generated_questions(
    limit: int = 100,
    offset: int = 0,
    source: str | None = None,
    only_duplicates: bool = False,
) -> list[dict]:
    ensure_schema()
    query = """
        SELECT id, question, source, doc_source, doc_chunk, variant, request_id,
               verdict, cluster_id, duplicate_of, similarity, created_at
        FROM generated_questions WHERE 1 = 1
    """
    params: list = []
    if source:
        query += " AND source = ?"
        params.append(source)
    if only_duplicates:
        query += " AND duplicate_of IS NOT NULL"
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def count_generated_questions(source: str | None = None, only_duplicates: bool = False) -> int:
    ensure_schema()
    query = "SELECT COUNT(*) AS n FROM generated_questions WHERE 1 = 1"
    params: list = []
    if source:
        query += " AND source = ?"
        params.append(source)
    if only_duplicates:
        query += " AND duplicate_of IS NOT NULL"
    with connect() as conn:
        return int(conn.execute(query, params).fetchone()["n"])


def generated_vectors(limit: int = 2000) -> list[dict]:
    """Все запомненные вопросы с кэшированными векторами (свежие первыми)."""
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, question, vector FROM generated_questions
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def set_generated_similarity(assignments: list[dict]) -> int:
    """Записать результаты анализа схожести: кластер и почти-дубль."""
    ensure_schema()
    with _LOCK, connect() as conn:
        for item in assignments:
            conn.execute(
                """
                UPDATE generated_questions
                SET cluster_id = ?, duplicate_of = ?, similarity = ?
                WHERE id = ?
                """,
                (
                    item.get("cluster_id"),
                    item.get("duplicate_of"),
                    float(item.get("similarity", 0)),
                    int(item["id"]),
                ),
            )
    return len(assignments)


def generated_clusters(limit: int = 50) -> list[dict]:
    """Кластеры похожих вопросов: представитель и число формулировок."""
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT cluster_id,
                   COUNT(*) AS total,
                   SUM(CASE WHEN duplicate_of IS NOT NULL THEN 1 ELSE 0 END) AS duplicates,
                   MAX(similarity) AS best_similarity
            FROM generated_questions
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id
            HAVING COUNT(*) > 1
            ORDER BY total DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        clusters = []
        for row in rows:
            cluster_id = int(row["cluster_id"])
            members = conn.execute(
                """
                SELECT id, question, source, doc_source, duplicate_of, similarity, verdict
                FROM generated_questions WHERE cluster_id = ?
                ORDER BY duplicate_of IS NULL DESC, similarity DESC, id ASC
                """,
                (cluster_id,),
            ).fetchall()
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "total": int(row["total"]),
                    "duplicates": int(row["duplicates"] or 0),
                    "representative": (members[0]["question"] if members else ""),
                    "members": [dict(member) for member in members],
                }
            )
    return clusters


def generated_similarity_summary() -> dict:
    ensure_schema()
    with connect() as conn:
        total = int(conn.execute("SELECT COUNT(*) AS n FROM generated_questions").fetchone()["n"])
        duplicates = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM generated_questions WHERE duplicate_of IS NOT NULL"
            ).fetchone()["n"]
        )
        clusters = int(
            conn.execute(
                """
                SELECT COUNT(*) AS n FROM (
                    SELECT cluster_id FROM generated_questions
                    WHERE cluster_id IS NOT NULL
                    GROUP BY cluster_id HAVING COUNT(*) > 1
                )
                """
            ).fetchone()["n"]
        )
    return {"total": total, "clusters": clusters, "duplicates": duplicates}


def clear_generated_questions() -> None:
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM generated_questions")
