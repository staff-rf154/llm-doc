"""Подключение к PostgreSQL и общие помощники для слоя БД."""

import json
import os
import threading
from contextlib import contextmanager
from urllib.parse import quote, unquote, urlsplit

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

import config

SCHEMAS = ("app", "settings", "prompts", "synonyms", "index_state")
REQUIRED_TABLES = (
    "app.tenants",
    "app.users",
    "app.sessions",
    "app.requests",
    "app.ratings",
    "app.corrections",
    "app.lessons",
    "app.training_log",
    "app.settings",
    "app.request_evals",
    "app.answer_issues",
    "app.question_topics",
    "app.topic_requests",
    "app.generated_questions",
    "app.domains",
    "app.doc_exclusions",
    "settings.settings",
    "settings.meta",
    "prompts.prompts",
    "synonyms.synonyms",
    "index_state.runs",
    "index_state.file_progress",
)
LOCAL_HOSTS = {"postgres", "llm-postgres", "localhost", "127.0.0.1", "::1"}

_POOL = None
_POOL_LOCK = threading.RLock()


def _settings_path() -> str:
    return os.path.join(os.path.dirname(config.APP_DB), "db_connection.json")


def _parse_env() -> dict:
    raw = getattr(config, "DATABASE_URL", "") or ""
    try:
        parts = urlsplit(raw)
        host = parts.hostname or "postgres"
        port = parts.port or 5432
        database = (parts.path or "/").lstrip("/") or "postgres"
        user = unquote(parts.username or "")
        password = unquote(parts.password or "")
    except ValueError:
        host, port, database, user, password = "postgres", 5432, "postgres", "", ""
    return {
        "host": host,
        "port": int(port),
        "database": database,
        "user": user,
        "password": password,
    }


def load_settings() -> dict:
    base = _parse_env()
    path = _settings_path()
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return base
    if not isinstance(stored, dict):
        return base
    for key in ("host", "database", "user"):
        value = stored.get(key)
        if isinstance(value, str) and value:
            base[key] = value
    password = stored.get("password")
    if isinstance(password, str):
        base["password"] = password
    try:
        port = int(stored.get("port", base["port"]))
        if 1 <= port <= 65535:
            base["port"] = port
    except (TypeError, ValueError):
        pass
    return base


def describe_settings() -> dict:
    settings = load_settings()
    return {
        "host": settings["host"],
        "port": settings["port"],
        "database": settings["database"],
        "user": settings["user"],
        "password_set": bool(settings["password"]),
        "source": "file" if os.path.exists(_settings_path()) else "env",
    }


def _build_dsn(settings: dict) -> str:
    user = quote(settings["user"], safe="")
    password = quote(settings["password"], safe="")
    auth = f"{user}:{password}" if user else ""
    host = settings["host"]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    database = quote(settings["database"], safe="")
    return f"postgresql://{auth}@{host}:{settings['port']}/{database}"


def dsn() -> str:
    value = getattr(config, "DATABASE_URL", "")
    settings = load_settings()
    if value and not os.path.exists(_settings_path()):
        return value
    return _build_dsn(settings)


def _test_connection(settings: dict, check_tables: bool = False) -> None:
    with psycopg.connect(_build_dsn(settings), connect_timeout=5) as conn:
        conn.execute("SELECT 1")
        if check_tables:
            missing = [
                name
                for name in REQUIRED_TABLES
                if conn.execute("SELECT to_regclass(%s)", (name,)).fetchone()[0] is None
            ]
            if missing:
                raise ValueError(
                    "в целевой базе нет таблиц приложения: " + ", ".join(missing)
                )


def _write_settings(settings: dict) -> None:
    path = _settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    payload = {
        "host": settings["host"],
        "port": int(settings["port"]),
        "database": settings["database"],
        "user": settings["user"],
        "password": settings["password"],
    }
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _alter_role_password(settings: dict, user: str, password: str) -> None:
    with psycopg.connect(_build_dsn(settings), connect_timeout=5) as conn:
        conn.execute(
            sql.SQL("ALTER ROLE {} WITH PASSWORD {}").format(
                sql.Identifier(user), sql.Literal(password)
            )
        )


def _is_local(host: str) -> bool:
    return (host or "").lower() in LOCAL_HOSTS


def apply_settings(host: str, port, database: str, user: str, password: str = "") -> dict:
    host = (host or "").strip()
    database = (database or "").strip()
    user = (user or "").strip()
    if not host or not database or not user:
        raise ValueError("Заполните хост, базу и пользователя")
    try:
        port = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError("Порт должен быть числом") from exc
    if not 1 <= port <= 65535:
        raise ValueError("Порт должен быть в диапазоне 1–65535")

    with _POOL_LOCK:
        old = load_settings()
        new = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password if password else old["password"],
        }
        role_changed = False
        if (
            password
            and password != old["password"]
            and _is_local(host)
            and _is_local(old["host"])
            and user == old["user"]
        ):
            try:
                _alter_role_password(old, user, password)
                role_changed = True
            except Exception:
                role_changed = False
        try:
            _test_connection(new, check_tables=True)
        except Exception as exc:
            if role_changed:
                try:
                    _alter_role_password(new, user, old["password"])
                except Exception:
                    pass
            raise ValueError(f"Не удалось подключиться: {exc}") from exc

        try:
            _write_settings(new)
            _recreate_pool(new)
        except Exception as exc:
            try:
                _write_settings(old)
            except Exception:
                pass
            try:
                _recreate_pool(old)
            except Exception:
                pass
            if role_changed:
                try:
                    _alter_role_password(new, user, old["password"])
                except Exception:
                    pass
            raise ValueError(f"Не удалось применить настройки: {exc}") from exc
        return describe_settings()


def _create_pool(settings: dict | None = None) -> ConnectionPool:
    pool = ConnectionPool(
        _build_dsn(settings) if settings else dsn(),
        min_size=1,
        max_size=10,
        timeout=30,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    pool.open()
    pool.wait(timeout=30)
    with pool.connection() as conn:
        for schema in SCHEMAS:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    return pool


def _recreate_pool(settings: dict) -> None:
    global _POOL
    old_pool = _POOL
    _POOL = None
    if old_pool is not None:
        try:
            old_pool.close()
        except Exception:
            pass
    _POOL = _create_pool(settings)


def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = _create_pool()
    return _POOL


@contextmanager
def connect(schema: str):
    if schema not in SCHEMAS:
        raise ValueError(f"Неизвестная схема: {schema}")
    with get_pool().connection() as conn:
        conn.execute(f"SET search_path TO {schema}, public")
        yield conn


def exec_script(conn, script: str) -> None:
    for statement in script.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
