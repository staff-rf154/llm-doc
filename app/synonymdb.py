"""Словарь синонимов: отдельная БД SQLite, используется для расширения запроса."""

import re
import sqlite3
import threading
from datetime import datetime

import config

_LOCK = threading.RLock()
_INIT_DONE = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS synonyms (
    term TEXT PRIMARY KEY,
    synonyms TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SYNONYMS_DB, timeout=30, check_same_thread=False)
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


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def list_terms() -> list[dict]:
    ensure_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT term, synonyms, author, updated_at FROM synonyms ORDER BY term"
        ).fetchall()
    return [dict(row) for row in rows]


def get_term(term: str) -> dict | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT term, synonyms, author, updated_at FROM synonyms WHERE term = ?",
            (_normalize(term),),
        ).fetchone()
    return dict(row) if row else None


def set_term(term: str, synonyms: list[str], author: str = "") -> dict:
    ensure_schema()
    clean = []
    for item in synonyms:
        value = _normalize(str(item))
        if value and value != _normalize(term) and value not in clean:
            clean.append(value)
    with _LOCK, connect() as conn:
        conn.execute(
            """
            INSERT INTO synonyms (term, synonyms, author, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(term) DO UPDATE SET
                synonyms = excluded.synonyms,
                author = excluded.author,
                updated_at = excluded.updated_at
            """,
            (_normalize(term), ", ".join(clean), author, _now()),
        )
    return get_term(term) or {}


def delete_term(term: str) -> bool:
    ensure_schema()
    with _LOCK, connect() as conn:
        cur = conn.execute("DELETE FROM synonyms WHERE term = ?", (_normalize(term),))
    return bool(cur.rowcount)


def import_many(items: dict[str, list[str]], author: str = "") -> int:
    ensure_schema()
    with _LOCK, connect() as conn:
        now = _now()
        for term, synonyms in items.items():
            clean = []
            for item in synonyms:
                value = _normalize(str(item))
                if value and value != _normalize(term) and value not in clean:
                    clean.append(value)
            conn.execute(
                """
                INSERT INTO synonyms (term, synonyms, author, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(term) DO UPDATE SET
                    synonyms = excluded.synonyms,
                    author = excluded.author,
                    updated_at = excluded.updated_at
                """,
                (_normalize(term), ", ".join(clean), author, now),
            )
    invalidate()
    return count()


def count() -> int:
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM synonyms").fetchone()
    return int(row["n"])


def export_many() -> dict[str, list[str]]:
    """Текущий словарь в виде {термин: [синонимы]}."""
    ensure_schema()
    with connect() as conn:
        rows = conn.execute("SELECT term, synonyms FROM synonyms ORDER BY term").fetchall()
    result: dict[str, list[str]] = {}
    for row in rows:
        values = [s for s in row["synonyms"].split(",") if s]
        if values:
            result[row["term"]] = values
    return result


def replace_all(items: dict[str, list[str]], author: str = "") -> int:
    """Полностью заменить словарь указанным набором."""
    ensure_schema()
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM synonyms")
    invalidate()
    if not items:
        return 0
    return import_many(items, author)


_cache: dict = {"at": 0.0, "map": {}}


def _domain_map() -> dict[str, list[str]]:
    """Синонимы активной сферы деятельности поверх глобального словаря."""
    try:
        import domains

        return domains.active_synonyms()
    except Exception:
        return {}


def _load_map() -> dict[str, list[str]]:
    ensure_schema()
    now = datetime.now().timestamp()
    if _cache["map"] and now - _cache["at"] < config.SYNONYMS_CACHE_TTL:
        base_map = _cache["map"]
    else:
        base_map = {}
        for row in list_terms():
            values = [s for s in row["synonyms"].split(",") if s]
            if values:
                base_map[row["term"]] = values
        _cache.update(at=now, map=base_map)
    mapping = dict(base_map)
    for term, values in _domain_map().items():
        existing = mapping.get(term) or []
        merged = list(values)
        for value in existing:
            if value not in merged:
                merged.append(value)
        mapping[term] = merged
    return mapping


def invalidate() -> None:
    _cache.update(at=0.0, map={})


ENDINGS = (
    "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ах", "ях",
    "ия", "ию", "ий", "ый", "ое", "ые", "ой", "ем", "ом", "ов", "ев", "ам", "ям",
    "ую", "юю", "ая", "яя", "ое", "ые", "ы", "и", "а", "я", "о", "е", "у", "ю", "ь",
)

MIN_STEM = 4


def stem(word: str) -> str:
    """Грубая нормализация окончания, чтобы «лицензию» находило термин «лицензия»."""
    value = word.lower()
    for ending in ENDINGS:
        if len(value) - len(ending) >= MIN_STEM and value.endswith(ending):
            return value[: -len(ending)]
    return value


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    buffer = ""
    for char in text:
        if char.isalnum():
            buffer += char.lower()
        else:
            if buffer:
                tokens.append(buffer)
            buffer = ""
    if buffer:
        tokens.append(buffer)
    return tokens


def _replace_word(base: str, word: str, value: str) -> str:
    replaced = re.sub(re.escape(word), value, base, count=1, flags=re.IGNORECASE)
    return " ".join(replaced.split())


def _stems(text: str) -> set[str]:
    return {stem(word) for word in _tokenize(text)}


def expand(question: str, max_variants: int = 6) -> list[str]:
    """Варианты запроса: оригинал плюс подстановки синонимов найденных слов."""
    base = " ".join(question.split())
    mapping = _load_map()
    if not mapping:
        return [base]

    variants = [base]
    base_stems = _stems(base)

    def add_variant(candidate: str) -> None:
        if candidate != base and candidate not in variants and len(variants) <= max_variants:
            variants.append(candidate)

    for term, values in mapping.items():
        if " " in term:
            if term in base.lower():
                for value in values:
                    add_variant(
                        " ".join(re.sub(re.escape(term), value, base, flags=re.IGNORECASE).split())
                    )
            continue
        for word in _tokenize(base):
            if stem(word) != stem(term):
                continue
            for value in values:
                head = _tokenize(value)[0] if _tokenize(value) else ""
                if head and stem(head) in base_stems:
                    continue
                add_variant(_replace_word(base, word, value))
            break
        if len(variants) > max_variants:
            break

    return variants
