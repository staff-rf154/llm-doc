"""Словарь синонимов: отдельная БД SQLite, используется для расширения запроса."""

import threading
from datetime import datetime

import config
import db
import settings

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


def connect():
    return db.connect("synonyms")


def ensure_schema() -> None:
    global _INIT_DONE
    with _LOCK:
        if _INIT_DONE:
            return
        with connect() as conn:
            db.exec_script(conn, SCHEMA)
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
            "SELECT term, synonyms, author, updated_at FROM synonyms WHERE term = %s",
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
            VALUES (%s, %s, %s, %s)
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
        cur = conn.execute("DELETE FROM synonyms WHERE term = %s", (_normalize(term),))
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
                VALUES (%s, %s, %s, %s)
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
        values = [s.strip() for s in row["synonyms"].split(",") if s.strip()]
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
    if _cache["map"] and now - _cache["at"] < settings.int_value("SYNONYMS_CACHE_TTL", config.SYNONYMS_CACHE_TTL):
        base_map = _cache["map"]
    else:
        base_map = {}
        for row in list_terms():
            values = [s.strip() for s in row["synonyms"].split(",") if s.strip()]
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


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    buffer = ""
    start = 0
    for index, char in enumerate(text):
        if char.isalnum():
            if not buffer:
                start = index
            buffer += char.lower()
        else:
            if buffer:
                spans.append((buffer, start, index))
                buffer = ""
    if buffer:
        spans.append((buffer, start, len(text)))
    return spans


def _find_phrase(text: str, phrase: str) -> tuple[int, int] | None:
    words = _tokenize(phrase)
    if not words:
        return None
    spans = _token_spans(text)
    if len(words) > len(spans):
        return None
    needles = [stem(word) for word in words]
    for index in range(len(spans) - len(needles) + 1):
        if all(stem(spans[index + offset][0]) == needles[offset] for offset in range(len(needles))):
            return spans[index][1], spans[index + len(needles) - 1][2]
    return None


def _replace_word(base: str, word: str, value: str) -> str:
    span = _find_phrase(base, word)
    if not span:
        return base
    start, end = span
    return " ".join((base[:start] + value + base[end:]).split())


def _replace_phrase(base: str, phrase: str, value: str) -> str:
    span = _find_phrase(base, phrase)
    if not span:
        return base
    start, end = span
    return " ".join((base[:start] + value + base[end:]).split())


def _stems(text: str) -> set[str]:
    return {stem(word) for word in _tokenize(text)}


def matched_terms(question: str, max_terms: int = 8) -> list[tuple[str, list[str]]]:
    """Термины словаря, которые встречаются в вопросе прямо или через синоним."""
    base = " ".join(question.split())
    if not base:
        return []
    mapping = _load_map()
    if not mapping:
        return []
    lower = base.lower()
    base_stems = _stems(base)
    limit = max(1, int(max_terms))
    result: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for term, values in mapping.items():
        term_norm = _normalize(term)
        if not term_norm:
            continue
        matched = False
        if " " in term_norm:
            matched = _find_phrase(lower, term_norm) is not None
        else:
            matched = stem(term_norm) in base_stems
        if not matched:
            for value in values:
                value_norm = _normalize(value)
                if not value_norm or value_norm == term_norm:
                    continue
                if " " in value_norm:
                    if _find_phrase(lower, value_norm) is not None:
                        matched = True
                        break
                elif stem(value_norm) in base_stems:
                    matched = True
                    break
        if matched and term_norm not in seen:
            seen.add(term_norm)
            result.append((term_norm, list(values)))
            if len(result) >= limit:
                break
    return result


def expand(question: str, max_variants: int = 6) -> list[str]:
    """Варианты запроса: оригинал плюс подстановки синонимов в обе стороны."""
    base = " ".join(question.split())
    mapping = _load_map()
    if not mapping:
        return [base]

    limit = max(1, int(max_variants))
    variants = [base]
    base_stems = _stems(base)

    def add_variant(candidate: str) -> None:
        candidate = " ".join(candidate.split())
        if (
            candidate
            and candidate != base
            and candidate not in variants
            and len(variants) < limit
        ):
            variants.append(candidate)

    for term, values in mapping.items():
        term_norm = _normalize(term)
        if not term_norm:
            continue
        if " " in term_norm:
            if _find_phrase(base, term_norm) is not None:
                for value in values:
                    add_variant(_replace_phrase(base, term_norm, value))
            if len(variants) >= limit:
                break
            continue
        for word in _tokenize(base):
            if stem(word) != stem(term_norm):
                continue
            for value in values:
                head = _tokenize(value)[0] if _tokenize(value) else ""
                if head and stem(head) in base_stems:
                    continue
                add_variant(_replace_word(base, word, value))
            break
        if len(variants) >= limit:
            break

    if len(variants) < limit:
        hits: list[tuple[str, str, str, list[str]]] = []
        for term, values in mapping.items():
            term_norm = _normalize(term)
            if not term_norm:
                continue
            for value in values:
                value_norm = _normalize(value)
                if not value_norm or value_norm == term_norm:
                    continue
                if " " in value_norm:
                    if _find_phrase(base, value_norm) is not None:
                        hits.append((term_norm, value_norm, "", values))
                else:
                    for word in _tokenize(base):
                        if stem(word) == stem(value_norm):
                            hits.append((term_norm, value_norm, word, values))
                            break
        for term_norm, value_norm, word, _ in hits:
            if word:
                add_variant(_replace_word(base, word, term_norm))
            else:
                add_variant(_replace_phrase(base, value_norm, term_norm))
            if len(variants) >= limit:
                break
        if len(variants) < limit:
            for term_norm, value_norm, word, values in hits:
                for other in values:
                    other_norm = _normalize(other)
                    if not other_norm or other_norm == value_norm:
                        continue
                    if word:
                        add_variant(_replace_word(base, word, other_norm))
                    else:
                        add_variant(_replace_phrase(base, value_norm, other_norm))
                    if len(variants) >= limit:
                        break
                if len(variants) >= limit:
                    break

    return variants
