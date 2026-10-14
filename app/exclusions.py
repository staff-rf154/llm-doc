"""Исключения документов из ответов.

Источники правил:
- комментарий пользователя к оценке: «исключить из ответа FileAuditor_manual.pdf» —
  такие файлы разбираются автоматически и попадают сюда же;
- вкладка «Исключения» в кабинете администратора.
"""

import re
import time

import appdb
import config

TRIGGERS = (
    r"исключить",
    r"исключи",
    r"не использовать",
    r"не выводи",
    r"не показывай",
    r"не показывать",
    r"убрать",
    r"убери",
    r"не применять",
    r"не учитывать",
)

_TRIGGER_RE = re.compile(
    "(" + "|".join(TRIGGERS) + r")\s*(?:из\s+ответа\s*)?[:\-–—]?\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)

_CLEAN_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._\-/ ]")
_STOP_WORDS = {
    "документ",
    "документы",
    "файл",
    "файлы",
    "и",
    "в",
    "из",
    "для",
    "это",
    "этого",
    "который",
    "которые",
    "также",
    "пожалуйста",
}

_cache: dict = {"at": 0.0, "items": []}


def parse_comment(comment: str) -> list[str]:
    """Достать из комментария имена файлов, которые просят исключить."""
    if not comment:
        return []
    match = _TRIGGER_RE.search(comment)
    if not match:
        return []
    tail = match.group(2)
    tail = re.split(r"[\n;!?]|\.(?=\s|$)", tail)[0]
    found: list[str] = []
    for chunk in re.split(r"[,;]| и | также ", tail, flags=re.IGNORECASE):
        name = _CLEAN_RE.sub("", chunk).strip(" .,:;-—")
        name = re.sub(r"\s+(в|из|для|к)\s+ответ\w*$", "", name, flags=re.IGNORECASE).strip()
        if not name or len(name) < 4:
            continue
        if name.lower() in _STOP_WORDS:
            continue
        if not re.search(r"[0-9A-Za-zА-Яа-яЁё]{3}", name):
            continue
        if name.lower() not in [item.lower() for item in found]:
            found.append(name)
    return found


def _load() -> list[dict]:
    now = time.time()
    if _cache["items"] and now - _cache["at"] < config.EXCLUSIONS_CACHE_TTL:
        return _cache["items"]
    items = appdb.list_exclusions()
    _cache.update(at=now, items=items)
    return items


def invalidate() -> None:
    _cache.update(at=0.0, items=[])


def patterns() -> list[str]:
    return [item["pattern"] for item in _load()]


def is_excluded(source: str) -> bool:
    if not source:
        return False
    name = source.strip().lower()
    for pattern in patterns():
        if not pattern:
            continue
        if name == pattern or name.endswith("/" + pattern) or pattern in name:
            return True
    return False


def add_from_comment(
    comment: str,
    author: str = "",
    request_id: int | None = None,
) -> list[dict]:
    """Разобрать комментарий и добавить новые исключения."""
    added: list[dict] = []
    for name in parse_comment(comment):
        if appdb.get_exclusion_by_pattern(name):
            continue
        item = appdb.add_exclusion(
            name,
            origin="comment",
            author=author,
            comment=comment,
            request_id=request_id,
        )
        if item:
            added.append(item)
    if added:
        invalidate()
    return added


def filter_chunks(chunks: list) -> list:
    """Убрать фрагменты из исключённых документов."""
    active = patterns()
    if not active:
        return chunks
    result = []
    for chunk in chunks:
        source = (getattr(chunk, "source", "") or "").lower()
        if any(source == p or source.endswith("/" + p) or p in source for p in active):
            continue
        result.append(chunk)
    return result
