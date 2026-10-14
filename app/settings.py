"""Настройки приложения, изменяемые из кабинета администратора.

Значения хранятся в таблице `settings` БД app.db. Если параметр там не задан,
берётся значение из переменной окружения (то есть из .env).
"""

import threading
import time
from dataclasses import dataclass, field

import appdb
import config

BOOL_TRUE = ("1", "true", "yes", "on")


@dataclass
class Param:
    key: str
    label: str
    type: str = "str"
    group: str = "general"
    hint: str = ""
    requires_reindex: bool = False
    secret: bool = False
    options: list[str] = field(default_factory=list)

    def default(self) -> str:
        value = getattr(config, self.key, "")
        if self.type == "int":
            return str(int(value)) if str(value).lstrip("-").isdigit() else "0"
        if self.type == "float":
            try:
                return str(float(value))
            except (TypeError, ValueError):
                return "0"
        if self.type == "bool":
            return "1" if value else "0"
        if self.type == "prompt":
            return config.SYSTEM_PROMPT
        return str(value)


GENERATION = "Генерация"
RETRIEVAL = "Поиск и чанкинг"
BEHAVIOUR = "Поведение ответа"
INGEST = "Индексация"

PARAMS: list[Param] = [
    Param("LLM_MODEL", "Модель генерации", group=GENERATION,
          hint="Идентификатор модели у LLM-сервера, например qwen2.5:3b-instruct"),
    Param("LLM_BASE_URL", "Адрес LLM-сервера", group=GENERATION,
          hint="OpenAI-совместимый endpoint, например http://ollama:11434/v1"),
    Param("LLM_API_KEY", "Ключ LLM", group=GENERATION, secret=True,
          hint="Обычно не нужен для локального сервера"),
    Param("LLM_TEMPERATURE", "Температура генерации", "float", group=GENERATION,
          hint="От 0 (строго по тексту) до 1 (свободнее). Для документации 0.0–0.3"),
    Param("LLM_MAX_TOKENS", "Максимум токенов в ответе", "int", group=GENERATION,
          hint="Ограничение длины ответа, 0 — не ограничивать"),
    Param("RETRIEVE_TOP_K", "Фрагментов в ответе", "int", group=RETRIEVAL,
          hint="Сколько фрагментов документации попадает в промпт"),
    Param("CORRECTION_MATCH_THRESHOLD", "Порог подстановки исправления", "float", group=RETRIEVAL,
          hint="Схожесть вопроса с исправлением, выше — отдаём исправленный ответ как есть"),
    Param("CORRECTION_EXAMPLES", "Примеров исправлений в промпте", "int", group=RETRIEVAL,
          hint="Похожие исправления добавляются в промпт как образцы ответа"),
    Param("SYNONYMS_MAX_VARIANTS", "Вариантов запроса по синонимам", "int", group=RETRIEVAL,
          hint="Сколько вариантов запроса строит словарь синонимов"),
    Param("PROMPTS_MAX", "Стартовых промтов в чате", "int", group=BEHAVIOUR,
          hint="Сколько примеров вопросов показывать на главной"),
    Param("SYSTEM_PROMPT", "Системная инструкция", "prompt", group=BEHAVIOUR,
          hint="Как ассистент отвечает: стиль, цитаты, отказ при отсутствии ответа"),
    Param("EMBEDDING_MODEL", "Модель эмбеддингов", group=RETRIEVAL, requires_reindex=True,
          hint="Смена требует полной переиндексации документации"),
    Param("CHUNK_SIZE", "Размер чанка", "int", group=RETRIEVAL, requires_reindex=True),
    Param("CHUNK_OVERLAP", "Перекрытие чанков", "int", group=RETRIEVAL, requires_reindex=True),
    Param("INGEST_WORKERS", "Потоков индексации", "int", group=INGEST,
          hint="Сколько файлов обрабатывается одновременно"),
    Param("INGEST_BATCH_SIZE", "Точек в одном upsert", "int", group=INGEST),
    Param("INGEST_CHECKPOINT_EVERY", "Чанков между срезами", "int", group=INGEST),
]

CACHE_TTL = 3.0

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "values": {}}


def invalidate() -> None:
    with _lock:
        _cache.update(at=0.0, values={})


def _stored() -> dict[str, str]:
    now = time.time()
    with _lock:
        if _cache["values"] and now - _cache["at"] < CACHE_TTL:
            return _cache["values"]
    values: dict[str, str] = {}
    try:
        with appdb.connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            values = {row["key"]: row["value"] for row in rows}
    except Exception:
        values = {}
    with _lock:
        _cache.update(at=now, values=values)
    return values


def raw(key: str) -> str:
    stored = _stored()
    if key in stored:
        return stored[key]
    return ""


def value(key: str, default: str = "") -> str:
    stored = raw(key)
    if stored:
        return stored
    return default


def int_value(key: str) -> int:
    try:
        return int(float(value(key)))
    except (TypeError, ValueError):
        return int(getattr(config, key, 0) or 0)


def float_value(key: str) -> float:
    try:
        return float(value(key))
    except (TypeError, ValueError):
        try:
            return float(getattr(config, key, 0.0))
        except (TypeError, ValueError):
            return 0.0


def bool_value(key: str, default: bool = False) -> bool:
    stored = raw(key)
    if stored:
        return stored.strip().lower() in BOOL_TRUE
    return default


def effective(key: str) -> str:
    """Действующее значение параметра: из БД, иначе из .env."""
    stored = raw(key)
    if stored:
        return stored
    for param in PARAMS:
        if param.key == key:
            return param.default()
    return ""


def describe() -> list[dict]:
    items = []
    for param in PARAMS:
        items.append(
            {
                "key": param.key,
                "label": param.label,
                "type": param.type,
                "group": param.group,
                "hint": param.hint,
                "secret": param.secret,
                "requires_reindex": param.requires_reindex,
                "value": effective(param.key),
                "default": param.default(),
                "overridden": bool(raw(param.key)),
            }
        )
    return items


def _validate(param: Param, value: str) -> str:
    text = (value or "").strip()
    if param.type == "int":
        try:
            number = int(float(text))
        except ValueError as exc:
            raise ValueError(f"{param.label}: нужно целое число") from exc
        return str(number)
    if param.type == "float":
        try:
            number = float(text.replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"{param.label}: нужно число") from exc
        return str(number)
    if param.type == "prompt":
        if not text:
            raise ValueError(f"{param.label}: текст не может быть пустым")
        return text
    if param.type == "bool":
        return "1" if text.lower() in BOOL_TRUE else "0"
    if param.secret and not text:
        raise ValueError(f"{param.label}: значение не может быть пустым")
    if param.options and text not in param.options:
        raise ValueError(f"{param.label}: допустимые значения — {', '.join(param.options)}")
    return text


def save(values: dict[str, str]) -> list[str]:
    """Сохранить значения; возвращает ключи, требующие переиндексации."""
    changed: list[str] = []
    reindex: list[str] = []
    by_key = {param.key: param for param in PARAMS}
    for key, raw_value in values.items():
        param = by_key.get(key)
        if param is None:
            raise ValueError(f"Неизвестный параметр: {key}")
        text = _validate(param, str(raw_value))
        if effective(key) == text:
            continue
        appdb.set_setting(key, text)
        changed.append(key)
        if param.requires_reindex:
            reindex.append(key)
    invalidate()
    return reindex


def reset(key: str) -> None:
    appdb.set_setting(key, "")
    invalidate()


def groups() -> list[str]:
    order = [GENERATION, RETRIEVAL, BEHAVIOUR, INGEST]
    present = {param.group for param in PARAMS}
    return [group for group in order if group in present]