"""Настройки приложения, изменяемые из кабинета администратора.

Значения хранятся в таблице `settings` БД app.db. Если параметр там не задан,
берётся значение из переменной окружения (то есть из .env).
"""

import threading
import time
from dataclasses import dataclass, field

import appdb
import config
import settingsdb

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
    editable: bool = True
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
WEBSITES = "Сайты и обход"
LESSONS = "Обучение"
HYBRID = "Гибридный поиск"
RERANK = "Переранжирование"
UPLOADS = "Загрузка документов"
AUTOLEARN = "Автообучение"
INFRA = "Инфраструктура (только чтение)"

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
    Param("CORRECTION_OVERLAP_WORDS", "Общих слов для точного исправления", "int", group=RETRIEVAL,
          hint="Сколько значимых слов должно совпадать в вопросе и в исправлении, "
               "иначе оно идёт только как пример. 0 — проверка отключена"),
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
    Param("E5_QUERY_PREFIX", "Префикс запроса e5", group=RETRIEVAL, requires_reindex=True,
          hint="Для multilingual-e5-base обязательно 'query: '. Менять только вместе с моделью эмбеддингов"),
    Param("E5_PASSAGE_PREFIX", "Префикс документа e5", group=RETRIEVAL, requires_reindex=True,
          hint="Для multilingual-e5-base обязательно 'passage: '"),
    Param("EMBED_BATCH", "Батч эмбеддингов", "int", group=RETRIEVAL,
          hint="Сколько фрагментов считать эмбеддингами за раз при индексации"),
    Param("EMBEDDER_DEVICE", "Устройство модели эмбеддингов", group=RETRIEVAL,
          options=["auto", "cpu", "cuda"],
          hint="auto — GPU при наличии CUDA, иначе CPU. Смена модели эмбеддингов требует переиндексации"),
    Param("HYBRID_SPARSE", "Гибридный поиск (dense + BM25)", "bool", group=HYBRID,
          hint="Включать точный поиск BM25 вместе с векторным. Требует индексации со sparse-векторами"),
    Param("SPARSE_MODEL", "BM25-модель", group=HYBRID, requires_reindex=True,
          hint="Модель fastembed для точного поиска по терминам и названиям продуктов"),
    Param("HYBRID_CANDIDATES", "Кандидатов от каждого канала", "int", group=HYBRID,
          hint="Ширина пула до переранжирования: 15–30"),
    Param("HYBRID_FUSION", "Способ слияния каналов", group=HYBRID, options=["rrf", "dense"],
          hint="rrf — Reciprocal Rank Fusion (рекомендуется), dense — только векторный канал"),
    Param("RERANKER_ENABLED", "Переранжирование ответа", "bool", group=RERANK,
          hint="Cross-encoder пересчитывает порядок фрагментов перед формированием ответа"),
    Param("RERANKER_MODEL", "Модель reranker'а", group=RERANK,
          hint="Мультиязычная модель пары «запрос-фрагмент», например BAAI/bge-reranker-v2-m3"),
    Param("RERANKER_CANDIDATES", "Кандидатов на переранжирование", "int", group=RERANK,
          hint="Сколько фрагментов оценивает модель: 10–30"),
    Param("RERANKER_TOP_K", "Фрагментов после reranker'а", "int", group=RERANK,
          hint="Сколько лучших фрагментов уходит в промпт"),
    Param("RERANKER_MAX_LENGTH", "Максимум токенов на пару", "int", group=RERANK,
          hint="Ограничение длины фрагмента для оценки: меньше — быстрее"),
    Param("RERANKER_BATCH", "Батч reranker'а", "int", group=RERANK,
          hint="Сколько пар оценивается за раз"),
    Param("RERANKER_DEVICE", "Устройство reranker'а", group=RERANK,
          options=["auto", "cpu", "cuda"],
          hint="auto — GPU при наличии CUDA, иначе CPU"),
    Param("SSE_HEARTBEAT", "Пауза между ping в чате, сек", "int", group=GENERATION,
          hint="Служебная строка в потоке ответа, чтобы соединение не рвалось"),
    Param("UPLOAD_MAX_MB", "Максимум МБ для файла", "int", group=UPLOADS,
          hint="Ограничение для загрузки с ПК и скачивания по ссылке"),
    Param("UPLOAD_URL_TIMEOUT", "Таймаут скачивания, сек", "int", group=UPLOADS),
    Param("CHUNK_SIZE", "Размер чанка", "int", group=RETRIEVAL, requires_reindex=True),
    Param("CHUNK_OVERLAP", "Перекрытие чанков", "int", group=RETRIEVAL, requires_reindex=True),
    Param("INGEST_WORKERS", "Потоков индексации", "int", group=INGEST,
          hint="Сколько файлов обрабатывается одновременно"),
    Param("INGEST_BATCH_SIZE", "Точек в одном upsert", "int", group=INGEST),
    Param("INGEST_CHECKPOINT_EVERY", "Чанков между срезами", "int", group=INGEST),
    Param("LESSONS_MAX", "Уроков в промпте ответа", "int", group=LESSONS,
          hint="Сколько релевантных уроков администратора подмешивается в ответ"),
    Param("LESSONS_MIN_SCORE", "Порог релевантности урока", "float", group=LESSONS,
          hint="Ниже порога урок в ответ не попадает"),
    Param("LESSONS_COLLECTION", "Коллекция уроков", group=LESSONS),
    Param("TRAINING_FROM_CORRECTIONS", "Обучать на исправлениях автоматически", "bool", group=LESSONS,
          hint="Каждое исправление администратора сразу превращается в урок"),
    Param("WEB_CRAWL_MAX_PAGES", "Максимум страниц с одного сайта", "int", group=WEBSITES,
          hint="Предохранитель: обход остановится, когда лимит исчерпан"),
    Param("WEB_CRAWL_TIMEOUT", "Таймаут запроса, сек", "int", group=WEBSITES),
    Param("WEB_CRAWL_DELAY_MS", "Пауза между запросами, мс", "int", group=WEBSITES,
          hint="Не нагружайте сайт: рекомендуется 500–1000 мс"),
    Param("WEB_CRAWL_RESPECT_ROBOTS", "Уважать robots.txt", "bool", group=WEBSITES,
          hint="Отключать только для доверенных сайтов"),
    Param("WEB_CRAWL_USER_AGENT", "User-Agent обхода", group=WEBSITES),
    Param("AUTOLEARN_WINDOW", "Запросов в окне анализа", "int", group=AUTOLEARN,
          hint="Сколько последних запросов пользователей оценивается автоматически"),
    Param("AUTOLEARN_DAYS", "Окно анализа, дней", "int", group=AUTOLEARN,
          hint="Ограничить анализ последними днями, 0 — без ограничения по дате"),
    Param("AUTOLEARN_LLM_ENABLED", "Оценивать ответы моделью", "bool", group=AUTOLEARN,
          hint="LLM-судья уточняет вердикт и предлагает правильный ответ — точнее, но медленнее"),
    Param("AUTOLEARN_LLM_LIMIT", "Запросов для LLM-оценки", "int", group=AUTOLEARN,
          hint="Сколько запросов за один запуск отправлять на LLM-оценку"),
    Param("AUTOLEARN_TOPIC_THRESHOLD", "Порог схожести вопросов в теме", "float", group=AUTOLEARN,
          hint="Выше — темы точнее, но их больше. Для e5 обычно 0.82–0.88"),
    Param("AUTOLEARN_MIN_TOPIC_HITS", "Проблемных вопросов для урока", "int", group=AUTOLEARN,
          hint="Сколько проблемных обращений в теме нужно, чтобы собрать урок-кандидат"),
    Param("AUTOLEARN_MAX_TOPICS", "Максимум тем", "int", group=AUTOLEARN),
    Param("QUESTIONGEN_COUNT", "Вопросов для генерации", "int", group=AUTOLEARN,
          hint="Сколько проверочных вопросов генерировать и прогонять за один запуск"),
    Param("QUESTIONGEN_SOURCES", "Источники вопросов", group=AUTOLEARN,
          hint="Список через запятую: docs — из документации, topics — из тем, domain — из сферы деятельности"),
    Param("QUESTIONGEN_JUDGE", "Оценивать генерацию моделью", "bool", group=AUTOLEARN,
          hint="После прогона проверить ответы LLM-судьёй и предложить эталоны"),
    Param("QUESTIONGEN_VARIANTS", "Вариантов вопроса на фрагмент", "int", group=AUTOLEARN,
          hint="Сколько переформулировок одного факта из фрагмента просит модель (разные наклонения и формы)"),
    Param("QUESTIONGEN_SIM_THRESHOLD", "Порог схожести вопросов", "float", group=AUTOLEARN,
          hint="Выше — группы точнее, но их больше. Для e5 обычно 0.82–0.88"),
    Param("QUESTIONGEN_DEDUP_THRESHOLD", "Порог дубликата вопроса", "float", group=AUTOLEARN,
          hint="Выше этого порога вопрос помечается почти-дублем уже заданного"),
    Param("QUESTIONGEN_SIM_LIMIT", "Вопросов для анализа схожести", "int", group=AUTOLEARN,
          hint="Сколько последних запомненных вопросов сравнивать за один анализ"),
    Param("BOOTSTRAP_CYCLES", "Циклов первичного обучения", "int", group=AUTOLEARN,
          hint="Сколько раз прогонять вопросы сферы при первом обучении после загрузки документов"),
    Param("BOOTSTRAP_COUNT", "Вопросов в цикле первичного обучения", "int", group=AUTOLEARN,
          hint="Сколько вопросов сферы прогонять за цикл первичного обучения"),
    Param("BOOTSTRAP_JUDGE", "Оценивать первичное обучение моделью", "bool", group=AUTOLEARN,
          hint="Подключать LLM-судью и сбор уроков-кандидатов при первичном обучении"),
    Param("CORRECTIONS_COLLECTION", "Коллекция исправлений", group=RETRIEVAL,
          hint="Имя коллекции Qdrant для исправлений ответов"),
    Param("SYNONYMS_CACHE_TTL", "Кэш словаря синонимов, сек", "int", group=RETRIEVAL),
    Param("SYNONYMS_SEED_ON_START", "Заполнять словарь при старте", "bool", group=RETRIEVAL,
          hint="Добавлять стартовый набор синонимов, если словарь пуст"),
    Param("PROMPTS_CACHE_TTL", "Кэш стартовых промтов, сек", "int", group=BEHAVIOUR),
    Param("DOCS_BASE_URL", "Базовый адрес документации", group=WEBSITES,
          hint="Префикс ссылок на документы в ответах, если ссылка не задана отдельно"),
    Param("APP_DB", "База данных приложения", group=INFRA, editable=False),
    Param("SETTINGS_DB", "База данных настроек", group=INFRA, editable=False),
    Param("QDRANT_URL", "Адрес Qdrant", group=INFRA, editable=False),
    Param("QDRANT_COLLECTION", "Базовая коллекция Qdrant", group=INFRA, editable=False),
    Param("QDRANT_API_KEY", "Ключ Qdrant", group=INFRA, secret=True, editable=False),
    Param("INDEX_STATE_DB", "База состояния индексации", group=INFRA, editable=False),
    Param("DOCS_DIR", "Каталог документации", group=INFRA, editable=False),
    Param("UPLOADS_DIR", "Каталог загрузок", group=INFRA, editable=False),
    Param("EVAL_DIR", "Каталог оценок", group=INFRA, editable=False),
    Param("PROMPTS_DB", "База стартовых промтов", group=INFRA, editable=False),
    Param("SYNONYMS_DB", "База синонимов", group=INFRA, editable=False),
    Param("ADMIN_USERNAME", "Логин администратора по умолчанию", group=INFRA, editable=False),
    Param("ADMIN_PASSWORD", "Пароль администратора по умолчанию", group=INFRA, secret=True, editable=False),
    Param("DOMAIN_PROFILE", "Профиль сферы по умолчанию", group=INFRA, editable=False),
    Param("EMBED_DIM", "Размерность эмбеддингов", "int", group=INFRA, editable=False),
]

CACHE_TTL = 3.0

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "values": {}}
_domain_cache: dict = {"at": 0.0, "values": {}}


def _domain_values() -> dict[str, str]:
    """Настройки активной сферы деятельности: имеют приоритет над локальными."""
    now = time.time()
    with _lock:
        if _domain_cache["values"] and now - _domain_cache["at"] < CACHE_TTL:
            return _domain_cache["values"]
    values: dict[str, str] = {}
    try:
        import domains

        values = domains.active_settings()
    except Exception:
        values = {}
    with _lock:
        _domain_cache.update(at=now, values=values)
    return values


def invalidate() -> None:
    with _lock:
        _cache.update(at=0.0, values={})
        _domain_cache.update(at=0.0, values={})


def _stored() -> dict[str, str]:
    now = time.time()
    with _lock:
        if _cache["values"] and now - _cache["at"] < CACHE_TTL:
            return _cache["values"]
    values: dict[str, str] = {}
    try:
        values = settingsdb.all()
    except Exception:
        values = {}
    with _lock:
        _cache.update(at=now, values=values)
    return values


def raw(key: str) -> str:
    domain = _domain_values()
    if key in domain:
        return domain[key]
    stored = _stored()
    if key in stored:
        return stored[key]
    return ""


def local_raw(key: str) -> str:
    """Только локальное значение (из БД настроек), без настроек сферы."""
    stored = _stored()
    return stored.get(key, "")


def value(key: str, default: str = "") -> str:
    stored = raw(key)
    if stored:
        return stored
    if default:
        return default
    if key == "SYSTEM_PROMPT":
        return config.SYSTEM_PROMPT
    if hasattr(config, key):
        return str(getattr(config, key))
    return ""


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
    """Действующее значение параметра: сфера, локальное значение, иначе .env."""
    value = raw(key)
    if value:
        return value
    for param in PARAMS:
        if param.key == key:
            return param.default()
    return ""


def source(key: str) -> str:
    """Источник действующего значения: domain, local или default."""
    if key in _domain_values():
        return "domain"
    if key in _stored():
        return "local"
    return "default"


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
                "editable": param.editable,
                "requires_reindex": param.requires_reindex,
                "value": effective(param.key),
                "default": param.default(),
                "local": local_raw(param.key),
                "source": source(param.key),
                "overridden": bool(raw(param.key)),
            }
        )
    return items


def _validate(param: Param, value: str) -> str:
    raw = value if value is not None else ""
    if param.key in PREFIX_KEYS:
        return raw
    text = raw.strip()
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


def _as_int(text: str, fallback: int = 0) -> int:
    try:
        return int(float(str(text)))
    except (TypeError, ValueError):
        return fallback


def _as_float(text: str, fallback: float = 0.0) -> float:
    try:
        return float(str(text).replace(",", "."))
    except (TypeError, ValueError):
        return fallback


RANGES = {
    "LLM_TEMPERATURE": (0.0, 2.0),
    "CORRECTION_MATCH_THRESHOLD": (0.0, 1.0),
    "AUTOLEARN_TOPIC_THRESHOLD": (0.0, 1.0),
    "QUESTIONGEN_SIM_THRESHOLD": (0.0, 1.0),
    "QUESTIONGEN_DEDUP_THRESHOLD": (0.0, 1.0),
    "LESSONS_MIN_SCORE": (0.0, 1.0),
}


def consistency(prospective: dict[str, str]) -> tuple[list[str], list[str]]:
    """Перекрёстная проверка значений: (ошибки, предупреждения)."""
    errors: list[str] = []
    warnings: list[str] = []

    def num(key: str) -> float:
        return _as_float(prospective.get(key, effective(key)))

    def flag(key: str) -> bool:
        text = prospective.get(key, effective(key))
        return str(text).strip().lower() in BOOL_TRUE

    for key, (low, high) in RANGES.items():
        value = num(key)
        if value < low or value > high:
            warnings.append(f"{key}: значение {value} вне диапазона {low}–{high}")

    size, overlap = _as_int(prospective.get("CHUNK_SIZE", effective("CHUNK_SIZE"))), _as_int(
        prospective.get("CHUNK_OVERLAP", effective("CHUNK_OVERLAP"))
    )
    if overlap >= size:
        errors.append("Перекрытие чанков должно быть меньше размера чанка")

    top_k = _as_int(prospective.get("RETRIEVE_TOP_K", effective("RETRIEVE_TOP_K")))
    candidates = _as_int(prospective.get("HYBRID_CANDIDATES", effective("HYBRID_CANDIDATES")))
    if top_k > candidates:
        warnings.append(
            "Фрагментов в ответе больше, чем кандидатов гибридного поиска: часть не находится"
        )
    if flag("RERANKER_ENABLED"):
        if not str(prospective.get("RERANKER_MODEL", effective("RERANKER_MODEL"))).strip():
            errors.append("Переранжирование включено, но модель reranker'а не задана")
        rerank_cand = _as_int(prospective.get("RERANKER_CANDIDATES", effective("RERANKER_CANDIDATES")))
        rerank_top = _as_int(prospective.get("RERANKER_TOP_K", effective("RERANKER_TOP_K")))
        if rerank_top > rerank_cand:
            warnings.append("Фрагментов после reranker'а больше, чем кандидатов на переранжирование")
        if rerank_cand > candidates:
            warnings.append("Кандидатов reranker'а больше, чем кандидатов поиска")
    if flag("HYBRID_SPARSE") and not str(
        prospective.get("SPARSE_MODEL", effective("SPARSE_MODEL"))
    ).strip():
        errors.append("Гибридный поиск включён, но BM25-модель не задана")

    sources = str(prospective.get("QUESTIONGEN_SOURCES", effective("QUESTIONGEN_SOURCES")))
    allowed = {"docs", "topics", "domain"}
    unknown = {part.strip() for part in sources.split(",") if part.strip()} - allowed
    if unknown:
        errors.append(
            "Источники вопросов могут быть только: docs, topics, domain (лишние: "
            + ", ".join(sorted(unknown))
            + ")"
        )

    model = str(prospective.get("EMBEDDING_MODEL", effective("EMBEDDING_MODEL")))
    if "e5" in model.lower():
        query_prefix = str(prospective.get("E5_QUERY_PREFIX", effective("E5_QUERY_PREFIX")))
        passage_prefix = str(prospective.get("E5_PASSAGE_PREFIX", effective("E5_PASSAGE_PREFIX")))
        if query_prefix != "query: ":
            warnings.append("Для модели e5 префикс запроса должен быть «query: » (с пробелом)")
        if passage_prefix != "passage: ":
            warnings.append("Для модели e5 префикс документа должен быть «passage: » (с пробелом)")
    return errors, warnings


def save(values: dict[str, str]) -> dict:
    """Сохранить значения; возвращает ключи для переиндексации и предупреждения."""
    changed: list[str] = []
    reindex: list[str] = []
    by_key = {param.key: param for param in PARAMS}
    prospective: dict[str, str] = {}
    for key, raw_value in values.items():
        param = by_key.get(key)
        if param is None:
            raise ValueError(f"Неизвестный параметр: {key}")
        if not param.editable:
            raise ValueError(f"{param.label}: параметр доступен только для чтения")
        text = _validate(param, str(raw_value))
        prospective[key] = text
    errors, warnings = consistency(prospective)
    if errors:
        raise ValueError("; ".join(errors))
    for key, text in prospective.items():
        param = by_key[key]
        if effective(key) == text:
            continue
        appdb.set_setting(key, text)
        changed.append(key)
        if param.requires_reindex:
            reindex.append(key)
    invalidate()
    if changed:
        _reload_models()
    return {"reindex": reindex, "warnings": warnings, "changed": changed}


MODEL_KEYS = (
    "SPARSE_MODEL",
    "RERANKER_MODEL",
    "RERANKER_MAX_LENGTH",
    "EMBEDDING_MODEL",
    "EMBEDDER_DEVICE",
    "RERANKER_DEVICE",
)
PREFIX_KEYS = ("E5_QUERY_PREFIX", "E5_PASSAGE_PREFIX")


def _reload_models() -> None:
    """Сбросить кеш загруженных моделей, если сменились модели поиска."""
    try:
        import vectorstore as vs
    except Exception:
        return
    vs.get_sparse_model.cache_clear()
    vs.get_reranker.cache_clear()
    vs.get_embedder.cache_clear()


def reset(key: str) -> None:
    appdb.set_setting(key, "")
    invalidate()
    if key in MODEL_KEYS:
        _reload_models()


def groups() -> list[str]:
    order = [
        GENERATION,
        RETRIEVAL,
        HYBRID,
        RERANK,
        BEHAVIOUR,
        LESSONS,
        WEBSITES,
        UPLOADS,
        AUTOLEARN,
        INGEST,
        INFRA,
    ]
    present = {param.group for param in PARAMS}
    return [group for group in order if group in present]