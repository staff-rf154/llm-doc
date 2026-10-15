"""Исключения из ответов: документы, разделы, изображения и привязки к вопросу.

Виды правил (scope):
- `document` — весь документ не показывается нигде;
- `section` — каталог или часть имени файла (например, `AnalyticConsole_help`);
- `image` — конкретная картинка из документа (файл изображения или «документ, с. N»);
- `question` — документ исключается только для похожих вопросов.

Правила появляются из комментариев к оценке («исключить из ответа …») и заполняются
вручную в кабинете администра��ора.
"""

import re
import time

import appdb
import config
import vectorstore as vs

SCOPES = ("document", "section", "image", "question")

SCOPE_LABELS = {
    "document": "Документ",
    "section": "Раздел (папка) документа",
    "image": "Изображение",
    "question": "Для вопроса — документ",
}

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

_EXT_RE = re.compile(r"\.(?=[A-Za-zа-яА-Я]{1,3}(?=[\s,;]|$))")
_ABBR_RE = re.compile(r"\b([A-Za-zа-яА-Я]{1,3})\.(?=\s)")
_SENTENCE_RE = re.compile(r"[\n;!?]|\.(?=\s|$)")
_EXT_MARK = "\u0000"

_IMAGE_RE = re.compile(
    r"^(.+?)[,\s]+с(?:траниц[аы]|\.)?\.??\s*(\d+)", re.IGNORECASE
)
_QUESTION_RE = re.compile(r"(для\s+вопроса\s+|в\s+вопросе\s+)(.+)$", re.IGNORECASE)

_CLEAN_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._\-/ ]")
_STOP_WORDS = {
    "документ",
    "документы",
    "файл",
    "файлы",
    "картинка",
    "картинки",
    "изображение",
    "изображения",
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


def detect_scope(name: str) -> str:
    """Каталог (раздел) это или отдельный документ."""
    if "/" in name or "\\" in name:
        return "section"
    target = name.lower()
    try:
        for source in vs.known_sources(limit=3000):
            parent = source.rsplit("/", 1)[0] if "/" in source else ""
            if parent and parent.lower() == target:
                return "section"
            if source.lower() == target:
                return "document"
    except Exception:
        pass
    return "document"


def parse_comment(comment: str) -> list[dict]:
    """Разобрать комментарий в правила исключений."""
    if not comment:
        return []
    match = _TRIGGER_RE.search(comment)
    if not match:
        return []
    full_tail = match.group(2)
    page_hint = None
    page_match = _IMAGE_RE.search(full_tail.split(",")[0])
    if page_match and re.search(r"^(картинк|изображен|скриншот)", full_tail, re.IGNORECASE):
        page_hint = int(page_match.group(2))
    marked = _EXT_RE.sub(_EXT_MARK, full_tail)
    marked = _ABBR_RE.sub(lambda m: m.group(1) + _EXT_MARK, marked)
    tail = _SENTENCE_RE.split(marked)[0].replace(_EXT_MARK, ".")

    question = ""
    tail = tail.strip(" .,:;-—")
    question_match = _QUESTION_RE.search(tail)
    if question_match:
        question = question_match.group(2).strip(" .,:;-—")
        tail = tail[: question_match.start()].strip(" ,;.")

    rules: list[dict] = []
    for chunk in re.split(r"[,;]| и | также ", tail, flags=re.IGNORECASE):
        chunk = chunk.strip(" .,:;-—")
        chunk = re.sub(r"^(из|в|от|для)\s+", "", chunk, flags=re.IGNORECASE).strip()
        page_only = re.match(r"^(страниц\w*|с\.?)\s*(\d+)\s*$", chunk, re.IGNORECASE)
        if page_only:
            if rules and rules[-1]["scope"] == "image":
                rules[-1]["page"] = int(page_only.group(2))
            continue
        if not chunk or re.match(r"^(страниц\w*|с\.?)\s*\d*\s*$", chunk, re.IGNORECASE):
            continue
        chunk = re.sub(r"[\s,]+(?:страниц\w*|с)\s*\d*$", "", chunk, flags=re.IGNORECASE).strip()
        chunk = chunk.strip(" .,:;-—")
        if not chunk:
            continue

        scope_hint = ""
        hint_match = re.match(
            r"^(картинк[аиуы]|изображен\w+|скриншот\w*|вопрос|раздел)\s+(.+)$",
            chunk,
            flags=re.IGNORECASE,
        )
        if hint_match:
            scope_hint = hint_match.group(1).lower()
            chunk = re.sub(r"^(из|в|от|для)\s+", "", hint_match.group(2).strip(), flags=re.IGNORECASE).strip(" .,:;-—")

        image_match = _IMAGE_RE.match(chunk)
        if image_match or scope_hint.startswith(("картин", "изображен", "скриншот")):
            page = int(image_match.group(2)) if image_match else page_hint
            document = (image_match.group(1) if image_match else chunk).strip()
            rules.append(
                {
                    "scope": "image",
                    "pattern": document,
                    "document": document,
                    "question": "",
                    "page": page,
                }
            )
            continue

        name = _CLEAN_RE.sub("", chunk).strip(" .,:;-—")
        name = re.sub(r"\s+(в|из|для|к)\s+ответ\w*$", "", name, flags=re.IGNORECASE).strip()
        if not name or len(name) < 4 or name.lower() in _STOP_WORDS:
            continue
        if not re.search(r"[0-9A-Za-zА-Яа-яЁё]{3}", name):
            continue

        if scope_hint.startswith("раздел"):
            detected = "section"
        else:
            detected = detect_scope(name)
        rules.append(
            {"scope": detected, "pattern": name, "document": name, "question": "", "page": None}
        )

    if question and rules and rules[-1]["scope"] in ("document", "section"):
        rules[-1]["scope"] = "question"
        rules[-1]["question"] = question

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for rule in rules:
        key = (rule["scope"], rule["pattern"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return deduped


def _load() -> list[dict]:
    now = time.time()
    if _cache["items"] and now - _cache["at"] < config.EXCLUSIONS_CACHE_TTL:
        return _cache["items"]
    items = appdb.list_exclusions()
    _cache.update(at=now, items=items)
    return items


def invalidate() -> None:
    _cache.update(at=0.0, items=[])


def rules() -> list[dict]:
    return list(_load())


def _matches(source: str, pattern: str) -> bool:
    name = (source or "").lower()
    pattern = (pattern or "").lower()
    if not name or not pattern:
        return False
    return name == pattern or name.endswith("/" + pattern) or pattern in name


def is_excluded(source: str) -> bool:
    """Документ исключён целиком или как раздел."""
    if not config.EXCLUSIONS_ENABLED:
        return False
    for rule in rules():
        if rule["scope"] in ("document", "section") and _matches(source, rule["pattern"]):
            return True
    return False


def _similar(question: str, pattern: str) -> bool:
    """Похож ли вопрос на текст правила: вхождение либо близость эмбеддингов."""
    question = (question or "").lower().strip()
    pattern = (pattern or "").lower().strip()
    if not question or not pattern:
        return False
    if pattern in question or question in pattern:
        return True
    words = {word for word in re.findall(r"[0-9a-zа-яё]{4,}", pattern) if word not in _STOP_WORDS}
    if not words:
        return False
    hits = sum(1 for word in words if word in question)
    if hits / len(words) >= 0.6:
        return True
    try:
        first = vs.embed_query(question)
        second = vs.embed_query(pattern)
        score = sum(a * b for a, b in zip(first, second))
        return score >= config.EXCLUSION_QUESTION_THRESHOLD
    except Exception:
        return False


def documents_for_question(question: str) -> set[str]:
    """Документы, исключённые для этого вопроса."""
    if not config.EXCLUSIONS_ENABLED:
        return set()
    blocked: set[str] = set()
    for rule in rules():
        if rule["scope"] != "question":
            continue
        document = rule["document"] or rule["pattern"]
        reference = rule["question"] or rule["pattern"]
        if _similar(question, reference):
            blocked.add(document.lower())
    return blocked


def filter_chunks(chunks: list, question: str = "") -> list:
    """Убрать фрагменты из исключённых документов и разделов."""
    if not config.EXCLUSIONS_ENABLED:
        return chunks
    blocked = documents_for_question(question) if question else set()
    result = []
    for chunk in chunks:
        source = getattr(chunk, "source", "") or ""
        if blocked and any(_matches(source, doc) for doc in blocked):
            continue
        if is_excluded(source):
            continue
        result.append(chunk)
    return result


def excluded_images(available: list[dict]) -> list[dict]:
    """Отфильтровать найденные картинки по правилам вида image/document/section."""
    if not config.EXCLUSIONS_ENABLED:
        return available
    image_rules = [rule for rule in rules() if rule["scope"] == "image"]
    result = []
    for item in available:
        source = (item.get("source") or "").lower()
        file_name = (item.get("file") or "").lower()
        blocked = False
        if is_excluded(source):
            blocked = True
        for rule in image_rules:
            pattern = (rule["document"] or rule["pattern"]).lower()
            if not pattern:
                continue
            page = item.get("page")
            rule_page = rule.get("page")
            if _matches(source, pattern) and rule_page and int(page or 0) == int(rule_page):
                blocked = True
            if pattern and pattern in file_name:
                blocked = True
        if not blocked:
            result.append(item)
    return result