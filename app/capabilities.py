"""Ответ на вопрос «что ты умеешь»: перечень возможностей + подсказки из БД промтов."""

import json
import re

import appdb
import config
import domains
import prompts as prompts_mod

SETTING_KEY = "capabilities"

INTENTS = (
    r"что\s+ты\s+умеешь",
    r"что\s+умеешь",
    r"чем\s+(ты\s+)?можешь\s+помочь",
    r"чем\s+(ты\s+)?помогаешь",
    r"(какие\s+)?(у\s+тебя\s+)?возможности",
    r"как\s+ты\s+(умеешь|работаешь|помогаешь)",
    r"что\s+ты\s+(делаешь|умеешь\s+делать)",
    r"на\s+что\s+ты\s+способен",
    r"твои\s+функции",
    r"функционал",
)

_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INTENTS]

GROUPS: list[tuple[str, list[str]]] = [
    (
        "Текст и работа с информацией",
        [
            "Отвечаю на вопросы и объясняю сложные темы простыми словами",
            "Пишу, редактирую, сокращаю и перевожу тексты",
            "Анализирую документы, статьи, договоры и инструкции",
            "Помогаю с письмами, резюме, отчётами и структурой презентаций",
        ],
    ),
    (
        "База знаний",
        [
            "Ищу ответы по проиндексированной документации продуктов",
            "Показываю источники и ссылки на оригиналы документов",
            "Примеры вопросов подбираю по составу вашего индекса",
        ],
    ),
    (
        "Чего я не делаю",
        [
            "Не выхожу в интернет и не открываю внешние сайты",
            "Не выполняю действия в системах: только текст и знания из базы",
            "Не выдумываю факты: если ответа в документации нет, скажу об этом",
        ],
    ),
]


def is_capabilities_question(question: str) -> bool:
    text = question.strip().lower()
    return any(pattern.search(text) for pattern in _PATTERNS)


def groups() -> list[dict]:
    """Актуальные группы: из активного направления, иначе из БД админа, иначе из кода."""
    domain_groups = domains.capabilities()
    if domain_groups:
        return domain_groups
    raw = appdb.get_setting(SETTING_KEY, "")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return data
        except json.JSONDecodeError:
            pass
    return [{"title": title, "items": list(items)} for title, items in GROUPS]


def source() -> str:
    """Откуда взялся перечень «Что я умею»."""
    if domains.capabilities():
        return f"направление: {domains.current().get('title', '')} (БД направлений)"
    if is_custom():
        return "кабинет администратора"
    return "код: app/capabilities.py, константа GROUPS"


def is_custom() -> bool:
    return bool(appdb.get_setting(SETTING_KEY, ""))


def save_groups(data: list[dict]) -> list[dict]:
    """Сохранить перечень из кабинета администратора."""
    clean: list[dict] = []
    for group in data:
        title = str(group.get("title", "")).strip()
        items = [str(item).strip() for item in group.get("items", []) if str(item).strip()]
        if not title or not items:
            continue
        clean.append({"title": title, "items": items})
    if not clean:
        raise ValueError("Нужен хотя бы один раздел с пунктами")
    appdb.set_setting(SETTING_KEY, json.dumps(clean, ensure_ascii=False))
    return clean


def reset_groups() -> list[dict]:
    appdb.set_setting(SETTING_KEY, "")
    return groups()


def suggestions(limit: int | None = None) -> list[str]:
    items = prompts_mod.get_prompts()
    texts = [item["text"] for item in items]
    if limit is not None:
        return texts[:limit]
    return texts


def render(limit: int | None = None) -> str:
    lines = [f"Я — ассистент по базе знаний {config.APP_TITLE}.", "", "Что я умею:"]
    for group in groups():
        lines.append("")
        lines.append(f"{group['title']}:")
        lines.extend(f"— {item}" for item in group["items"])
    hints = suggestions(limit)
    if hints:
        lines.append("")
        lines.append("Примеры вопросов по вашей базе знаний:")
        lines.extend(f"• {hint}" for hint in hints)
    lines.append("")
    lines.append("Спросите меня о любом продукте — отвечу по документации со ссылками на источники.")
    return "\n".join(lines)
