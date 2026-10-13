"""Ответ на вопрос «что ты умеешь»: перечень возможностей + подсказки из БД промтов."""

import re

import prompts as prompts_mod

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
        "База знаний company",
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


def suggestions(limit: int | None = None) -> list[str]:
    items = prompts_mod.get_prompts()
    texts = [item["text"] for item in items]
    if limit is not None:
        return texts[:limit]
    return texts


def render(limit: int | None = None) -> str:
    lines = ["Я — ассистент по базе знаний company.", "", "Что я умею:"]
    for title, items in GROUPS:
        lines.append("")
        lines.append(f"{title}:")
        lines.extend(f"— {item}" for item in items)
    hints = suggestions(limit)
    if hints:
        lines.append("")
        lines.append("Примеры вопросов по вашей базе знаний:")
        lines.extend(f"• {hint}" for hint in hints)
    lines.append("")
    lines.append("Спросите меня о любом продукте — отвечу по документации со ссылками на источники.")
    return "\n".join(lines)
