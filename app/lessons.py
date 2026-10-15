"""Обучение ассистента: уроки администратора.

Админ обсуждает с моделью, как правильно отвечать на вопрос: на что обращать
внимание, какие формулировки верны, как дополнить конкретный ответ. Итогом
диалога становится урок — тема, эталонный ответ, ключевые моменты. Уроки
индексируются в отдельной коллекции Qdrant и подмешиваются в промпт ко всем
пользователям вместе с примерами исправлений.
"""

import threading
import time

import appdb
import config
import rag
import vectorstore as vs

_COLLECTION = None

TRAINER_SYSTEM = (
    "Ты — наставник администратора базы знаний и одновременно её разработчик. "
    "Администратор приносит вопросы пользователей и ответы ассистента, которые его не "
    "устроили, либо разбирает, как правильно отвечать.\n\n"
    "Твои задачи в этом диалоге:\n"
    "1. Уточнить, в чём именно ошибка ответа: неверный факт, не та документация, "
    "неполные шаги, неясная формулировка или отсутствие ссылки на источник.\n"
    "2. Подсказать, на какие моменты обращать внимание при ответах такого типа "
    "(условия применения, версии, платформы, место настройки, поведение при ошибке).\n"
    "3. Дать конкретный улучшенный ответ на вопрос пользователя — в той же форме, "
    "в какой его должен выдать ассистент, со ссылкой на документ и страницы, если знаешь их.\n"
    "4. Если данных не хватает, прямо скажи, чего не хватает, и как это проверить в документации.\n\n"
    "Отвечай по-русски, кратко и по делу, структурированно: «Ошибка», «На что обратить внимание», "
    "«Как ответить». Не повторяй общие советы и не выдумывай факты: если не знаешь, "
    "скажи, что нужно открыть документацию."
)

DRAFT_PROMPT = (
    "На основе диалога сформируй урок для базы знаний.\n"
    "Верни строго JSON без пояснений, с полями:\n"
    '  "topic" — короткая тема урока, до 80 символов;\n'
    '  "question" — типичный вопрос пользователя, на который урок распространяется;\n'
    '  "answer" — эталонный ответ, который должен давать ассистент, без ссылок на диалог;\n'
    '  "focus" — ключевые моменты, на которые обязательно обращать внимание, через "; ".\n'
    "Если данных недостаточно для полноценного урока, верни {\"topic\": \"\"}."
)


def collection() -> str:
    return config.LESSONS_COLLECTION


def ensure_collection() -> None:
    client = vs.get_qdrant()
    name = collection()
    if not client.collection_exists(name):
        from qdrant_client.http import models

        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=config.EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )


def reindex() -> int:
    """Синхронизировать активные уроки с векторным индексом."""
    from qdrant_client.http import models

    client = vs.get_qdrant()
    ensure_collection()
    name = collection()

    stored: dict[int, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            stored[int(point.id)] = dict(point.payload or {})
        if offset is None:
            break

    lessons = [item for item in appdb.list_lessons() if item["status"] == "active"]
    keep = {int(item["id"]) for item in lessons}
    stale = [pid for pid in stored if pid not in keep]
    if stale:
        client.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=stale),
        )

    by_id = {int(item["id"]): item for item in lessons}
    for lesson_id, payload in stored.items():
        lesson = by_id.get(lesson_id)
        if not lesson:
            continue
        expected = _payload(lesson)
        if payload != expected:
            client.upsert(
                collection_name=name,
                points=[
                    models.PointStruct(
                        id=lesson_id,
                        vector=vs.embed_query(_search_text(lesson)),
                        payload=expected,
                    )
                ],
                wait=True,
            )

    missing = [item for item in lessons if int(item["id"]) not in stored]
    for lesson in missing:
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=int(lesson["id"]),
                    vector=vs.embed_query(_search_text(lesson)),
                    payload=_payload(lesson),
                )
            ],
            wait=True,
        )
    return len(lessons)


def _payload(lesson: dict) -> dict:
    return {
        "lesson_id": int(lesson["id"]),
        "topic": lesson["topic"],
        "question": lesson["question"],
        "answer": lesson["answer"],
        "focus": lesson["focus"],
        "author": lesson["author"],
    }


def _search_text(lesson: dict) -> str:
    return " ".join(
        part for part in (lesson.get("topic"), lesson.get("question"), lesson.get("focus")) if part
    ) or lesson.get("answer", "")


def save_lesson(
    topic: str,
    question: str,
    answer: str,
    focus: str,
    author: str,
) -> dict:
    lesson = appdb.create_lesson(topic, question, answer, focus, author)
    reindex()
    return lesson


def relevant(question: str, limit: int | None = None) -> list[dict]:
    """Уроки, релевантные вопросу, — в порядке убывания схожести."""
    client = vs.get_qdrant()
    name = collection()
    if not client.collection_exists(name):
        return []
    limit = limit or config.LESSONS_MAX
    try:
        points = client.query_points(
            collection_name=name,
            query=vs.embed_query(question),
            limit=limit,
            with_payload=True,
        ).points
    except Exception:
        return []
    result = []
    for point in points:
        score = float(point.score)
        if score < config.LESSONS_MIN_SCORE:
            continue
        payload = point.payload or {}
        result.append(
            {
                "lesson_id": int(payload.get("lesson_id", point.id)),
                "topic": payload.get("topic", ""),
                "question": payload.get("question", ""),
                "answer": payload.get("answer", ""),
                "focus": payload.get("focus", ""),
                "score": round(score, 4),
            }
        )
        try:
            appdb.count_uses(int(payload.get("lesson_id", point.id)))
        except Exception:
            pass
    return result


def as_prompt_block(lessons: list[dict]) -> str:
    if not lessons:
        return ""
    blocks = []
    for lesson in lessons:
        blocks.append(
            f"<lesson topic=\"{lesson['topic']}\">\n"
            f"Вопрос: {lesson['question']}\n"
            f"Правильный ответ: {lesson['answer']}\n"
            f"Обратить внимание: {lesson['focus']}\n"
            "</lesson>"
        )
    return (
        "Уроки администратора — следуй им при ответах на похожие вопросы:\n\n"
        + "\n\n".join(blocks)
    )


def chat(history: list[dict], question: str) -> str:
    """Диалог с наставником: история + новая реплика администратора."""
    messages = [{"role": "system", "content": TRAINER_SYSTEM}]
    for turn in history[-8:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    client = rag.get_client()
    completion = client.chat.completions.create(
        model=_model(),
        messages=messages,
        temperature=0.2,
    )
    return (completion.choices[0].message.content or "").strip()


def draft(history: list[dict]) -> dict:
    """Собрать черновик урока из диалога."""
    transcript = "\n".join(
        f"{'Администратор' if turn.get('role') == 'user' else 'Наставник'}: "
        f"{str(turn.get('content', '')).strip()}"
        for turn in history[-12:]
        if str(turn.get("content", "")).strip()
    )
    client = rag.get_client()
    completion = client.chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": TRAINER_SYSTEM},
            {"role": "user", "content": f"{DRAFT_PROMPT}\n\nДиалог:\n{transcript}"},
        ],
        temperature=0.1,
    )
    return parse_draft((completion.choices[0].message.content or "").strip())


def parse_draft(raw: str) -> dict:
    import json
    import re

    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"topic": "", "question": "", "answer": "", "focus": "", "raw": raw}
    return {
        "topic": str(data.get("topic", "")).strip(),
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "focus": str(data.get("focus", "")).strip(),
        "raw": raw,
    }


def _model() -> str:
    import settings

    return settings.value("LLM_MODEL", config.LLM_MODEL)


_cache: dict = {"at": 0.0, "value": None}


def invalidate() -> None:
    _cache.update(at=0.0, value=None)

CORRECTION_DIALOG = """Пользователь спросил: «{question}»

Администратор ответил так:
{answer}

Источник: {source}
"""


def _same_lesson(question: str) -> dict | None:
    target = (question or "").strip().lower()
    if not target:
        return None
    for item in appdb.list_lessons(status="active"):
        if item["question"].strip().lower() == target:
            return item
    return None


def learn_from_correction(correction: dict, author: str) -> dict:
    """Собрать или обновить урок на основе исправления администратора."""
    existing = _same_lesson(correction.get("question", ""))
    if existing:
        appdb.update_lesson(
            existing["id"],
            topic=(correction.get("question") or existing["topic"])[:80],
            question=correction.get("question") or existing["question"],
            answer=correction.get("answer") or existing["answer"],
            focus=correction.get("doc_ref") or existing["focus"],
        )
        reindex()
        return appdb.get_lesson(existing["id"]) or {}

    dialog = CORRECTION_DIALOG.format(
        question=correction.get("question", ""),
        answer=correction.get("answer", ""),
        source=correction.get("doc_ref") or "документ и страницы не указаны",
    )
    client = rag.get_client()
    completion = client.chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": TRAINER_SYSTEM},
            {"role": "user", "content": f"{DRAFT_PROMPT}\n\nДиалог:\n{dialog}"},
        ],
        temperature=0.1,
    )
    draft = parse_draft((completion.choices[0].message.content or "").strip())
    if not draft["topic"]:
        draft = {
            "topic": (correction.get("question") or "Исправление администратора")[:80],
            "question": correction.get("question", ""),
            "answer": correction.get("answer", ""),
            "focus": "",
        }
    if not draft["focus"]:
        draft["focus"] = correction.get("doc_ref", "")
    return save_lesson(
        draft["topic"], draft["question"], draft["answer"], draft["focus"], author
    )


class LearnJob:
    """Фоновое обучение на основе исправлений администратора."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict = {
            "status": "idle",
            "done": 0,
            "total": 0,
            "created": 0,
            "error": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def queue(self, correction: dict, author: str) -> int:
        log_id = appdb.add_training_log(
            source="correction",
            correction_id=int(correction["id"]),
            lesson_id=None,
            status="pending",
            author=author,
        )
        with self._lock:
            self.state = {
                "status": "running",
                "done": 0,
                "total": self.state.get("total", 0) + 1,
                "created": 0,
                "error": None,
            }
        thread = threading.Thread(target=self._run, args=(log_id, correction, author), daemon=True)
        thread.start()
        self._thread = thread
        return log_id

    def _run(self, log_id: int, correction: dict, author: str) -> None:
        try:
            lesson = learn_from_correction(correction, author)
            appdb.update_training_log(
                log_id, "done", lesson_id=int(lesson["id"]), note=lesson["topic"]
            )
            with self._lock:
                self.state["done"] += 1
                self.state["created"] += 1
        except Exception as exc:
            appdb.update_training_log(log_id, "error", note=f"{type(exc).__name__}: {exc}")
            with self._lock:
                self.state["done"] += 1
                self.state["error"] = f"{type(exc).__name__}: {exc}"


learn_job = LearnJob()
