"""Генерация проверочных вопросов и прогон их через ассистента.

Вопросы собираются из трёх источников:
- `docs` — по фрагментам проиндексированной документации (LLM формулирует вопросы,
  на которые фрагмент даёт ответ);
- `topics` — по уже накопленным темам вопросов пользователей;
- `domain` — по активной сфере деятельности (примеры вопросов, категории, термины).

Каждый вопрос прогоняется через обычный ответ ассистента (`rag.ask`), пишется в
журнал запросов с origin='generated' и оценивается эвристиками; при необходимости
дополняется LLM-судьёй. Результат виден в отчёте кабинета администратора.
"""

import json
import random
import re
import threading
import time

import appdb
import autolearn
import config
import domains
import rag
import settings
import vectorstore as vs

SOURCES = ("docs", "topics", "domain")
ORIGIN = "generated"

DOC_SYSTEM = (
    "Ты — методист по внутренней документации компании. "
    "Тебе дают фрагменты документа. Составь по ним проверочные вопросы, "
    "на которые эти фрагменты дают ответ. Формулируй конкретно, на русском, "
    "используй термины и названия из фрагментов и не выдумывай факты. "
    'Верни строго JSON без пояснений: {"questions": ["вопрос", "вопрос"]}.'
)

TOPIC_SYSTEM = (
    "Ты — методист. Ниже темы вопросов пользователей по внутренней документации. "
    "Сформулируй по ним новые проверочные вопросы, близкие по смыслу, но иначе "
    "сформулированные и более конкретные. На русском, без выдуманных фактов. "
    'Верни строго JSON без пояснений: {"questions": ["вопрос", "вопрос"]}.'
)

DOMAIN_SYSTEM = (
    "Ты — методист. Ниже сфера деятельности компании, её категории вопросов и "
    "терминология. Составь проверочные вопросы по внутренней документации этой "
    "сферы. На русском, конкретно, с использованием терминологии. "
    'Верни строго JSON без пояснений: {"questions": ["вопрос", "вопрос"]}.'
)


def _count() -> int:
    return settings.int_value("QUESTIONGEN_COUNT") or config.QUESTIONGEN_COUNT


def _sources() -> list[str]:
    raw = (settings.value("QUESTIONGEN_SOURCES") or config.QUESTIONGEN_SOURCES or "").lower()
    items = [part.strip() for part in raw.split(",") if part.strip()]
    chosen = [item for item in items if item in SOURCES]
    return chosen or list(SOURCES)


def _judge_enabled() -> bool:
    return settings.bool_value("QUESTIONGEN_JUDGE", config.QUESTIONGEN_JUDGE)


def _client():
    return rag.get_client()


def _parse_questions(raw: str) -> list[str]:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        question = str(item).strip().strip('"').strip()
        if len(question) >= 8:
            result.append(question.rstrip("?") + "?")
    return result


def _ask_questions(system: str, user: str) -> list[str]:
    completion = _client().chat.completions.create(
        model=settings.value("LLM_MODEL", config.LLM_MODEL),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )
    return _parse_questions(completion.choices[0].message.content or "")


def _point_chunk(source: str, chunk: int) -> str:
    from qdrant_client import models

    client = vs.get_qdrant()
    target = vs.target_name()
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=source)),
            models.FieldCondition(key="chunk", match=models.MatchValue(value=chunk)),
        ]
    )
    points, _ = client.scroll(
        collection_name=target,
        limit=1,
        scroll_filter=scroll_filter,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return ""
    return str((points[0].payload or {}).get("text") or "").strip()


def _doc_samples(count: int) -> list[dict]:
    sources = vs.known_sources()
    if not sources:
        return []
    random.shuffle(sources)
    samples: list[dict] = []
    for source in sources:
        total = vs.count_source_points(source)
        if total <= 0:
            continue
        chunk = random.randrange(total)
        text = _point_chunk(source, chunk)
        if len(text) < 80:
            continue
        samples.append({"source": source, "text": text})
        if len(samples) >= count:
            break
    return samples


def _from_docs(count: int) -> list[str]:
    samples = _doc_samples(min(count, 12))
    questions: list[str] = []
    for start in range(0, len(samples), 3):
        batch = samples[start:start + 3]
        if not batch:
            break
        blocks = [
            f"Фрагмент {index + 1} (документ: {item['source']}):\n{item['text']}"
            for index, item in enumerate(batch)
        ]
        user = (
            "Составь до 4 проверочных вопросов по этим фрагментам:\n\n"
            + "\n\n".join(blocks)
        )
        try:
            questions.extend(_ask_questions(DOC_SYSTEM, user))
        except Exception:
            continue
        if len(questions) >= count:
            break
    return questions[:count]


def _from_topics(count: int) -> list[str]:
    topics = appdb.list_topics()
    if not topics:
        return []
    lines = []
    for topic in topics[:30]:
        representative = topic.get("representative") or topic.get("title") or ""
        if representative:
            lines.append(f"— {representative}")
    if not lines:
        return []
    user = "Темы вопросов:\n" + "\n".join(lines) + f"\n\nСоставь до {count} вопросов."
    try:
        return _ask_questions(TOPIC_SYSTEM, user)[:count]
    except Exception:
        return [topic.get("representative", "") for topic in topics[:count] if topic.get("representative")]


def _from_domain(count: int) -> list[str]:
    domain = domains.current()
    seeds = [q for q in domains.sample_questions() if q]
    if len(seeds) >= count:
        return seeds[:count]
    categories = domains.question_categories()
    terms = domains.terminology()
    if not categories and not terms:
        return seeds
    user = (
        f"Сфера деятельности: {domain.get('title', '')}.\n"
        f"Категории вопросов: {', '.join(categories) or '—'}.\n"
        f"Терминология: {', '.join(terms) or '—'}.\n\n"
        f"Составь до {max(1, count - len(seeds))} вопросов."
    )
    try:
        generated = _ask_questions(DOMAIN_SYSTEM, user)
    except Exception:
        generated = []
    result = seeds + [q for q in generated if q]
    return result[:count]


def _existing_questions() -> set[str]:
    known: set[str] = set()
    for row in appdb.recent_requests(1000):
        question = (row.get("question") or "").strip().lower()
        if question:
            known.add(question)
    for topic in appdb.list_topics():
        text = (topic.get("representative") or topic.get("title") or "").strip().lower()
        if text:
            known.add(text)
    for item in appdb.list_corrections():
        question = (item.get("question") or "").strip().lower()
        if question:
            known.add(question)
    return known


def generate(count: int | None = None, sources: list[str] | None = None) -> list[dict]:
    """Сгенерировать вопросы. Возвращает список {question, source} без дублей."""
    count = count or _count()
    sources = sources or _sources()
    known = _existing_questions()
    collected: list[dict] = []
    seen: set[str] = set()

    def add(question: str, source: str) -> None:
        text = (question or "").strip()
        if len(text) < 8:
            return
        key = text.lower()
        if key in known or key in seen:
            return
        seen.add(key)
        collected.append({"question": text, "source": source})

    for source in sources:
        if len(collected) >= count:
            break
        remaining = count - len(collected)
        if source == "docs":
            generated = _from_docs(remaining)
        elif source == "topics":
            generated = _from_topics(remaining)
        elif source == "domain":
            generated = _from_domain(remaining)
        else:
            generated = []
        for question in generated:
            add(question, source)
            if len(collected) >= count:
                break
    return collected


def _run_question(question: str) -> dict:
    started = time.time()
    answer = rag.ask(question)
    latency_ms = int((time.time() - started) * 1000)
    request_id = appdb.log_request(
        None,
        question,
        answer.text,
        answer.grounded,
        answer.corrected,
        len(answer.chunks),
        latency_ms,
        reference=answer.reference,
        origin=ORIGIN,
    )
    info = autolearn.evaluate_request(request_id) or {}
    return {
        "request_id": request_id,
        "question": question,
        "grounded": bool(answer.grounded),
        "sources": len(answer.chunks),
        "verdict": info.get("verdict", "unknown"),
    }


def run(questions: list[dict], progress=None) -> dict:
    """Прогнать вопросы через ассистента: ответ + запись в журнал + эвристики."""
    report: list[dict] = []
    for index, item in enumerate(questions):
        question = item["question"] if isinstance(item, dict) else str(item)
        if progress:
            progress(f"Прогон вопроса {index + 1} из {len(questions)}")
        try:
            row = _run_question(question)
            row["source"] = item.get("source", "") if isinstance(item, dict) else ""
            report.append(row)
        except Exception as exc:
            report.append({
                "request_id": None,
                "question": question,
                "grounded": False,
                "sources": 0,
                "verdict": "unknown",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return {
        "ran": len(report),
        "grounded": sum(1 for row in report if row.get("grounded")),
        "problems": sum(
            1 for row in report
            if row.get("verdict") in autolearn.PROBLEM_VERDICTS
        ),
        "items": report,
    }


def report(limit: int = 100, offset: int = 0) -> dict:
    items = appdb.list_request_evals(limit=limit, offset=offset, origin=ORIGIN)
    total = appdb.count_request_evals(origin=ORIGIN)
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def summary() -> dict:
    total = appdb.count_request_evals(origin=ORIGIN)
    problems = appdb.count_request_evals(only_problems=True, origin=ORIGIN)
    return {"generated": total, "problems": problems}


class GenerateJob:
    """Фоновый прогон: генерация вопросов → ответы → эвристики → LLM-судья."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict = {
            "status": "idle",
            "stage": "",
            "started_at": None,
            "finished_at": None,
            "questions": [],
            "result": {},
            "error": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _update(self, **fields) -> None:
        with self._lock:
            self.state.update(fields)

    def start(self, count: int | None = None, sources: list[str] | None = None, judge: bool | None = None) -> dict:
        if self.snapshot()["status"] == "running":
            return {**self.snapshot(), "started": False}
        self._update(
            status="running",
            stage="Запуск",
            started_at=appdb.now(),
            finished_at=None,
            questions=[],
            result={},
            error=None,
        )
        self._thread = threading.Thread(target=self._run, args=(count, sources, judge), daemon=True)
        self._thread.start()
        return {**self.snapshot(), "started": True}

    def _run(self, count: int | None, sources: list[str] | None, judge: bool | None) -> None:
        try:
            self._update(stage="Генерация вопросов")
            questions = generate(count, sources)
            self._update(questions=questions)
            result = run(questions, progress=lambda stage: self._update(stage=stage))
            if judge is None:
                judge = _judge_enabled()
            if judge and questions:
                self._update(stage="Оценка ответов моделью")
                judged = 0
                errors = 0
                for item in questions:
                    try:
                        row = next(
                            (r for r in result["items"] if r.get("question") == item["question"]),
                            None,
                        )
                        if row and row.get("request_id"):
                            autolearn.judge(int(row["request_id"]))
                            judged += 1
                    except Exception:
                        errors += 1
                result["judged"] = judged
                result["judge_errors"] = errors
            result["summary"] = summary()
            self._update(status="done", stage="Готово", finished_at=appdb.now(), result=result)
        except Exception as exc:
            self._update(
                status="error",
                stage="Ошибка",
                finished_at=appdb.now(),
                error=f"{type(exc).__name__}: {exc}",
            )


job = GenerateJob()
