"""Автообучение: оценка ответов, ассоциации вопросов и уроки-кандидаты.

Конвейер:
1. Эвристики по каждому запросу (grounded, источники, оценки пользователя)
   пишутся в `request_evals`; проблемные случаи попадают в `answer_issues`.
2. Ассоциации: похожие вопросы группируются в темы (`question_topics`).
3. LLM-судья (по заданию) уточняет вердикт и предлагает правильный ответ.
4. Из проблемных тем собираются уроки-кандидаты (status='candidate'). Они не
   влияют на ответы, пока администратор не одобрит их в кабинете.
"""

import json
import re
import threading
import time

import numpy as np

import appdb
import config
import lessons
import rag
import settings
import vectorstore as vs

VERDICTS = ("ok", "partial", "wrong", "refused_wrongly", "refused", "unknown")
PROBLEM_VERDICTS = ("partial", "wrong", "refused_wrongly")
MANAGED_ISSUES = ("downvote", "refusal", "no_sources")

JUDGE_SYSTEM = (
    "Ты — строгий проверяющий ответов ассистента по внутренней документации. "
    "Тебе дают вопрос пользователя, найденные фрагменты документации и ответ ассистента. "
    "Оцени, насколько ответ верен и полон относительно фрагментов.\n"
    "Верни строго JSON без пояснений:\n"
    '  "verdict" — один из ok|partial|wrong|refused_wrongly;\n'
    '  "score" — число от 0 до 1 (уверенность, что ответ корректен);\n'
    '  "reason" — одно короткое предложение, в чём ошибка или почему ответ хорош;\n'
    '  "suggested_answer" — эталонный ответ по фрагментам, если verdict не ok, иначе пусто.\n'
    "Трактовка: ok — ответ верный и достаточный; partial — неполный или расплывчатый; "
    "wrong — противоречит фрагментам или уводит в сторону; "
    "refused_wrongly — ассистент ответил, что в документации нет информации, "
    "хотя во фрагментах ответ есть. Не выдумывай факты и номера страниц."
)


def _window() -> int:
    return settings.int_value("AUTOLEARN_WINDOW") or config.AUTOLEARN_WINDOW


def _days() -> int:
    return settings.int_value("AUTOLEARN_DAYS")


def heuristic_verdict(request: dict) -> dict:
    """Оценка запроса без LLM: по факту ответа и оценке пользователя."""
    grounded = int(request.get("grounded") or 0)
    corrected = int(request.get("corrected") or 0)
    sources = int(request.get("sources") or 0)
    vote = request.get("vote")
    if corrected:
        return {
            "verdict": "ok",
            "score": 1.0,
            "reason": "ответ взят из исправления администратора",
            "issue": None,
        }
    if vote == "down":
        return {
            "verdict": "wrong",
            "score": 0.1,
            "reason": "пользователь поставил дизлайк",
            "issue": ("downvote", "пользователь поставил дизлайк"),
        }
    if not grounded:
        return {
            "verdict": "refused",
            "score": 0.3,
            "reason": "ассистент ответил, что в документации нет информации",
            "issue": ("refusal", "отказ: ответа нет в найденных фрагментах"),
        }
    if not sources:
        return {
            "verdict": "partial",
            "score": 0.5,
            "reason": "ответ сформирован без источников",
            "issue": ("no_sources", "ответ без найденных фрагментов документации"),
        }
    return {
        "verdict": "ok",
        "score": 0.9,
        "reason": "ответ по документации с источниками",
        "issue": None,
    }


def _sync_issues(request_id: int, info: dict) -> None:
    issue = info.get("issue")
    if issue:
        appdb.add_issue(request_id, issue[0], issue[1])
        appdb.close_issues(request_id, tuple(k for k in MANAGED_ISSUES if k != issue[0]))
    else:
        appdb.close_issues(request_id, MANAGED_ISSUES)


def evaluate_request(request_id: int, force: bool = False) -> dict | None:
    """Эвристическая оценка одного запроса. Не затирает вердикт модели."""
    request = appdb.get_request_full(request_id)
    if not request:
        return None
    existing = appdb.get_request_eval(request_id)
    info = heuristic_verdict(request)
    if force or not existing or existing.get("method") != "llm":
        appdb.upsert_request_eval(
            request_id, info["verdict"], info["score"], info["reason"], method="heuristic"
        )
    _sync_issues(request_id, info)
    return info


def evaluate_recent(limit: int | None = None) -> dict:
    """Эвристики по последним запросам пользователей."""
    limit = limit or _window()
    rows = appdb.recent_requests(limit, _days() or None)
    done = 0
    for row in rows:
        if len((row.get("question") or "").strip()) < 2:
            continue
        evaluate_request(int(row["id"]))
        done += 1
    return {"evaluated": done, "window": limit}


def _problem_ids() -> set[int]:
    return appdb.problem_request_ids()


def _shortest(questions: list[str]) -> str:
    return min(questions, key=len) if questions else ""


def build_topics(
    limit: int | None = None,
    threshold: float | None = None,
    max_topics: int | None = None,
) -> int:
    """Построить ассоциации: сгруппировать похожие вопросы в темы."""
    limit = limit or _window()
    threshold = threshold or settings.float_value("AUTOLEARN_TOPIC_THRESHOLD") or config.AUTOLEARN_TOPIC_THRESHOLD
    max_topics = max_topics or settings.int_value("AUTOLEARN_MAX_TOPICS") or config.AUTOLEARN_MAX_TOPICS
    rows = appdb.recent_requests(limit, _days() or None)
    problems = _problem_ids()

    clusters: list[dict] = []
    for row in rows:
        question = (row.get("question") or "").strip()
        if len(question) < 3:
            continue
        try:
            vec = np.asarray(vs.embed_query(question), dtype=np.float32)
        except Exception:
            continue
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            continue
        vec = vec / norm

        best = None
        best_score = -1.0
        for cluster in clusters:
            score = float(np.dot(vec, cluster["centroid"]))
            if score > best_score:
                best_score = score
                best = cluster
        is_problem = int(row["id"]) in problems
        if best is not None and best_score >= threshold:
            count = best["n"]
            centroid = (best["centroid"] * count + vec) / (count + 1)
            length = float(np.linalg.norm(centroid))
            best["centroid"] = centroid / length if length else centroid
            best["n"] = count + 1
            best["members"].append((int(row["id"]), best_score))
            best["questions"].append(question)
            best["problematic"] += 1 if is_problem else 0
        else:
            clusters.append(
                {
                    "centroid": vec,
                    "n": 1,
                    "members": [(int(row["id"]), 1.0)],
                    "questions": [question],
                    "problematic": 1 if is_problem else 0,
                }
            )

    clusters.sort(key=lambda item: (item["problematic"], item["n"]), reverse=True)
    items = []
    for cluster in clusters[:max_topics]:
        representative = _shortest(cluster["questions"])
        items.append(
            {
                "title": representative[:80],
                "representative": representative,
                "total": cluster["n"],
                "problematic": cluster["problematic"],
                "members": cluster["members"],
            }
        )
    return appdb.replace_topics(items)


def _fragments(question: str, limit: int = 4) -> str:
    try:
        chunks = rag.retrieve(question, limit)
    except Exception:
        return ""
    blocks = []
    for chunk in chunks:
        blocks.append(f"Документ: {rag.document_name(chunk)} ({chunk.source})\n{chunk.text}")
    return "\n\n".join(blocks)


def _parse_json(raw: str) -> dict:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def judge(request_id: int) -> dict | None:
    """LLM-судья: уточнить вердикт и предложить правильный ответ."""
    request = appdb.get_request_full(request_id)
    if not request:
        return None
    question = request.get("question") or ""
    answer = request.get("answer") or ""
    context = _fragments(question)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Вопрос пользователя: {question}\n\n"
                f"Ответ ассистента: {answer}\n\n"
                f"Фрагменты документации:\n{context or 'фрагменты не найдены'}"
            ),
        },
    ]
    client = rag.get_client()
    completion = client.chat.completions.create(
        model=settings.value("LLM_MODEL", config.LLM_MODEL),
        messages=messages,
        temperature=0.0,
    )
    data = _parse_json(completion.choices[0].message.content or "")
    verdict = str(data.get("verdict", "")).strip()
    if verdict not in VERDICTS:
        verdict = "unknown"
    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    reason = str(data.get("reason", "")).strip()
    suggested = str(data.get("suggested_answer", "")).strip()
    appdb.upsert_request_eval(
        request_id, verdict, score, reason or "оценка модели", method="llm", suggested_answer=suggested
    )
    if verdict in PROBLEM_VERDICTS:
        appdb.add_issue(request_id, "llm_wrong", reason or f"вердикт модели: {verdict}")
    elif verdict == "ok":
        appdb.close_issues(request_id, ("llm_wrong",))
    return {"request_id": request_id, "verdict": verdict, "score": score, "reason": reason}


def judge_batch(limit: int | None = None) -> dict:
    limit = limit or settings.int_value("AUTOLEARN_LLM_LIMIT") or config.AUTOLEARN_LLM_LIMIT
    done = 0
    errors = 0
    for request_id in appdb.requests_for_llm(limit):
        try:
            judge(request_id)
        except Exception:
            errors += 1
            continue
        done += 1
    return {"judged": done, "errors": errors}


def _lesson_exists(question: str) -> bool:
    target = (question or "").strip().lower()
    if not target:
        return False
    for item in appdb.list_lessons():
        if item["status"] in ("active", "candidate") and item["question"].strip().lower() == target:
            return True
    return False


def _draft_from_question(question: str) -> dict:
    context = lessons.documentation_context(question)
    messages = [{"role": "system", "content": lessons.TRAINER_SYSTEM}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{lessons.DRAFT_PROMPT}\n\nДиалог:\n"
                f"Пользователь спросил: «{question}». Ответ ассистента оказался неполным "
                "или ошибочным. Собери эталонный урок по фрагментам документации."
            ),
        }
    )
    client = rag.get_client()
    completion = client.chat.completions.create(
        model=settings.value("LLM_MODEL", config.LLM_MODEL),
        messages=messages,
        temperature=0.1,
    )
    return lessons.parse_draft((completion.choices[0].message.content or "").strip())


def generate_candidates(min_hits: int | None = None, max_topics: int | None = None) -> list[dict]:
    """Собрать уроки-кандидаты из проблемных тем (в промпт не попадают)."""
    min_hits = min_hits or settings.int_value("AUTOLEARN_MIN_TOPIC_HITS") or config.AUTOLEARN_MIN_TOPIC_HITS
    max_topics = max_topics or settings.int_value("AUTOLEARN_MAX_TOPICS") or config.AUTOLEARN_MAX_TOPICS
    created: list[dict] = []
    for topic in appdb.list_topics(max_topics):
        if int(topic["problematic"]) < int(min_hits):
            continue
        representative = topic.get("representative") or topic.get("title") or ""
        if not representative or _lesson_exists(representative):
            continue
        draft = _draft_from_question(representative)
        if not draft.get("answer"):
            continue
        lesson = lessons.save_lesson(
            draft["topic"] or representative[:80],
            draft["question"] or representative,
            draft["answer"],
            draft["focus"],
            "автообучение",
            status="candidate",
        )
        created.append(lesson)
    return created


def summary() -> dict:
    candidates = appdb.list_lessons(status="candidate")
    return {
        "evals": appdb.eval_summary(),
        "issues": appdb.issue_summary(),
        "topics": len(appdb.list_topics()),
        "candidates": len(candidates),
    }


def refresh(llm: bool | None = None, progress=None) -> dict:
    """Полный прогон: эвристики → ассоциации → LLM-оценка → уроки-кандидаты."""
    if llm is None:
        llm = settings.bool_value("AUTOLEARN_LLM_ENABLED", config.AUTOLEARN_LLM_ENABLED)
    result: dict = {}
    if progress:
        progress("Эвристическая оценка запросов")
    result.update(evaluate_recent())
    if progress:
        progress("Построение ассоциаций")
    result["topics"] = build_topics()
    if llm:
        if progress:
            progress("Оценка ответов моделью")
        result.update(judge_batch())
        if progress:
            progress("Сбор уроков-кандидатов")
        result["candidates"] = len(generate_candidates())
    return result


class AnalyzeJob:
    """Фоновый прогон автообучения."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict = {
            "status": "idle",
            "stage": "",
            "started_at": None,
            "finished_at": None,
            "result": {},
            "error": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _update(self, **fields) -> None:
        with self._lock:
            self.state.update(fields)

    def start(self, llm: bool | None = None) -> dict:
        if self.snapshot()["status"] == "running":
            return {**self.snapshot(), "started": False}
        self._update(
            status="running",
            stage="Запуск",
            started_at=appdb.now(),
            finished_at=None,
            result={},
            error=None,
        )
        self._thread = threading.Thread(target=self._run, args=(llm,), daemon=True)
        self._thread.start()
        return {**self.snapshot(), "started": True}

    def _run(self, llm: bool | None) -> None:
        try:
            result = refresh(llm=llm, progress=lambda stage: self._update(stage=stage))
            self._update(status="done", stage="Готово", finished_at=appdb.now(), result=result)
        except Exception as exc:
            self._update(
                status="error",
                stage="Ошибка",
                finished_at=appdb.now(),
                error=f"{type(exc).__name__}: {exc}",
            )


job = AnalyzeJob()
