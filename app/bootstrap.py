"""Первичное обучение после выбора сферы деятельности и загрузки документов.

Как только документы проиндексированы и выбрано направление с примерами вопросов,
система запускает несколько циклов обучения: прогон вопросов сферы и сгенерированных
вопросов через ассистента, эвристическая оценка, LLM-судья, построение тем и сбор
уроков-кандидатов. Итог виден в кабинете администратора.
"""

import threading

import appdb
import autolearn
import config
import domains
import questiongen
import settings
import vectorstore as vs

SETTING_KEY = "bootstrap_signature"


def _cycles() -> int:
    return settings.int_value("BOOTSTRAP_CYCLES") or config.BOOTSTRAP_CYCLES


def _count() -> int:
    return settings.int_value("BOOTSTRAP_COUNT") or config.BOOTSTRAP_COUNT


def _judge_enabled() -> bool:
    return settings.bool_value("BOOTSTRAP_JUDGE", config.BOOTSTRAP_JUDGE)


def _index_points() -> int:
    try:
        client = vs.get_qdrant()
        name = vs.target_name()
        if not client.collection_exists(name):
            return 0
        return client.get_collection(name).points_count or 0
    except Exception:
        return 0


def readiness() -> dict:
    domain = domains.current()
    questions = domains.sample_questions()
    points = _index_points()
    return {
        "domain": domain.get("slug", ""),
        "domain_title": domain.get("title", ""),
        "questions": len(questions),
        "points": points,
        "ready": bool(points > 0 and questions),
    }


def signature() -> str:
    return f"{domains.current_id()}:{_index_points()}"


def maybe_start(reason: str = "", force: bool = False, cycles: int | None = None, count: int | None = None) -> dict:
    """Запустить первичное обучение, если есть документы и вопросы сферы."""
    state = readiness()
    if not state["ready"]:
        return {"started": False, "reason": "not_ready", **state}
    sig = signature()
    if not force and (settings.raw(SETTING_KEY) or "").strip() == sig:
        return {"started": False, "reason": "already_done", **state}
    started = job.start(reason=reason or "auto", cycles=cycles, count=count)
    if started.get("started"):
        appdb.set_setting(SETTING_KEY, sig)
        settings.invalidate()
    return {**started, **state}


class BootstrapJob:
    """Фоновые циклы первичного обучения."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict = {
            "status": "idle",
            "stage": "",
            "cycle": 0,
            "cycles": 0,
            "reason": "",
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

    def start(self, reason: str = "manual", cycles: int | None = None, count: int | None = None) -> dict:
        if self.snapshot()["status"] == "running":
            return {**self.snapshot(), "started": False}
        cycles = max(1, cycles or _cycles())
        self._update(
            status="running",
            stage="Подготовка",
            cycle=0,
            cycles=cycles,
            reason=reason,
            started_at=appdb.now(),
            finished_at=None,
            result={},
            error=None,
        )
        self._thread = threading.Thread(target=self._run, args=(cycles, count), daemon=True)
        self._thread.start()
        return {**self.snapshot(), "started": True}

    def _run(self, cycles: int, count: int | None) -> None:
        try:
            count = max(1, count or _count())
            domain = domains.current()
            seeds = [
                {"question": question, "source": "domain"}
                for question in domains.sample_questions()
                if question
            ][:count]
            aggregates: list[dict] = []
            for cycle in range(1, cycles + 1):
                if cycle == 1 and seeds:
                    questions = seeds
                    self._update(stage=f"Цикл {cycle}/{cycles}: вопросы сферы")
                else:
                    self._update(stage=f"Цикл {cycle}/{cycles}: генерация вопросов")
                    questions = questiongen.generate(count, ["docs", "topics"])
                    questions = [item for item in questions if item.get("question")]
                if not questions:
                    aggregates.append({"cycle": cycle, "questions": 0, "ran": 0})
                    continue
                self._update(stage=f"Цикл {cycle}/{cycles}: прогон вопросов")
                run_result = questiongen.run(
                    questions,
                    progress=lambda stage, cycle=cycle: self._update(
                        stage=f"Цикл {cycle}/{cycles}: {stage}"
                    ),
                )
                refreshed = autolearn.refresh(
                    llm=_judge_enabled(),
                    progress=lambda stage, cycle=cycle: self._update(
                        stage=f"Цикл {cycle}/{cycles}: {stage}"
                    ),
                )
                aggregates.append({
                    "cycle": cycle,
                    "questions": len(questions),
                    "ran": run_result.get("ran", 0),
                    "grounded": run_result.get("grounded", 0),
                    "problems": run_result.get("problems", 0),
                    "candidates": refreshed.get("candidates", 0),
                })
            self._update(
                status="done",
                stage="Готово",
                finished_at=appdb.now(),
                result={
                    "domain": domain.get("title", ""),
                    "cycles": aggregates,
                    "questiongen": questiongen.summary(),
                },
            )
        except Exception as exc:
            self._update(
                status="error",
                stage="Ошибка",
                finished_at=appdb.now(),
                error=f"{type(exc).__name__}: {exc}",
            )


job = BootstrapJob()
