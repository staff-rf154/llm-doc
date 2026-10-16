"""Оценка качества RAG: фоновый запуск eval/eval.py по датасету из data/eval.

Отчёт пишется в data/eval/report.json; кабинет показывает сводные метрики
(context_recall, answer_keyword_hit_rate, grounded_rate) и последние ответы.
"""

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import config

REPORT_PATH = Path(config.EVAL_DIR) / "report.json"
DATASET_PATH = Path(config.EVAL_DIR) / "dataset.json"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def read_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summary() -> dict:
    report = read_report()
    rows = report.get("rows") or []
    return {
        "exists": bool(report),
        "dataset_exists": DATASET_PATH.exists(),
        "updated_at": datetime.fromtimestamp(REPORT_PATH.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )
        if REPORT_PATH.exists()
        else None,
        "samples": report.get("samples", 0),
        "grounded_rate": report.get("grounded_rate"),
        "context_recall": report.get("context_recall"),
        "answer_keyword_hit_rate": report.get("answer_keyword_hit_rate"),
        "exact_match": report.get("exact_match"),
        "ragas": report.get("ragas"),
        "rows": [
            {
                "question": row.get("question", ""),
                "context_recall": row.get("context_recall"),
                "answer_keyword_hit_rate": row.get("answer_keyword_hit_rate"),
                "grounded": row.get("grounded"),
                "refused": row.get("refused"),
                "sources": row.get("sources", []),
            }
            for row in rows
        ],
    }


class EvalJob:
    """Фоновый запуск оценки качества."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict = {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "error": None,
            "ragas": False,
            "log": "",
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _update(self, **fields) -> None:
        with self._lock:
            self.state.update(fields)

    def start(self, ragas: bool = False) -> dict:
        if self.snapshot()["status"] == "running":
            return {**self.snapshot(), "started": False}
        if not DATASET_PATH.exists():
            return {"started": False, "error": "Не найден датасет data/eval/dataset.json"}
        self._update(
            status="running",
            started_at=_now(),
            finished_at=None,
            error=None,
            ragas=bool(ragas),
            log="",
        )
        self._thread = threading.Thread(target=self._run, args=(bool(ragas),), daemon=True)
        self._thread.start()
        return {**self.snapshot(), "started": True}

    def _run(self, ragas: bool) -> None:
        command = [sys.executable, "eval/eval.py"] + (["--ragas"] if ragas else [])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=7200,
                cwd=str(Path(__file__).parent),
            )
            log = (result.stdout or "")[-4000:]
            self._update(
                status="done" if result.returncode == 0 else "error",
                finished_at=_now(),
                error=None if result.returncode == 0 else (result.stderr or "")[-500:],
                log=log,
            )
        except Exception as exc:
            self._update(status="error", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")


job = EvalJob()