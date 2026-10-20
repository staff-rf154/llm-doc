"""Регрессионная проверка метрик относительно baseline."""

import argparse
import json
from pathlib import Path

import config

METRICS = (
    "grounded_rate",
    "context_recall",
    "answer_keyword_hit_rate",
    "exact_match",
    "answer_correctness",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнить отчёт оценки с baseline")
    parser.add_argument(
        "--report", default=str(Path(config.EVAL_DIR) / "report.json")
    )
    parser.add_argument(
        "--baseline", default=str(Path(config.EVAL_DIR) / "baseline.json")
    )
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    report_path = Path(args.report)
    baseline_path = Path(args.baseline)
    if not report_path.exists():
        raise SystemExit(f"нет отчёта: {report_path}")
    report = load(report_path)

    if not baseline_path.exists():
        baseline_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"baseline создан: {baseline_path}")
        return

    baseline = load(baseline_path)
    regressions = []
    print(f"{'метрика':<28}{'baseline':>10}{'текущая':>10}{'дельта':>10}")
    for key in METRICS:
        if key not in report:
            continue
        current = float(report.get(key) or 0.0)
        base = float(baseline.get(key) or 0.0)
        delta = current - base
        limit = max(abs(base) * args.threshold, 0.01)
        mark = ""
        if delta < -limit:
            mark = "  РЕГРЕСС"
            regressions.append(key)
        print(f"{key:<28}{base:>10.4f}{current:>10.4f}{delta:>+10.4f}{mark}")

    if regressions:
        print("регресс: " + ", ".join(regressions))
        raise SystemExit(1)
    print("регресса нет")


if __name__ == "__main__":
    main()
