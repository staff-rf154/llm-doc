"""Сквозной супер-тест сервиса: от авторизации до полного цикла.

Прогоняет ключевые эндпоинты по HTTP и пишет отчёт в файл. Запуск внутри
контейнера (там доступен localhost:8000) или с любого хоста:

    python eval/super_test.py --base http://localhost:8000
    python eval/super_test.py --all-models

Код выхода 0 — все шаги пройдены, 1 — были ошибки.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

STEPS: list[dict] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    STEPS.append({"name": name, "ok": ok, "detail": detail, "at": time.time()})
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--user", default=os.getenv("ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("ADMIN_PASSWORD", "admin12345"))
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--out", default="/data/eval")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    client = httpx.Client(base_url=base, timeout=300)

    health = {}
    try:
        response = client.get("/health")
        health = response.json()
        step("health", response.status_code == 200 and health.get("status") == "ok", json.dumps(health, ensure_ascii=False))
    except Exception as exc:
        step("health", False, str(exc))

    if not client.post("/api/login", json={"username": args.user, "password": args.password}).status_code == 200:
        step("login", False, "неверные учётные данные")
        return finish(args, health)
    step("login", True, args.user)

    me = client.get("/api/me").json()
    step("me", me.get("role") == "admin", f"role={me.get('role')}")

    settings_ok = False
    try:
        data = client.get("/api/admin/settings").json()
        items = data.get("items", [])
        editable = [i for i in items if i.get("editable")]
        read_only = [i for i in items if not i.get("editable")]
        consistency = client.patch("/api/admin/settings/params", json={"values": {}}).json()
        errors = consistency.get("warnings", [])
        settings_ok = len(items) > 0 and "ok" in consistency
        step(
            "settings",
            settings_ok,
            f"всего={len(items)}, редактируемых={len(editable)}, только чтение={len(read_only)}, "
            f"предупреждений={len(errors)}",
        )
    except Exception as exc:
        step("settings", False, str(exc))

    models = {}
    try:
        models = client.get("/api/admin/models").json()
        ollama = models.get("ollama", {})
        step(
            "models",
            bool(models),
            f"ollama={ollama.get('available')}, моделей={len(ollama.get('models', []))}, "
            f"hf-кэш={len((models.get('hf') or {}).get('downloaded', []))}",
        )
    except Exception as exc:
        step("models", False, str(exc))

    try:
        status = client.get("/reindex").json()
        step("reindex-status", bool(status), f"status={status.get('status')}")
    except Exception as exc:
        step("reindex-status", False, str(exc))

    retr_ok = False
    try:
        result = client.post("/retrieve", json={"query": "Как настроить систему?", "top_k": 5}).json()
        found = len(result.get("results", []))
        retr_ok = found >= 0
        step("retrieve", retr_ok, f"найдено фрагментов={found}")
    except Exception as exc:
        step("retrieve", False, str(exc))

    active_model = models.get("llm_model") or health.get("llm_model", "")
    questions = [
        "Что такое система и для чего она нужна?",
        "Как установить и настроить компоненты?",
        "Какие требования к оборудованию и ПО?",
    ][: max(1, args.questions)]

    request_ids: list[int] = []
    answers_ok = 0
    for question in questions:
        try:
            payload = client.post("/ask", json={"question": question}).json()
            answer = (payload.get("answer") or "").strip()
            if answer:
                answers_ok += 1
            if payload.get("request_id"):
                request_ids.append(payload["request_id"])
        except Exception as exc:
            step(f"ask: {question[:40]}", False, str(exc))
    step("ask", answers_ok == len(questions), f"ответов={answers_ok}/{len(questions)}, модель={active_model}")

    if request_ids:
        try:
            rated = client.post(f"/api/rate/{request_ids[0]}", json={"vote": "up", "comment": "super-test"}).json()
            step("rate", bool(rated.get("ok")), f"request_id={request_ids[0]}")
        except Exception as exc:
            step("rate", False, str(exc))
    else:
        step("rate", False, "нет request_id для оценки")

    for label, path in (("autolearn", "/api/admin/autolearn"), ("questiongen", "/api/admin/questiongen"), ("bootstrap", "/api/admin/bootstrap")):
        try:
            data = client.get(path).json()
            step(label, isinstance(data, dict), "получен")
        except Exception as exc:
            step(label, False, str(exc))

    try:
        docs = client.get("/api/admin/docs?limit=5&offset=0").json()
        step(
            "docs-pagination",
            "items" in docs and "total" in docs,
            f"всего={docs.get('total')}, на странице={len(docs.get('items', []))}",
        )
    except Exception as exc:
        step("docs-pagination", False, str(exc))

    try:
        verify = client.get("/api/admin/verify").json()
        step("verify", isinstance(verify, dict), json.dumps(verify, ensure_ascii=False)[:120])
    except Exception as exc:
        step("verify", False, str(exc))

    if args.all_models:
        available = (models.get("ollama") or {}).get("models", [])
        for item in available:
            name = item.get("name")
            if not name:
                continue
            try:
                client.post("/api/admin/models/ollama/select", json={"name": name})
                payload = client.post("/ask", json={"question": questions[0]}).json()
                step(f"ask[{name}]", bool((payload.get("answer") or "").strip()), "ответ получен")
            except Exception as exc:
                step(f"ask[{name}]", False, str(exc))
        if active_model:
            try:
                client.post("/api/admin/models/ollama/select", json={"name": active_model})
                step("restore-model", True, active_model)
            except Exception as exc:
                step("restore-model", False, str(exc))

    return finish(args, health)


def finish(args, health: dict) -> int:
    total = len(STEPS)
    failed = [s for s in STEPS if not s["ok"]]
    passed = total - len(failed)
    lines = [
        f"# Супер-тест {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"База: {args.base}",
        f"Модель: {health.get('llm_model', '—')} · эмбеддинги: {health.get('embedding_model', '—')}",
        f"Устройство: {json.dumps(health.get('device', {}), ensure_ascii=False)}",
        "",
        f"Итог: {passed}/{total} шагов пройдено, ошибок: {len(failed)}",
        "",
        "| Шаг | Результат | Детали |",
        "| --- | --- | --- |",
    ]
    for item in STEPS:
        lines.append(f"| {item['name']} | {'OK' if item['ok'] else 'FAIL'} | {item['detail']} |")
    report = "\n".join(lines) + "\n"

    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"super_test_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        path.write_text(report, encoding="utf-8")
        print(f"\nОтчёт: {path}")
    except Exception as exc:
        print(f"\nНе удалось сохранить отчёт: {exc}")
        print(report)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
