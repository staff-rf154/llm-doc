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
ROLE = ""


def step(name: str, ok: bool, detail: str = "") -> None:
    STEPS.append({"name": name, "ok": ok, "detail": detail, "at": time.time()})
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def login(client: httpx.Client, user: str, password: str) -> bool:
    try:
        response = client.post("/api/login", json={"username": user, "password": password})
    except Exception as exc:
        step("login", False, str(exc))
        return False
    if response.status_code != 200:
        return False
    me = client.get("/api/me").json()
    step("login", me.get("role") in ("admin", "system"), f"user={user}, role={me.get('role')}")
    return True


def main() -> int:
    global ROLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--user", default=os.getenv("ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("ADMIN_PASSWORD", "admin12345"))
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--out", default="/data/eval")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    client = httpx.Client(base_url=base, timeout=600)

    health: dict = {}
    try:
        response = client.get("/health")
        health = response.json()
        step(
            "health",
            response.status_code == 200 and health.get("status") == "ok",
            f"collection={health.get('collection')}, model={health.get('llm_model')}",
        )
    except Exception as exc:
        step("health", False, str(exc))

    try:
        deep = client.get("/health?deep=1").json()
        checks = deep.get("checks", {})
        failed = [name for name, value in checks.items() if not value.get("ok")]
        step(
            "deep-health",
            deep.get("status") == "ok" and not failed,
            f"status={deep.get('status')}, проверок={len(checks)}, сбоев={len(failed)}",
        )
    except Exception as exc:
        step("deep-health", False, str(exc))

    if not login(client, args.user, args.password):
        if not login(client, os.getenv("SYSTEM_USERNAME", "system"), os.getenv("SYSTEM_PASSWORD", "")):
            step("login", False, "неверные учётные данные")
            return finish(args, health)

    me = client.get("/api/me").json()
    ROLE = me.get("role", "")
    step("me", ROLE in ("admin", "system"), f"role={ROLE}")

    try:
        data = client.get("/api/admin/backup/status").json()
        step(
            "backup-status",
            "count" in data and "latest" in data,
            f"копий={data.get('count')}, каталог={data.get('dir')}",
        )
    except Exception as exc:
        step("backup-status", False, str(exc))

    try:
        data = client.get("/api/admin/settings").json()
        items = data.get("items", [])
        editable = [i for i in items if i.get("editable")]
        read_only = [i for i in items if not i.get("editable")]
        consistency = client.patch("/api/admin/settings/params", json={"values": {}}).json()
        warnings = consistency.get("warnings", [])
        step(
            "settings",
            len(items) > 0 and consistency.get("ok") is True,
            f"всего={len(items)}, редактируемых={len(editable)}, только чтение={len(read_only)}, "
            f"предупреждений={len(warnings)}",
        )
    except Exception as exc:
        step("settings", False, str(exc))

    models: dict = {}
    models_status = 0
    try:
        response = client.get("/api/admin/models")
        models_status = response.status_code
        models = response.json() if response.status_code == 200 else {}
        ollama = models.get("ollama", {})
        expected = 200 if ROLE == "system" else 403
        step(
            "models",
            response.status_code == expected,
            f"HTTP {response.status_code} (ожидалось {expected}), ollama={ollama.get('available')}, "
            f"моделей={len(ollama.get('models', []))}",
        )
    except Exception as exc:
        step("models", False, str(exc))

    domains_status = 0
    try:
        response = client.get("/api/admin/domain")
        domains_status = response.status_code
        expected = 200 if ROLE in ("system", "admin") else 403
        profiles = response.json().get("profiles", []) if response.status_code == 200 else []
        step(
            "domains",
            response.status_code == expected,
            f"HTTP {response.status_code} (ожидалось {expected}), направлений={len(profiles)}",
        )
    except Exception as exc:
        step("domains", False, str(exc))

    try:
        status = client.get("/reindex").json()
        step("reindex-status", bool(status), f"status={status.get('status')}, done={status.get('done')}/{status.get('total')}")
    except Exception as exc:
        step("reindex-status", False, str(exc))

    try:
        info = client.get("/collection").json()
        points = int(info.get("points") or info.get("points_count") or 0)
        step("collection", points > 0, f"points={points}, target={info.get('target')}")
    except Exception as exc:
        step("collection", False, str(exc))

    results_count = 0
    try:
        result = client.post("/retrieve", json={"query": "установка и настройка системы", "top_k": 5}).json()
        results_count = len(result.get("results", []))
        step("retrieve", results_count > 0, f"найдено фрагментов={results_count}")
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
    grounded = 0
    for question in questions:
        try:
            payload = client.post("/ask", json={"question": question}).json()
            answer = (payload.get("answer") or "").strip()
            if answer:
                answers_ok += 1
            if payload.get("grounded"):
                grounded += 1
            if payload.get("request_id"):
                request_ids.append(payload["request_id"])
        except Exception as exc:
            step(f"ask: {question[:40]}", False, str(exc))
    step(
        "ask",
        answers_ok == len(questions),
        f"ответов={answers_ok}/{len(questions)}, по документации={grounded}, модель={active_model}",
    )

    if request_ids:
        try:
            rated = client.post(f"/api/rate/{request_ids[0]}", json={"vote": "up", "comment": "super-test"}).json()
            step("rate", bool(rated.get("ok")), f"request_id={request_ids[0]}")
        except Exception as exc:
            step("rate", False, str(exc))
    else:
        step("rate", False, "нет request_id для оценки")

    for label, path in (
        ("autolearn", "/api/admin/autolearn"),
        ("questiongen", "/api/admin/questiongen"),
        ("bootstrap", "/api/admin/bootstrap"),
        ("generated-questions", "/api/admin/questiongen/generated?limit=5"),
        ("question-clusters", "/api/admin/questiongen/clusters"),
    ):
        try:
            data = client.get(path).json()
            step(label, isinstance(data, dict), "получен")
        except Exception as exc:
            step(label, False, str(exc))

    try:
        data = client.get("/api/admin/eval/history?limit=5").json()
        step("eval-history", isinstance(data.get("items"), list), f"записей={len(data.get('items', []))}")
    except Exception as exc:
        step("eval-history", False, str(exc))

    try:
        data = client.get("/api/admin/autolearn/calibration").json()
        step(
            "judge-calibration",
            "agreement" in data,
            f"оценок={data.get('total')}, согласие={data.get('agreement')}",
        )
    except Exception as exc:
        step("judge-calibration", False, str(exc))

    try:
        data = client.post("/api/admin/questiongen/similarity").json()
        step(
            "similarity",
            "analyzed" in data,
            f"analyzed={data.get('analyzed')}, clusters={data.get('clusters')}, duplicates={data.get('duplicates')}",
        )
    except Exception as exc:
        step("similarity", False, str(exc))

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

    if args.all_models and models_status == 200:
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

    return finish(args, health, {"role": ROLE, "models_status": models_status, "domains_status": domains_status})


def finish(args, health: dict, extra: dict | None = None) -> int:
    total = len(STEPS)
    failed = [s for s in STEPS if not s["ok"]]
    passed = total - len(failed)
    lines = [
        f"# Супер-тест {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"База: {args.base}",
        f"Роль: {(extra or {}).get('role', '—')}",
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
