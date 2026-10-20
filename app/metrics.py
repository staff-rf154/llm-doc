"""Метрики Prometheus: HTTP-запросы, процесс и бизнес-показатели RAG."""

import time
from datetime import datetime, timedelta
from pathlib import Path

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    GCCollector,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily

import config

REGISTRY = CollectorRegistry(auto_describe=False)
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)
GCCollector(registry=REGISTRY)

HTTP_REQUESTS = Counter(
    "llm_http_requests_total",
    "HTTP-запросы к приложению",
    ["method", "path", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "llm_http_request_duration_seconds",
    "Длительность HTTP-запросов",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)
RETRIEVAL_LATENCY = Histogram(
    "llm_retrieval_latency_seconds",
    "Длительность поиска и переранжирования",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=REGISTRY,
)
REINDEX_DURATION = Histogram(
    "llm_reindex_duration_seconds",
    "Длительность полной переиндексации",
    buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600, 7200),
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Токены генерации",
    ["kind", "type"],
    registry=REGISTRY,
)


def observe_llm_usage(usage, kind: str = "chat") -> None:
    if usage is None:
        return
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0) or prompt + completion
    LLM_TOKENS.labels(kind, "prompt").inc(prompt)
    LLM_TOKENS.labels(kind, "completion").inc(completion)
    LLM_TOKENS.labels(kind, "total").inc(total)


async def middleware(request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        if not path.startswith("/api/images/"):
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(
                time.perf_counter() - started
            )


def _percentile(values, ratio):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(ratio * (len(ordered) - 1)))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def _iso_ago(**delta):
    return (datetime.now() - timedelta(**delta)).isoformat(timespec="seconds")


def _business_families():
    import appdb
    import domains
    import documents
    import ingest
    import settings
    import synonymdb
    import vectorstore

    families = []

    try:
        active = domains.current()
        info = GaugeMetricFamily(
            "llm_app_info",
            "Сведения о приложении",
            labels=["collection", "llm_model", "embedding_model", "device", "title"],
        )
        info.add_metric(
            [
                config.QDRANT_COLLECTION,
                settings.value("LLM_MODEL", config.LLM_MODEL),
                settings.value("EMBEDDING_MODEL", config.EMBEDDING_MODEL),
                vectorstore.device_info().get("embedder", "cpu"),
                active.get("title", ""),
            ],
            1,
        )
        families.append(info)

        active_metric = GaugeMetricFamily(
            "llm_domain_active",
            "Активное направление",
            labels=["slug", "title"],
        )
        active_metric.add_metric([active.get("slug", ""), active.get("title", "")], 1)
        families.append(active_metric)
    except Exception:
        pass

    try:
        conn = appdb.connect()
    except Exception:
        return families

    try:
        users = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END), 0) AS active "
            "FROM users"
        ).fetchone()
        metric = GaugeMetricFamily(
            "llm_users", "Пользователи", labels=["status"]
        )
        metric.add_metric(["total"], int(users["total"]))
        metric.add_metric(["active"], int(users["active"]))
        families.append(metric)

        req = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN grounded = 1 THEN 1 ELSE 0 END), 0) AS grounded, "
            "COALESCE(SUM(CASE WHEN corrected = 1 THEN 1 ELSE 0 END), 0) AS corrected, "
            "COALESCE(SUM(CASE WHEN origin = '' OR origin = 'web' THEN 1 ELSE 0 END), 0) AS web, "
            "COALESCE(SUM(CASE WHEN origin NOT IN ('', 'web') THEN 1 ELSE 0 END), 0) AS api "
            "FROM requests"
        ).fetchone()
        total_requests = int(req["total"])

        origins = GaugeMetricFamily(
            "llm_requests", "Запросы к ассистенту", labels=["origin"]
        )
        origins.add_metric(["web"], int(req["web"]))
        origins.add_metric(["api"], int(req["api"]))
        families.append(origins)

        quality = GaugeMetricFamily(
            "llm_answers",
            "Качество ответов",
            labels=["kind"],
        )
        quality.add_metric(["grounded"], int(req["grounded"]))
        quality.add_metric(["corrected"], int(req["corrected"]))
        quality.add_metric(
            ["ungrounded"], total_requests - int(req["grounded"])
        )
        families.append(quality)

        ratio = GaugeMetricFamily(
            "llm_grounded_ratio", "Доля ответов с опорой на документы"
        )
        ratio.add_metric(
            [], (int(req["grounded"]) / total_requests) if total_requests else 0.0
        )
        families.append(ratio)

        windows = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM requests WHERE created_at >= ?) AS h1, "
            "(SELECT COUNT(*) FROM requests WHERE created_at >= ?) AS h24, "
            "(SELECT COUNT(*) FROM requests WHERE created_at >= ?) AS d7",
            (_iso_ago(hours=1), _iso_ago(hours=24), _iso_ago(days=7)),
        ).fetchone()
        window_metric = GaugeMetricFamily(
            "llm_requests_recent", "Запросы за окно", labels=["window"]
        )
        window_metric.add_metric(["1h"], int(windows["h1"]))
        window_metric.add_metric(["24h"], int(windows["h24"]))
        window_metric.add_metric(["7d"], int(windows["d7"]))
        families.append(window_metric)

        latency_rows = conn.execute(
            "SELECT latency_ms FROM requests ORDER BY id DESC LIMIT 1000"
        ).fetchall()
        latencies = [int(row["latency_ms"]) for row in latency_rows]
        latency_metric = GaugeMetricFamily(
            "llm_request_latency_ms",
            "Длительность ответа, мс",
            labels=["quantile"],
        )
        for label, q in (("0.5", 0.5), ("0.9", 0.9), ("0.95", 0.95), ("0.99", 0.99)):
            latency_metric.add_metric([label], _percentile(latencies, q))
        latency_metric.add_metric(
            ["avg"], (sum(latencies) / len(latencies)) if latencies else 0.0
        )
        families.append(latency_metric)

        ratings = conn.execute(
            "SELECT vote, COUNT(*) AS n FROM ratings GROUP BY vote"
        ).fetchall()
        rating_metric = GaugeMetricFamily(
            "llm_ratings", "Оценки пользователей", labels=["vote"]
        )
        rating_values = {row["vote"]: int(row["n"]) for row in ratings}
        rating_metric.add_metric(["up"], rating_values.get("up", 0))
        rating_metric.add_metric(["down"], rating_values.get("down", 0))
        families.append(rating_metric)

        unrated = conn.execute(
            "SELECT COUNT(*) AS n FROM requests r "
            "LEFT JOIN ratings t ON t.request_id = r.id "
            "WHERE t.request_id IS NULL"
        ).fetchone()
        unrated_metric = GaugeMetricFamily(
            "llm_requests_unrated", "Запросы без оценки"
        )
        unrated_metric.add_metric([], int(unrated["n"]))
        families.append(unrated_metric)

        evals = conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM request_evals GROUP BY verdict"
        ).fetchall()
        eval_metric = GaugeMetricFamily(
            "llm_request_evals", "Оценки ответов автообучением", labels=["verdict"]
        )
        problems = 0
        for row in evals:
            count = int(row["n"])
            eval_metric.add_metric([row["verdict"]], count)
            if row["verdict"] in ("partial", "wrong", "refused_wrongly"):
                problems += count
        families.append(eval_metric)

        problem_metric = GaugeMetricFamily(
            "llm_eval_problems", "Проблемные ответы по автооценке"
        )
        problem_metric.add_metric([], problems)
        families.append(problem_metric)

        issues = conn.execute(
            "SELECT status, COUNT(*) AS n FROM answer_issues GROUP BY status"
        ).fetchall()
        issue_metric = GaugeMetricFamily(
            "llm_issues", "Замечания к ответам", labels=["status"]
        )
        for row in issues:
            issue_metric.add_metric([row["status"]], int(row["n"]))
        families.append(issue_metric)

        corrections = conn.execute(
            "SELECT status, COUNT(*) AS n FROM corrections GROUP BY status"
        ).fetchall()
        correction_metric = GaugeMetricFamily(
            "llm_corrections", "Исправления ответов", labels=["status"]
        )
        for row in corrections:
            correction_metric.add_metric([row["status"]], int(row["n"]))
        families.append(correction_metric)

        lessons = conn.execute(
            "SELECT status, COUNT(*) AS n FROM lessons GROUP BY status"
        ).fetchall()
        lesson_metric = GaugeMetricFamily(
            "llm_lessons", "Уроки-кандидаты", labels=["status"]
        )
        for row in lessons:
            lesson_metric.add_metric([row["status"]], int(row["n"]))
        families.append(lesson_metric)

        domain_row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN is_builtin = 1 THEN 1 ELSE 0 END), 0) AS builtin "
            "FROM domains"
        ).fetchone()
        domain_metric = GaugeMetricFamily(
            "llm_domains", "Направления", labels=["kind"]
        )
        domain_metric.add_metric(["total"], int(domain_row["total"]))
        domain_metric.add_metric(["builtin"], int(domain_row["builtin"]))
        families.append(domain_metric)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    try:
        count_metric = GaugeMetricFamily(
            "llm_synonyms", "Термины в словаре синонимов"
        )
        count_metric.add_metric([], synonymdb.count())
        families.append(count_metric)
    except Exception:
        pass

    try:
        total_docs = 0
        for root in config.SOURCE_ROOTS:
            path = Path(root)
            if not path.exists():
                continue
            total_docs += sum(
                1
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in documents.SUPPORTED
            )
        docs_metric = GaugeMetricFamily(
            "llm_documents", "Файлы документации", labels=["kind"]
        )
        docs_metric.add_metric(["total"], total_docs)
        try:
            docs_metric.add_metric(["excluded"], len(appdb.excluded_keys()))
        except Exception:
            docs_metric.add_metric(["excluded"], 0)
        families.append(docs_metric)
    except Exception:
        pass

    try:
        collection = vectorstore.target_name()
        info = vectorstore.collection_info(collection)
        points_metric = GaugeMetricFamily(
            "llm_qdrant_points", "Точки в коллекции Qdrant"
        )
        points_metric.add_metric([], int(info.get("points") or 0))
        families.append(points_metric)

        exists_metric = GaugeMetricFamily(
            "llm_collection_exists",
            "Коллекция Qdrant существует",
            labels=["collection"],
        )
        exists_metric.add_metric([collection], 1 if info.get("exists") else 0)
        families.append(exists_metric)
    except Exception:
        pass

    try:
        state = ingest.job.snapshot()
        status = state.get("status", "idle")
        done = int(state.get("done") or 0)
        total = int(state.get("total") or 0)
        running = GaugeMetricFamily("llm_ingest_running", "Индексация выполняется")
        running.add_metric([], 1 if status == "running" else 0)
        families.append(running)
        paused = GaugeMetricFamily("llm_ingest_paused", "Индексация на паузе")
        paused.add_metric([], 1 if status == "paused" else 0)
        families.append(paused)
        progress = GaugeMetricFamily(
            "llm_ingest_progress_ratio", "Доля обработанных фрагментов"
        )
        progress.add_metric([], (done / total) if total else 0.0)
        families.append(progress)
        counts = GaugeMetricFamily(
            "llm_ingest_chunks", "Фрагменты индексации", labels=["kind"]
        )
        counts.add_metric(["done"], done)
        counts.add_metric(["total"], total)
        counts.add_metric(["skipped"], int(state.get("skipped_files") or 0))
        counts.add_metric(["failed"], len(state.get("failed_files") or []))
        families.append(counts)
        status_metric = GaugeMetricFamily(
            "llm_ingest_status", "Состояние индексации", labels=["status"]
        )
        for name in ("idle", "running", "paused", "done", "error"):
            status_metric.add_metric([name], 1 if status == name else 0)
        families.append(status_metric)
    except Exception:
        pass

    return families


class BusinessCollector:
    def collect(self):
        yield from _business_families()


REGISTRY.register(BusinessCollector())


def render():
    return generate_latest(REGISTRY)
