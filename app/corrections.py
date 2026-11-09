"""Исправления ответов админом: отдельная БД + векторный индекс для подбора."""

import re

from qdrant_client.http import models

import appdb
import config
import settings
import tenancy
import vectorstore as vs


def _collection() -> str:
    return settings.value("CORRECTIONS_COLLECTION", config.CORRECTIONS_COLLECTION)


def ensure_collection(name: str | None = None, dim: int | None = None) -> None:
    vs.ensure_collection(name or _collection(), dim=dim, named=False, sparse=False)


def _payload(row: dict) -> dict:
    correction_id = int(row["id"])
    return {
        "correction_id": correction_id,
        "question": row["question"],
        "answer": row["answer"],
        "doc_name": row["doc_name"],
        "doc_pages": row["doc_pages"],
        "doc_ref": row["doc_ref"],
        "author": row["author"],
        "tenant_id": row.get("tenant_id") or tenancy.DEFAULT,
        "updated_at": row["updated_at"],
    }


def rebuild(spec: vs.EmbedSpec | None = None, name: str | None = None) -> int:
    """Полностью пересобрать коллекцию исправлений под указанную модель."""
    client = vs.get_qdrant()
    name = name or _collection()
    if client.collection_exists(name):
        client.delete_collection(name)
    ensure_collection(name, dim=spec.dim if spec else None)
    rows = appdb.list_corrections(status="active")
    for start in range(0, len(rows), 128):
        batch = rows[start : start + 128]
        points = [
            models.PointStruct(
                id=int(row["id"]),
                vector=vs.embed_query(row["question"], spec=spec),
                payload=_payload(row),
            )
            for row in batch
        ]
        if points:
            client.upsert(collection_name=name, points=points, wait=True)
    return len(rows)


def reindex() -> int:
    """Синхронизировать активные исправления из БД с векторным индексом."""
    client = vs.get_qdrant()
    name = _collection()
    ensure_collection()
    rows = appdb.list_corrections(status="active")
    keep = {int(row["id"]) for row in rows}

    stored: dict[int, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1024,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if "correction_id" in payload:
                stored[int(point.id)] = {"id": int(point.id), "payload": payload}
        if offset is None:
            break

    stale = [pid for pid in stored if pid not in keep]
    if stale:
        client.delete(collection_name=name, points_selector=models.PointIdsList(points=stale))

    for row in rows:
        correction_id = int(row["id"])
        payload = _payload(row)
        current = stored.get(correction_id)
        if current and current["payload"] == payload:
            continue
        vector = vs.embed_query(row["question"])
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=correction_id,
                    vector=vector,
                    payload=payload,
                )
            ],
            wait=True,
        )
    return len(rows)


def deactivate(correction_id: int) -> None:
    client = vs.get_qdrant()
    if client.collection_exists(_collection()):
        client.delete(
            collection_name=_collection(),
            points_selector=models.PointIdsList(points=[int(correction_id)]),
        )


def search(question: str, limit: int = 3, tenant: str | None = None) -> list[dict]:
    """Похожие вопросы с исправленными ответами."""
    client = vs.get_qdrant()
    name = _collection()
    if not client.collection_exists(name):
        return []
    try:
        points = client.query_points(
            collection_name=name,
            query=vs.embed_query(question),
            limit=limit,
            query_filter=tenancy.qdrant_filter(tenant),
            with_payload=True,
        ).points
    except Exception:
        return []
    return [
        {
            "correction_id": int((p.payload or {}).get("correction_id", p.id)),
            "question": (p.payload or {}).get("question", ""),
            "answer": (p.payload or {}).get("answer", ""),
            "doc_name": (p.payload or {}).get("doc_name", ""),
            "doc_pages": (p.payload or {}).get("doc_pages", ""),
            "doc_ref": (p.payload or {}).get("doc_ref", ""),
            "author": (p.payload or {}).get("author", ""),
            "updated_at": (p.payload or {}).get("updated_at", ""),
            "score": float(p.score),
        }
        for p in points
    ]


STOP_WORDS = {
    "как", "что", "где", "когда", "какой", "какая", "какие", "это", "для", "или",
    "и", "в", "на", "с", "по", "из", "не", "no", "the", "a", "of", "to", "имеет",
    "можно", "нужно", "надо", "его", "её", "их", "который", "которая",
}

STRONG_OVERLAP = 0.95


def keywords(text: str) -> set[str]:
    """Значимые слова вопроса без служебных."""
    return {
        word
        for word in re.findall(r"[\w-]+", (text or "").lower())
        if len(word) > 2 and word not in STOP_WORDS and not word.isdigit()
    }


def is_same_topic(question: str, stored_question: str, score: float) -> bool:
    """Отсечь ложные совпадения: вектор бывает близок при одном общем слове."""
    if score >= STRONG_OVERLAP:
        return True
    shared = keywords(question) & keywords(stored_question)
    needed = settings.int_value("CORRECTION_OVERLAP_WORDS")
    if not needed:
        needed = 2
    return len(shared) >= needed


def match(question: str, tenant: str | None = None) -> tuple[dict | None, list[dict]]:
    """Исправление для прямого ответа и примеры для подсказки модели."""
    found = search(
        question,
        limit=max(1, settings.int_value("CORRECTION_EXAMPLES") + 1),
        tenant=tenant,
    )
    exact = None
    examples: list[dict] = []
    threshold = settings.float_value("CORRECTION_MATCH_THRESHOLD")
    for item in found:
        if exact is None and item["score"] >= threshold and is_same_topic(
            question, item["question"], item["score"]
        ):
            exact = item
            continue
        if len(examples) < settings.int_value("CORRECTION_EXAMPLES"):
            examples.append(item)
    return exact, examples
