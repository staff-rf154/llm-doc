"""Исправления ответов админом: отдельная БД + векторный индекс для подбора."""

from qdrant_client.http import models

import appdb
import config
import settings
import vectorstore as vs


def _collection() -> str:
    return config.CORRECTIONS_COLLECTION


def ensure_collection() -> None:
    client = vs.get_qdrant()
    name = _collection()
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=config.EMBED_DIM,
            distance=models.Distance.COSINE,
        ),
    )


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
        payload = {
            "correction_id": correction_id,
            "question": row["question"],
            "answer": row["answer"],
            "doc_name": row["doc_name"],
            "doc_pages": row["doc_pages"],
            "doc_ref": row["doc_ref"],
            "author": row["author"],
            "updated_at": row["updated_at"],
        }
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


def search(question: str, limit: int = 3) -> list[dict]:
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
            with_payload=True,
        ).points
    except Exception:
        return []
    excluded: list[str] = []
    try:
        import exclusions

        if config.EXCLUSIONS_ENABLED:
            excluded = exclusions.patterns()
    except Exception:
        excluded = []
    return [
        {
            "correction_id": int((p.payload or {}).get("correction_id", p.id)),
            "question": (p.payload or {}).get("question", ""),
            "answer": (p.payload or {}).get("answer", ""),
            "doc_name": (p.payload or {}).get("doc_name", ""),
            "doc_pages": (p.payload or {}).get("doc_pages", ""),
            "doc_ref": (p.payload or {}).get("doc_ref", ""),
            "score": float(p.score),
        }
        for p in points
        if not any(
            (p.payload or {}).get("doc_name", "").lower().endswith(pattern)
            or pattern in (p.payload or {}).get("doc_name", "").lower()
            for pattern in excluded
        )
    ]


def match(question: str) -> tuple[dict | None, list[dict]]:
    """Точное исправление (порог схожести) и примеры для подсказки модели."""
    found = search(question, limit=max(1, settings.int_value("CORRECTION_EXAMPLES") + 1))
    exact = None
    examples: list[dict] = []
    for item in found:
        if exact is None and item["score"] >= settings.float_value("CORRECTION_MATCH_THRESHOLD"):
            exact = item
            continue
        if len(examples) < settings.int_value("CORRECTION_EXAMPLES"):
            examples.append(item)
    return exact, examples
