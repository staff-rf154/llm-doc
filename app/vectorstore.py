import threading
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

import config

EMBED_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60,
    )


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    model.max_seq_length = 512
    return model


def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [config.E5_PASSAGE_PREFIX + t for t in texts]
    with EMBED_LOCK:
        vectors = get_embedder().encode(
            prefixed,
            batch_size=config.EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=len(prefixed) > 200,
        )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    with EMBED_LOCK:
        vector = get_embedder().encode(
            [config.E5_QUERY_PREFIX + text],
            normalize_embeddings=True,
        )
    return vector[0].tolist()


def orphan_collections() -> list[str]:
    """Версионированные коллекции, на которые больше не ссылается алиас."""
    client = get_qdrant()
    alias_target = target_name()
    prefix = f"{config.QDRANT_COLLECTION}_v"
    names = []
    for collection in client.get_collections().collections:
        if not collection.name.startswith(prefix) or collection.name == alias_target:
            continue
        if client.get_collection(collection.name).points_count:
            names.append(collection.name)
    return sorted(names)


def loaded_chunks(name: str) -> dict[str, set[int]]:
    """Карта source -> уже загруженные номера чанков (для возобновления индексации)."""
    client = get_qdrant()
    if not client.collection_exists(name):
        return {}
    result: dict[str, set[int]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=2048,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            source = payload.get("source")
            if source is None:
                continue
            result.setdefault(source, set()).add(int(payload.get("chunk", 0)))
        if offset is None:
            break
    return result


def known_sources(limit: int = 5000) -> list[str]:
    """Имена всех документов в активной коллекции."""
    client = get_qdrant()
    target = target_name()
    if not client.collection_exists(target):
        return []
    sources: list[str] = []
    seen: set[str] = set()
    offset = None
    while len(sources) < limit:
        points, offset = client.scroll(
            collection_name=target,
            limit=2048,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            source = (point.payload or {}).get("source")
            if source and source not in seen:
                seen.add(source)
                sources.append(source)
        if offset is None:
            break
    return sources


def target_name() -> str:
    """Имя коллекции, по которой сейчас идёт поиск: алиас или сама коллекция."""
    client = get_qdrant()
    try:
        aliases = client.get_aliases().aliases
    except Exception:
        return config.QDRANT_COLLECTION
    for alias in aliases:
        if alias.alias_name == config.QDRANT_COLLECTION:
            return alias.collection_name
    return config.QDRANT_COLLECTION


def ensure_collection(name: str, recreate: bool = False) -> None:
    client = get_qdrant()
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if exists:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=config.EMBED_DIM,
            distance=models.Distance.COSINE,
        ),
    )


def swap_target(new_name: str) -> str:
    """Переключить алиас QDRANT_COLLECTION на новую коллекцию, старую удалить."""
    client = get_qdrant()
    previous = target_name()
    alias_name = config.QDRANT_COLLECTION
    actions = []
    if client.collection_exists(alias_name):
        client.delete_collection(alias_name)
    actions.append(models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name=alias_name, collection_name=new_name)))
    client.update_collection_aliases(actions)
    if previous not in (new_name, alias_name) and client.collection_exists(previous):
        client.delete_collection(previous)
    return previous


def collection_info(name: str | None = None) -> dict:
    client = get_qdrant()
    name = name or config.QDRANT_COLLECTION
    if not client.collection_exists(name):
        return {"exists": False, "points": 0, "collection": name}
    info = client.get_collection(name)
    return {
        "exists": True,
        "points": info.points_count,
        "status": str(info.status),
        "collection": name,
    }