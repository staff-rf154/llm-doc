import threading
from functools import lru_cache
from typing import NamedTuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

import config
import settings

EMBED_LOCK = threading.Lock()
SWITCH_LOCK = threading.RLock()


class EmbedSpec(NamedTuple):
    model: str
    dim: int
    device: str = ""


def embedding_dim() -> int:
    return settings.int_value("EMBED_DIM") or config.EMBED_DIM


def resolve_device(pref: str) -> str:
    """Выбор устройства torch: auto → cuda при наличии, иначе cpu."""
    pref = (pref or "auto").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def device_info() -> dict:
    info = {
        "cuda": False,
        "torch": "",
        "cuda_version": "",
        "gpu": "",
        "embedder": resolve_device(settings.value("EMBEDDER_DEVICE", config.EMBEDDER_DEVICE)),
        "reranker": resolve_device(settings.value("RERANKER_DEVICE", config.RERANKER_DEVICE)),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_version"] = torch.version.cuda or ""
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def sparse_enabled() -> bool:
    """Гибридный поиск включён и настроен на слияние rrf."""
    if not settings.bool_value("HYBRID_SPARSE", config.HYBRID_SPARSE):
        return False
    return (settings.value("HYBRID_FUSION", config.HYBRID_FUSION) or "rrf").lower() == "rrf"


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60,
    )


@lru_cache(maxsize=4)
def get_embedder_for(name: str, device: str = "") -> SentenceTransformer:
    model = SentenceTransformer(name, device=resolve_device(device))
    model.max_seq_length = 512
    return model


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return get_embedder_for(
        settings.value("EMBEDDING_MODEL", config.EMBEDDING_MODEL),
        settings.value("EMBEDDER_DEVICE", config.EMBEDDER_DEVICE),
    )


def clear_models() -> None:
    get_embedder.cache_clear()
    get_embedder_for.cache_clear()
    get_sparse_model.cache_clear()
    get_reranker.cache_clear()


@lru_cache(maxsize=1)
def get_sparse_model(name: str = ""):
    """BM25-модель fastembed для точного поиска по терминам (sparse-вектора)."""
    from fastembed.sparse import SparseTextEmbedding

    return SparseTextEmbedding(
        model_name=name or settings.value("SPARSE_MODEL", config.SPARSE_MODEL)
    )


@lru_cache(maxsize=1)
def get_reranker(name: str = "", max_length: int = 0):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        name or settings.value("RERANKER_MODEL", config.RERANKER_MODEL),
        device=resolve_device(settings.value("RERANKER_DEVICE", config.RERANKER_DEVICE)),
        max_length=int(
            max_length
            or settings.int_value("RERANKER_MAX_LENGTH")
            or config.RERANKER_MAX_LENGTH
        ),
    )


def embed_sparse(texts: list[str], name: str = "") -> list[dict]:
    """BM25-вектора fastembed: {индексы: [...], значения: [...]} для Qdrant."""
    if not texts or not sparse_enabled():
        return []
    vectors = []
    for item in get_sparse_model(name).embed(texts):
        vectors.append({"indices": item.indices.tolist(), "values": item.values.tolist()})
    return vectors


def sparse_query(text: str, name: str = ""):
    """BM25-вектор запроса для поиска по sparse-индексу."""
    if not sparse_enabled():
        return None
    item = next(iter(get_sparse_model(name).query_embed([text])))
    return models.SparseVector(indices=item.indices.tolist(), values=item.values.tolist())


def supports_sparse(name: str) -> bool:
    """Есть ли в коллекции sparse-индекс (новые коллекции после переиндексации)."""
    client = get_qdrant()
    try:
        info = client.get_collection(name)
    except Exception:
        return False
    return "sparse" in (info.config.params.sparse_vectors or {})


def _embedder(spec: EmbedSpec | None) -> SentenceTransformer:
    if spec is None:
        return get_embedder()
    return get_embedder_for(
        spec.model,
        spec.device or settings.value("EMBEDDER_DEVICE", config.EMBEDDER_DEVICE),
    )


def embed_passages(texts: list[str], spec: EmbedSpec | None = None) -> list[list[float]]:
    prefixed = [settings.value("E5_PASSAGE_PREFIX", config.E5_PASSAGE_PREFIX) + t for t in texts]
    with EMBED_LOCK:
        vectors = _embedder(spec).encode(
            prefixed,
            batch_size=settings.int_value("EMBED_BATCH") or config.EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=len(prefixed) > 200,
        )
    return vectors.tolist()


def embed_query(text: str, spec: EmbedSpec | None = None) -> list[float]:
    with EMBED_LOCK:
        vector = _embedder(spec).encode(
            [settings.value("E5_QUERY_PREFIX", config.E5_QUERY_PREFIX) + text],
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


def resolve_alias(alias_name: str) -> str:
    """Куда указывает алиас; если алиаса нет — возвращает само имя."""
    client = get_qdrant()
    try:
        aliases = client.get_aliases().aliases
    except Exception:
        return alias_name
    for alias in aliases:
        if alias.alias_name == alias_name:
            return alias.collection_name
    return alias_name


def target_name() -> str:
    """Имя коллекции, по которой сейчас идёт поиск: алиас или сама коллекция."""
    return resolve_alias(config.QDRANT_COLLECTION)


def _collection_dim(info) -> int | None:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        params = vectors.get("dense")
        return int(params.size) if params is not None else None
    return int(vectors.size) if vectors is not None else None


def ensure_collection(
    name: str,
    recreate: bool = False,
    dim: int | None = None,
    named: bool = True,
    sparse: bool = True,
) -> None:
    client = get_qdrant()
    size = int(dim or embedding_dim())
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if exists:
        current = _collection_dim(client.get_collection(name))
        if current and current != size:
            raise ValueError(
                f"Коллекция {name} имеет размерность {current}, ожидается {size}. "
                "Нужна переиндексация под новую модель."
            )
        return
    sparse_config = None
    if sparse and sparse_enabled():
        sparse_config = {
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    dense_config = models.VectorParams(
        size=size,
        distance=models.Distance.COSINE,
    )
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": dense_config} if named else dense_config,
        sparse_vectors_config=sparse_config,
    )


def swap_alias(alias_name: str, new_name: str) -> str:
    """Переключить алиас на новую коллекцию, старую коллекцию удалить."""
    client = get_qdrant()
    previous = resolve_alias(alias_name)
    if client.collection_exists(alias_name):
        client.delete_collection(alias_name)
    actions = [
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                alias_name=alias_name, collection_name=new_name
            )
        )
    ]
    client.update_collection_aliases(actions)
    if previous not in (new_name, alias_name) and client.collection_exists(previous):
        client.delete_collection(previous)
    return previous


def swap_target(new_name: str) -> str:
    """Переключить алиас QDRANT_COLLECTION на новую коллекцию, старую удалить."""
    return swap_alias(config.QDRANT_COLLECTION, new_name)


def source_filter(source: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
    )


def count_source_points(source: str, collection: str | None = None) -> int:
    """Сколько точек источника лежит в коллекции."""
    name = collection or target_name()
    client = get_qdrant()
    if not client.collection_exists(name):
        return 0
    result = client.count(collection_name=name, count_filter=source_filter(source), exact=True)
    return int(result.count)


def delete_source_points(source: str, collection: str | None = None) -> int:
    """Удалить из коллекции все точки одного источника, вернуть их количество."""
    name = collection or target_name()
    client = get_qdrant()
    if not client.collection_exists(name):
        return 0
    total = count_source_points(source, name)
    if not total:
        return 0
    client.delete(
        collection_name=name,
        points_selector=models.FilterSelector(filter=source_filter(source)),
        wait=True,
    )
    return total


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

def drop_all_collections() -> dict:
    """Удалить все коллекции документации, исправлений, уроков и алиасы.

    Используется при полном сбросе индекса из кабинета.
    """
    client = get_qdrant()
    import config as cfg

    wanted = {
        cfg.QDRANT_COLLECTION,
        cfg.CORRECTIONS_COLLECTION,
        cfg.LESSONS_COLLECTION,
    }
    dropped: list[str] = []
    for item in client.get_collections().collections:
        name = item.name
        if name in wanted or name.startswith(f"{cfg.QDRANT_COLLECTION}_v"):
            client.delete_collection(collection_name=name)
            dropped.append(name)
    for alias in (cfg.QDRANT_COLLECTION,):
        try:
            client.delete_collection(collection_name=alias)
            if alias not in dropped:
                dropped.append(alias)
        except Exception:
            pass
    return {"dropped": dropped, "count": len(dropped)}
