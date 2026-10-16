"""Стартовые промты хранятся только в базе (data/prompts/prompts.db) и правятся
администратором в кабинете: раздел «Стартовые промты».

В коде нет встроенного каталога — новые промты добавляются через API, поэтому
код не зависит от конкретных продуктов, доменов и имён файлов документации.
Адреса документации в промтах строятся из переменной окружения DOCS_BASE_URL.
"""

import time
from pathlib import Path

import config
import promptdb
import settings
import vectorstore as vs


def count() -> int:
    return promptdb.count_prompts()


_cache: dict = {"at": 0.0, "items": []}
_sources_cache: dict = {"at": 0.0, "sources": []}


def seed_if_empty() -> int:
    """Каталога в коде нет: возвращаем текущее количество промтов из базы."""
    return promptdb.count_prompts()


def indexed_sources() -> list[str]:
    now = time.time()
    if now - _sources_cache["at"] < config.PROMPTS_CACHE_TTL:
        return _sources_cache["sources"]
    client = vs.get_qdrant()
    target = vs.target_name()
    sources: list[str] = []
    seen: set[str] = set()
    if client.collection_exists(target):
        offset = None
        while len(sources) < 600:
            points, offset = client.scroll(
                collection_name=target,
                limit=1000,
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
    _sources_cache.update(at=now, sources=sources)
    return sources


def build_prompts() -> list[dict]:
    rows = promptdb.list_prompts()
    if not rows:
        return []
    haystack = " ".join(s.lower() for s in indexed_sources())
    matched = []
    unconditional = []
    for row in rows:
        keywords = row["keywords"].split()
        if not keywords:
            unconditional.append(row)
        elif any(k in haystack for k in keywords):
            matched.append(row)
    limit = settings.int_value("PROMPTS_MAX")
    selected = (matched + unconditional or rows)[:limit]
    return [
        {"text": r["text"], "product": r["product"], "source_url": r["source_url"]}
        for r in selected[:limit]
    ]


def get_prompts() -> list[dict]:
    now = time.time()
    if _cache["items"] and now - _cache["at"] < config.PROMPTS_CACHE_TTL:
        return _cache["items"]
    items = build_prompts()
    _cache.update(at=now, items=items)
    return items


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Показать стартовые промты из БД (каталога в коде нет)"
    )
    parser.add_argument("--replace", action="store_true", help="не поддерживается")
    args = parser.parse_args()
    if args.replace:
        print("каталог промтов хранится в БД и правится в кабинете, --replace не нужен")
    Path(config.PROMPTS_DB).parent.mkdir(parents=True, exist_ok=True)
    print(f"промтов в БД: {count()}")


if __name__ == "__main__":
    main()
