"""Разделение документации по платформам: Windows и Linux.

Часть материалов СёрчИнформ описана отдельно для каждой ОС (например,
документ с пометкой `_linux` и его версия без пометки для другой ОС).
Модуль определяет платформу документа, находит парный документ для другой ОС и
догружает из него фрагменты, чтобы ассистент мог дать сборный ответ.
"""

import re

import config
import vectorstore as vs

LINUX = "linux"
WINDOWS = "windows"

LABELS = {LINUX: "Linux", WINDOWS: "Windows"}

LINUX_WORDS = ("linux", "unix", "ubuntu", "debian", "centos", "rhel")
WINDOWS_WORDS = ("windows", "win32", "win64", "win7", "win10", "win11")

PLATFORM_MARKERS = (
    "linux",
    "unix",
    "ubuntu",
    "debian",
    "centos",
    "rhel",
    "windows",
    "win32",
    "win64",
)

LINUX_RE = re.compile(
    r"(?:^|[_\-\s.])(?:on[_\-])?(linux|unix|ubuntu|debian|centos|rhel)(?:$|[_\-\s.])",
    re.IGNORECASE,
)
WINDOWS_RE = re.compile(
    r"(?:^|[_\-\s.])(?:on[_\-])?(windows|win32|win64|msi)(?:$|[_\-\s.])",
    re.IGNORECASE,
)
MARKER_RE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-я])(?:on[_\-])?(?:"
    + "|".join(PLATFORM_MARKERS)
    + r")(?![0-9A-Za-zА-Яа-я])",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[^0-9A-Za-zА-Яа-я]+")


def platform_of_source(source: str) -> str | None:
    """Платформа документа по имени файла: None, если документ общий."""
    name = (source or "").rsplit("/", 1)[-1]
    if LINUX_RE.search(name):
        return LINUX
    if WINDOWS_RE.search(name):
        return WINDOWS
    return None


def question_platforms(question: str) -> set[str]:
    """Платформы, прямо названные в вопросе пользователя."""
    text = (question or "").lower()
    found: set[str] = set()
    if any(word in text for word in LINUX_WORDS):
        found.add(LINUX)
    if any(word in text for word in WINDOWS_WORDS):
        found.add(WINDOWS)
    return found


def _tokens(source: str) -> frozenset[str]:
    stem = (source or "").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    cleaned = MARKER_RE.sub(" ", stem)
    return frozenset(token.lower() for token in TOKEN_RE.split(cleaned) if token)


def counterpart(source: str, known: list[str] | None = None) -> str | None:
    """Парный документ для другой платформы: X_linux → X (общий/Windows) и наоборот."""
    platform = platform_of_source(source)
    if not platform:
        return None
    if known is None:
        try:
            known = vs.known_sources()
        except Exception:
            return None
    tokens = _tokens(source)
    if not tokens:
        return None
    fallback: str | None = None
    for name in known:
        other = platform_of_source(name)
        if other == platform:
            continue
        if _tokens(name) != tokens:
            continue
        if other:
            return name
        if fallback is None:
            fallback = name
    return fallback


def pairs_for(sources: list[str], known: list[str] | None = None) -> dict[str, str]:
    """Соответствие «документ платформы A → документ платформы B»."""
    if known is None:
        try:
            known = vs.known_sources()
        except Exception:
            return {}
    result: dict[str, str] = {}
    for source in sources:
        if platform_of_source(source):
            other = counterpart(source, known)
            if other:
                result[source] = other
    return result


def extra_chunks(query: str, sources: list[str], limit: int = 2, max_total: int = 6) -> list:
    """Фрагменты из документов второй платформы — параллели Windows/Linux."""
    from qdrant_client.http import models

    import rag

    if not sources:
        return []
    try:
        client = vs.get_qdrant()
        target = vs.target_name()
        if not client.collection_exists(target):
            return []
        mapping = pairs_for(sources)
        if not mapping:
            return []
        found: list[tuple[float, str, dict]] = []
        for name in dict.fromkeys(mapping.values()):
            points = client.query_points(
                collection_name=target,
                query=vs.embed_query(query),
                limit=limit,
                with_payload=True,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=name),
                        )
                    ]
                ),
            ).points
            for point in points:
                payload = point.payload or {}
                found.append((float(point.score), name, payload))
        found.sort(key=lambda item: item[0], reverse=True)

        chunks = []
        for score, _, payload in found[:max_total]:
            chunks.append(
                rag.RetrievedChunk(
                    text=payload.get("text", ""),
                    source=payload.get("source", "unknown"),
                    chunk=int(payload.get("chunk", 0)),
                    score=score,
                    url=payload.get("url", ""),
                    title=payload.get("title", ""),
                    page=int(payload.get("page", 0) or 0),
                    kind=payload.get("kind", "file") or "file",
                )
            )
        return chunks
    except Exception:
        return []


def platform_label(chunk) -> str:
    """Подпись платформы для фрагмента."""
    platform = platform_of_source(chunk.source)
    if platform:
        return LABELS[platform]
    if (chunk.kind or "") == "web":
        return ""
    return "общая документация"


def enabled() -> bool:
    return config.PLATFORM_AWARE