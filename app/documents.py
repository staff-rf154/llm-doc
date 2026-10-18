"""Управление документацией из кабинета: список, загрузка файла, скачивание по
ссылке и удаление.

Загруженные и скачанные файлы складываются в отдельный каталог `UPLOADS_DIR`,
который входит в `config.SOURCE_ROOTS`, поэтому они участвуют и в обычной
переиндексации. После добавления или удаления документа индекс обновляется
точечно: файл переиндексируется в активную коллекцию, а при удалении его точки
удаляются из индекса.
"""

import re
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import config
import index_state
import settings
import vectorstore as vs
from loaders import SUPPORTED, load_document

_LOCK = threading.Lock()

FILE_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")
SAFE_NAME_RE = re.compile(r"[^\w\-. ]+", re.UNICODE)

FORMAT_LABELS = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".txt": "TXT",
    ".md": "Markdown",
    ".html": "HTML",
    ".htm": "HTML",
}


def _max_mb() -> int:
    return settings.int_value("UPLOAD_MAX_MB") or config.UPLOAD_MAX_MB


def _url_timeout() -> int:
    return settings.int_value("UPLOAD_URL_TIMEOUT") or config.UPLOAD_URL_TIMEOUT


def _uploads_root() -> Path:
    root = Path(config.UPLOADS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_name(name: str) -> str:
    """Имя файла без путей и небезопасных символов."""
    cleaned = SAFE_NAME_RE.sub("_", (name or "").strip().replace("\\\\", "/").split("/")[-1])
    cleaned = cleaned.strip(" ._") or "document"
    return cleaned[:180]


def extension_of(name: str) -> str:
    return Path(name).suffix.lower()


def check_extension(name: str) -> str:
    suffix = extension_of(name)
    if suffix not in FILE_EXTENSIONS:
        raise ValueError(
            "Поддерживаются только файлы: " + ", ".join(FILE_EXTENSIONS)
        )
    return suffix


def name_exists(name: str, root: Path) -> bool:
    """Есть ли такое имя в других корнях документации.

    Имя источника в индексе — путь относительно корня, поэтому одинаковые имена
    в разных корнях указывали бы на одни и те же точки Qdrant.
    """
    for base in config.SOURCE_ROOTS:
        base_path = Path(base)
        if base_path == root or not base_path.exists():
            continue
        if (base_path / name).exists():
            return True
    return False


def unique_path(root: Path, name: str) -> Path:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = root / name
    if not candidate.exists() and not name_exists(name, root):
        return candidate
    for number in range(2, 1000):
        candidate = root / f"{stem}_{number}{suffix}"
        if not candidate.exists() and not name_exists(candidate.name, root):
            return candidate
    raise ValueError("Не удалось подобрать свободное имя файла")


def save_bytes(filename: str, content: bytes) -> dict:
    """Сохранить загруженный файл в каталог загрузок."""
    name = safe_name(filename)
    check_extension(name)
    limit = _max_mb() * 1024 * 1024
    if len(content) > limit:
        raise ValueError(f"Файл больше {_max_mb()} МБ")
    if not content:
        raise ValueError("Файл пустой")
    root = _uploads_root()
    with _LOCK:
        path = unique_path(root, name)
        path.write_bytes(content)
    document = load_document(path)
    if not document.get("text", "").strip():
        path.unlink(missing_ok=True)
        raise ValueError("Файл не содержит читаемого текста")
    return {"name": path.name, "path": str(path), "size": path.stat().st_size}


def name_from_url(url: str) -> str:
    path = urlparse(url).path
    name = unquote(Path(path).name) or "document"
    if not extension_of(name):
        name += ".pdf"
    return safe_name(name)


def fetch_url(url: str) -> dict:
    """Скачать документ по http-ссылке и сохранить его в каталог загрузок."""
    import httpx

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Нужна ссылка http:// или https://")
    if not parsed.netloc:
        raise ValueError("В ссылке нет адреса сайта")
    limit = _max_mb() * 1024 * 1024
    with httpx.Client(
        follow_redirects=True,
        timeout=_url_timeout(),
        headers={"User-Agent": config.WEB_CRAWL_USER_AGENT},
    ) as client:
        response = client.get(url.strip())
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type in ("text/html", "application/xhtml+xml"):
        raise ValueError("По ссылке отдаётся HTML-страница — нужна прямая ссылка на файл")
    if len(content) > limit:
        raise ValueError(f"Файл больше {_max_mb()} МБ")
    if not content:
        raise ValueError("Сервер вернул пустой файл")

    name = safe_name(Path(unquote(urlparse(url.strip()).path)).name)
    suffix = extension_of(name)
    if suffix not in FILE_EXTENSIONS:
        if content_type in ("application/pdf",):
            name += ".pdf"
        else:
            guessed = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            if content_type == guessed:
                name += ".docx"
            else:
                name += ".pdf"
    check_extension(name)

    root = _uploads_root()
    with _LOCK:
        path = unique_path(root, name)
        path.write_bytes(content)
    document = load_document(path)
    if not document.get("text", "").strip():
        path.unlink(missing_ok=True)
        raise ValueError("Файл не содержит читаемого текста")
    return {
        "name": path.name,
        "path": str(path),
        "size": path.stat().st_size,
        "content_type": content_type,
    }


def index_document(name: str) -> dict:
    """Точечно проиндексировать документ из каталога загрузок."""
    import ingest

    root = _uploads_root()
    path = safe_name(name)
    target = root / path
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if target.suffix.lower() not in SUPPORTED:
        raise ValueError("Файл не поддерживается")
    return ingest.index_single(str(root), target, path)


def is_uploaded(name: str) -> bool:
    """Файл лежит в каталоге загрузок — только такие можно удалять."""
    root = _uploads_root()
    target = root / safe_name(name)
    try:
        return target.resolve().parent == root.resolve() and target.is_file()
    except OSError:
        return False


def _writable(root: str) -> bool:
    try:
        return Path(root).resolve() == Path(config.UPLOADS_DIR).resolve()
    except OSError:
        return False


def delete_document(name: str, root: str | None = None) -> dict:
    """Удалить документ: файл (если каталог доступен на запись) и точки из индекса.

    Для read-only каталогов (data/docs) файл остаётся, но документ исключается из
    индекса и из последующих переиндексаций.
    """
    root = root or config.UPLOADS_DIR
    base = Path(root)
    rel = safe_name(name)
    target = base / rel
    writable = _writable(root)
    if writable and not target.is_file():
        raise ValueError("Файл не найден в каталоге загрузок")
    collection = vs.target_name()
    removed_file = False
    if writable and target.is_file():
        target.unlink()
        removed_file = True
    purged = vs.delete_source_points(rel, collection)
    if purged == 0:
        purged = vs.delete_source_points(target.name, collection)
    index_state.forget_source(collection, f"{root}|{rel}")
    if not writable:
        import appdb

        appdb.add_exclusion(root, rel)
    else:
        import appdb

        appdb.remove_exclusion(root, rel)
    return {
        "name": rel,
        "root": root,
        "removed_file": removed_file,
        "purged_points": purged,
        "excluded": not writable,
    }


def restore_document(name: str, root: str | None = None) -> dict:
    """Вернуть исключённый документ в индекс."""
    import appdb

    root = root or config.DOCS_DIR
    rel = safe_name(name)
    appdb.remove_exclusion(root, rel)
    path = Path(root) / rel
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {rel}")
    return index_single_document(root, path, rel)


def index_single_document(root: str, path: Path, rel: str) -> dict:
    import ingest

    return ingest.index_single(root, path, rel)


def list_documents() -> list[dict]:
    """Файлы во всех корнях документации и их состояние в индексе."""
    import appdb

    try:
        indexed: dict[str, int] = index_sources()
    except Exception:
        indexed = {}
    try:
        excluded = appdb.excluded_keys()
    except Exception:
        excluded = set()
    items: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for root in config.SOURCE_ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in sorted(root_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            try:
                rel = str(path.relative_to(root_path))
            except ValueError:
                rel = path.name
            if path.name.startswith("."):
                continue
            stat = path.stat()
            writable = _writable(root)
            items.append(
                {
                    "name": rel,
                    "root": root,
                    "editable": writable,
                    "can_delete": True,
                    "excluded": (root, rel) in excluded,
                    "format": FORMAT_LABELS.get(path.suffix.lower(), path.suffix.lower()),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "indexed": (rel in indexed or path.name in indexed)
                    and (root, rel) not in excluded,
                    "chunks": indexed.get(rel, indexed.get(path.name, 0)),
                }
            )
            seen.add((root, rel))
    return items


def index_sources() -> dict[str, int]:
    """Сколько чанков каждого источника лежит в активной коллекции."""
    try:
        from qdrant_client.http import models

        client = vs.get_qdrant()
        target = vs.target_name()
        if not client.collection_exists(target):
            return {}
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=target,
                limit=2048,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                source = (point.payload or {}).get("source")
                if source:
                    counts[source] = counts.get(source, 0) + 1
            if offset is None:
                break
        return counts
    except Exception:
        return {}