"""Управление документацией из кабинета: дерево папок, загрузка файлов,
скачивание по ссылкам и удаление.

Вся документация живёт в одном корне `DOCS_DIR` (`/data/docs`), который
доступен на запись. Каталог `UPLOADS_DIR` (`/data/docs/upload`) — подпапка
корня, куда по умолчанию складываются загруженные и скачанные файлы. Индекс
обновляется точечно: файл переиндексируется в активную коллекцию, а при
удалении его точки удаляются из индекса.
"""

import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import config
import index_state
import settings
import vectorstore as vs
from loaders import (
    FORMAT_LABELS,
    SUPPORTED,
    load_document,
    suffix_for_content_type,
)

_LOCK = threading.Lock()

SAFE_SEGMENT_RE = re.compile(r"[^\w\-. ]+", re.UNICODE)


def _max_mb() -> int:
    return settings.int_value("UPLOAD_MAX_MB") or config.UPLOAD_MAX_MB


def _url_timeout() -> int:
    return settings.int_value("UPLOAD_URL_TIMEOUT") or config.UPLOAD_URL_TIMEOUT


def docs_root() -> Path:
    root = Path(config.DOCS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def uploads_root() -> Path:
    root = Path(config.UPLOADS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_segment(name: str) -> str:
    """Один безопасный компонент пути."""
    cleaned = SAFE_SEGMENT_RE.sub("_", (name or "").strip().replace("\\", "/").split("/")[-1])
    cleaned = cleaned.strip(" ._") or "document"
    return cleaned[:180]


def safe_name(name: str) -> str:
    """Относительный путь без выхода за пределы корня документации.

    Подпапки сохраняются: `manuals/guide.pdf` остаётся путём, а не базовым
    именем файла. Пустые и «опасные» сегменты отбрасываются.
    """
    parts = [safe_segment(part) for part in (name or "").replace("\\", "/").split("/") if part.strip()]
    parts = [part for part in parts if part not in (".", "..")]
    return "/".join(parts)


def extension_of(name: str) -> str:
    return Path(name).suffix.lower()


def check_extension(name: str) -> str:
    suffix = extension_of(name)
    if suffix not in SUPPORTED:
        raise ValueError(
            "Поддерживаются только файлы: " + ", ".join(sorted(SUPPORTED))
        )
    return suffix


def resolve(rel: str) -> Path:
    """Абсолютный путь по относительному, с проверкой выхода за корень."""
    root = docs_root().resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Путь выходит за пределы каталога документации")
    return target


def is_uploaded(rel: str) -> bool:
    """Файл лежит в каталоге загрузок — только такие можно удалять с диска."""
    try:
        target = resolve(rel)
    except ValueError:
        return False
    uploads = uploads_root().resolve()
    return uploads in target.parents or target.parent == uploads


def name_exists(rel: str) -> bool:
    return resolve(rel).exists()


def unique_path(rel: str) -> Path:
    """Путь без конфликта имён: при занятости добавляет `_2`, `_3`…"""
    candidate = resolve(rel)
    if not candidate.exists():
        return candidate
    stem, suffix = Path(rel).stem, Path(rel).suffix
    parent = str(Path(rel).parent)
    parent = "" if parent == "." else parent + "/"
    for number in range(2, 1000):
        candidate = resolve(f"{parent}{stem}_{number}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError("Не удалось подобрать свободное имя файла")


def save_bytes(filename: str, content: bytes, folder: str = "") -> dict:
    """Сохранить загруженный файл в каталог загрузок."""
    rel = safe_name(filename)
    if folder:
        rel = f"{safe_name(folder)}/{rel}"
    if not rel:
        raise ValueError("Не удалось определить имя файла")
    check_extension(rel)
    limit = _max_mb() * 1024 * 1024
    if len(content) > limit:
        raise ValueError(f"Файл больше {_max_mb()} МБ")
    if not content:
        raise ValueError("Файл пустой")
    target = resolve(rel)
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        path = unique_path(rel)
        path.write_bytes(content)
    document = load_document(path)
    if not document.get("text", "").strip():
        path.unlink(missing_ok=True)
        raise ValueError("Файл не содержит читаемого текста")
    return {
        "name": str(path.relative_to(docs_root())),
        "path": str(path),
        "size": path.stat().st_size,
    }


def name_from_url(url: str) -> str:
    path = urlparse(url).path
    name = unquote(Path(path).name) or "document"
    if not extension_of(name):
        name += ".pdf"
    return safe_name(name)


def _download(url: str, client) -> dict:
    """Одна ссылка: скачать файл и сохранить в каталог загрузок."""
    response = client.get(url)
    response.raise_for_status()
    content = response.content
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type in ("text/html", "application/xhtml+xml") and not extension_of(
        Path(unquote(urlparse(url).path)).name
    ):
        raise ValueError("По ссылке отдаётся HTML-страница — нужна прямая ссылка на файл")
    limit = _max_mb() * 1024 * 1024
    if len(content) > limit:
        raise ValueError(f"Файл больше {_max_mb()} МБ")
    if not content:
        raise ValueError("Сервер вернул пустой файл")

    name = safe_name(Path(unquote(urlparse(url).path)).name)
    suffix = extension_of(name)
    if suffix not in SUPPORTED:
        guessed = suffix_for_content_type(content_type)
        if not guessed:
            raise ValueError("Не удалось определить формат файла по ссылке")
        name = f"{Path(name).stem or 'document'}{guessed}"
    check_extension(name)

    target_dir = uploads_root()
    with _LOCK:
        path = unique_path(str(target_dir / name).replace(str(docs_root()) + "/", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    document = load_document(path)
    if not document.get("text", "").strip():
        path.unlink(missing_ok=True)
        raise ValueError("Файл не содержит читаемого текста")
    return {
        "name": str(path.relative_to(docs_root())),
        "path": str(path),
        "size": path.stat().st_size,
        "content_type": content_type,
    }


def fetch_url(url: str, folder: str = "") -> dict:
    """Скачать документ по http-ссылке в каталог загрузок (или указанную папку)."""
    import httpx

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Нужна ссылка http:// или https://")
    if not parsed.netloc:
        raise ValueError("В ссылке нет адреса сайта")
    with httpx.Client(
        follow_redirects=True,
        timeout=_url_timeout(),
        headers={"User-Agent": config.WEB_CRAWL_USER_AGENT},
    ) as client:
        result = _download(url.strip(), client)
    if folder:
        target = resolve(f"{safe_name(folder)}/{Path(result['name']).name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        path = unique_path(str(target).replace(str(docs_root()) + "/", ""))
        shutil.move(result["path"], path)
        result["name"] = str(path.relative_to(docs_root()))
        result["path"] = str(path)
        result["size"] = path.stat().st_size
    return result


def fetch_urls(text: str, folder: str = "") -> dict:
    """Скачать несколько ссылок, каждая с новой строки.

    Возвращает результаты по каждой ссылке: успешные — с данными файла,
    ошибочные — с текстом ошибки.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Укажите хотя бы одну ссылку на документ")
    results: list[dict] = []
    ok = 0
    failed = 0
    with httpx_client() as client:
        for url in lines:
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    raise ValueError("Нужна ссылка http:// или https://")
                item = _download(url, client)
                if folder:
                    target_rel = f"{safe_name(folder)}/{Path(item['name']).name}"
                    target = resolve(target_rel)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    path = unique_path(target_rel)
                    shutil.move(item["path"], path)
                    item["name"] = str(path.relative_to(docs_root()))
                    item["path"] = str(path)
                    item["size"] = path.stat().st_size
                item["ok"] = True
                ok += 1
            except Exception as exc:
                item = {"url": url, "ok": False, "error": str(exc)}
                failed += 1
            item.setdefault("url", url)
            results.append(item)
    return {"items": results, "ok": ok, "failed": failed}


def httpx_client():
    import httpx

    return httpx.Client(
        follow_redirects=True,
        timeout=_url_timeout(),
        headers={"User-Agent": config.WEB_CRAWL_USER_AGENT},
    )


def index_document(rel: str) -> dict:
    """Точечно проиндексировать документ из каталога документации."""
    import ingest

    rel = safe_name(rel)
    target = resolve(rel)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Файл не найден: {rel}")
    if target.suffix.lower() not in SUPPORTED:
        raise ValueError("Файл не поддерживается")
    return ingest.index_single(str(docs_root()), target, rel)


def _writable(rel: str) -> bool:
    try:
        resolve(rel)
        return True
    except (ValueError, OSError):
        return False


def delete_document(rel: str) -> dict:
    """Удалить документ: файл и точки из индекса."""
    rel = safe_name(rel)
    target = resolve(rel)
    collection = vs.target_name()
    removed_file = False
    if target.is_file():
        target.unlink()
        removed_file = True
    purged = vs.delete_source_points(rel, collection)
    index_state.forget_source(collection, f"{config.DOCS_DIR}|{rel}")
    import appdb

    appdb.remove_exclusion(config.DOCS_DIR, rel)
    return {
        "name": rel,
        "root": config.DOCS_DIR,
        "removed_file": removed_file,
        "purged_points": purged,
        "excluded": False,
    }


def restore_document(rel: str) -> dict:
    """Вернуть документ в индекс."""
    import appdb

    rel = safe_name(rel)
    appdb.remove_exclusion(config.DOCS_DIR, rel)
    path = resolve(rel)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {rel}")
    return index_document(rel)


def create_folder(rel: str) -> dict:
    """Создать папку в каталоге документации."""
    rel = safe_name(rel)
    if not rel:
        raise ValueError("Укажите имя папки")
    target = resolve(rel)
    target.mkdir(parents=True, exist_ok=True)
    uploads_root().mkdir(parents=True, exist_ok=True)
    return {"name": rel, "created": True}


def delete_folder(rel: str) -> dict:
    """Удалить папку вместе с содержимым."""
    rel = safe_name(rel)
    if not rel:
        raise ValueError("Укажите имя папки")
    target = resolve(rel)
    if not target.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {rel}")
    if target.resolve() in (docs_root().resolve(), uploads_root().resolve()):
        raise ValueError("Корневую папку удалить нельзя")
    removed = 0
    collection = vs.target_name()
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            rel_file = str(path.relative_to(docs_root()))
            removed += vs.delete_source_points(rel_file, collection)
            index_state.forget_source(collection, f"{config.DOCS_DIR}|{rel_file}")
    shutil.rmtree(target, ignore_errors=True)
    return {"name": rel, "removed_files": removed}


def list_folders() -> list[str]:
    """Все папки каталога документации относительно корня."""
    root = docs_root()
    folders: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            folders.append(str(path.relative_to(root)))
    return folders


def list_documents() -> list[dict]:
    """Файлы каталога документации и их состояние в индексе."""
    import appdb

    try:
        indexed = index_sources()
    except Exception:
        indexed = {}
    try:
        excluded = appdb.excluded_keys()
    except Exception:
        excluded = set()
    root = docs_root()
    items: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if path.name.startswith("."):
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue
        stat = path.stat()
        items.append(
            {
                "name": rel,
                "folder": str(Path(rel).parent).replace("\\", "/").strip("."),
                "root": config.DOCS_DIR,
                "editable": True,
                "can_delete": True,
                "excluded": (config.DOCS_DIR, rel) in excluded,
                "format": FORMAT_LABELS.get(path.suffix.lower(), path.suffix.lower().lstrip(".")),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "indexed": rel in indexed and (config.DOCS_DIR, rel) not in excluded,
                "chunks": indexed.get(rel, 0),
            }
        )
    return items


def index_sources() -> dict[str, int]:
    """Сколько чанков каждого источника лежит в активной коллекции."""
    try:
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
