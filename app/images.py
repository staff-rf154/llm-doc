"""Встроенные изображения из PDF: извлечение, индексация и поиск по релевантности.

Изображения сохраняются в `data/images`, их подписи индексируются в отдельной
коллекции Qdrant `images` — так картинки находятся тем же эмбеддингом, что и текст.
"""

import hashlib
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from qdrant_client.http import models

import config
import settings
import vectorstore as vs


@dataclass
class ExtractedImage:
    source: str
    page: int
    index: int
    path: Path
    width: int
    height: int
    caption: str


def images_dir() -> Path:
    path = Path(config.IMAGES_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def collection_name() -> str:
    return config.IMAGES_COLLECTION


def ensure_collection() -> None:
    client = vs.get_qdrant()
    name = collection_name()
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=config.EMBED_DIM,
            distance=models.Distance.COSINE,
        ),
    )


def _caption_for(page_text: str, limit: int = 240) -> str:
    """Короткая подпись из текста рядом с изображением."""
    parts = [part.strip() for part in page_text.splitlines() if part.strip()]
    return " ".join(parts)[:limit]


def extract_from_pdf(
    path: Path,
    source: str,
    min_side: int,
    per_page: int = 2,
    per_document: int = 20,
) -> list[ExtractedImage]:
    """Крупные встроенные изображения документа.

    В мануалах много технических тайлов и повторяющихся фонов, поэтому отбрасываются
    мелкие картинки, полностраничные фоны и дубликаты по содержимому. На страницу
    и в документ попадает лишь несколько самых больших оставшихся картинок.
    """
    try:
        import pymupdf
    except ImportError:
        return []

    found: list[ExtractedImage] = []
    try:
        document = pymupdf.open(path)
    except Exception:
        return found

    target = images_dir()
    digests: set[str] = set()
    seen_xrefs: set[int] = set()
    collected: list[ExtractedImage] = []
    try:
        for page_number, page in enumerate(document, start=1):
            try:
                rect = page.rect
                page_area = max(1.0, rect.width * rect.height)
                candidates = []
                for info in page.get_images(full=True):
                    xref = info[0]
                    if xref in seen_xrefs:
                        continue
                    width = int(info[2]) if len(info) > 2 else 0
                    height = int(info[3]) if len(info) > 3 else 0
                    if width < min_side or height < min_side:
                        continue
                    if width * height >= page_area * float(
                        settings.value("IMAGE_SKIP_FULLPAGE", "0.6")
                    ):
                        seen_xrefs.add(xref)
                        continue
                    candidates.append((width * height, xref))
                if not candidates:
                    continue
                candidates.sort(reverse=True)
                caption = _caption_for(page.get_text("text") or "")
                for _, xref in candidates[:per_page]:
                    seen_xrefs.add(xref)
                    try:
                        payload = document.extract_image(xref)
                    except Exception:
                        continue
                    data = payload.get("image")
                    if not data:
                        continue
                    digest = hashlib.sha1(data).hexdigest()
                    if digest in digests:
                        continue
                    digests.add(digest)
                    image_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{digest}"))
                    ext = payload.get("ext", "png")
                    file_path = target / f"{image_id}.{ext}"
                    if not file_path.exists():
                        file_path.write_bytes(data)
                    collected.append(
                        ExtractedImage(
                            source=source,
                            page=page_number,
                            index=len(collected),
                            path=file_path,
                            width=payload.get("width", 0),
                            height=payload.get("height", 0),
                            caption=caption,
                        )
                    )
            except Exception:
                continue
    finally:
        try:
            document.close()
        except Exception:
            pass

    if len(collected) <= per_document:
        return collected

    step = len(collected) / per_document
    for i in range(per_document):
        found.append(collected[int(i * step)])
    return found


def _pdf_files() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for root in config.SOURCE_ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in sorted(root_path.rglob("*.pdf")):
            items.append((str(path.relative_to(root_path)), path))
    return items


def indexed_points() -> dict[str, dict]:
    client = vs.get_qdrant()
    name = collection_name()
    if not client.collection_exists(name):
        return {}
    result: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            result[str(point.id)] = {"id": str(point.id), "payload": payload}
        if offset is None:
            break
    return result


def index_images(progress=None) -> dict:
    """Извлечь и проиндексировать изображения всех PDF."""
    client = vs.get_qdrant()
    ensure_collection()
    existing = indexed_points()

    files = _pdf_files()
    total = 0
    kept = 0
    skipped = 0
    if progress:
        progress(done=0, total=len(files))

    current: set[str] = set()
    for position, (source, path) in enumerate(files, start=1):
        extracted = extract_from_pdf(
            path,
            source,
            int(settings.int_value("IMAGE_MIN_SIDE") or 80),
            per_page=int(settings.int_value("IMAGE_PER_PAGE") or 2),
            per_document=int(settings.int_value("IMAGE_PER_DOC") or 20),
        )
        total += len(extracted)
        for image in extracted:
            point_id = image.path.stem
            payload = {
                "source": image.source,
                "page": image.page,
                "file": image.path.name,
                "caption": image.caption,
                "width": image.width,
                "height": image.height,
            }
            current.add(point_id)
            stored = existing.get(point_id)
            if stored and stored["payload"] == payload:
                kept += 1
                continue
            vector = vs.embed_passages([image.caption or image.source])[0]
            client.upsert(
                collection_name=collection_name(),
                points=[
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                ],
                wait=True,
            )
            kept += 1
        if progress:
            progress(done=position, total=len(files), images=kept)
        skipped += 1

    stale = [pid for pid in existing if pid not in current]
    if stale:
        client.delete(
            collection_name=collection_name(),
            points_selector=models.PointIdsList(points=stale),
        )
    return {
        "files": len(files),
        "images": kept,
        "found": total,
        "removed": len(stale),
    }


def search(question: str, limit: int | None = None) -> list[dict]:
    """Картинки, релевантные вопросу."""
    client = vs.get_qdrant()
    name = collection_name()
    if not client.collection_exists(name):
        return []
    limit = limit or int(settings.int_value("IMAGE_MAX") or 2)
    try:
        points = client.query_points(
            collection_name=name,
            query=vs.embed_query(question),
            limit=limit,
            with_payload=True,
        ).points
    except Exception:
        return []
    threshold = float(settings.value("IMAGE_MATCH_THRESHOLD") or 0)
    items = []
    for point in points:
        score = float(point.score)
        if score < threshold:
            continue
        payload = point.payload or {}
        items.append(
            {
                "source": payload.get("source", ""),
                "page": payload.get("page", 0),
                "file": payload.get("file", ""),
                "caption": payload.get("caption", ""),
                "score": round(score, 4),
                "url": f"/api/images/{payload.get('file', '')}",
            }
        )
    return items


class ImageJob:
    """Фоновая индексация изображений с прогрессом."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.state: dict = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "done": 0,
            "total": 0,
            "images": 0,
            "found": 0,
            "removed": 0,
            "error": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _update(self, **fields) -> None:
        with self._lock:
            self.state.update(fields)

    def start(self) -> dict:
        with self._lock:
            if self.state["status"] == "running" and self._thread and self._thread.is_alive():
                return {**self.state, "started": False}
            self.state = {
                **self._idle(),
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return {**self.snapshot(), "started": True}

    def _run(self) -> None:
        try:
            result = index_images(progress=lambda **fields: self._update(**fields))
            self._update(
                status="done",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                **result,
            )
        except Exception as exc:
            self._update(
                status="error",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                error=f"{type(exc).__name__}: {exc}",
            )


job = ImageJob()