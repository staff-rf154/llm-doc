import argparse
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models

import config
import index_state
import vectorstore as vs
from loaders import iter_documents, load_file


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def chunk_points(rel: str, chunks: list[str]) -> list[dict]:
    return [
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel}:{index}")),
            "text": chunk,
            "source": rel,
            "chunk": index,
        }
        for index, chunk in enumerate(chunks)
    ]


def plan_files() -> list[tuple[str, Path, str]]:
    files: list[tuple[str, Path, str]] = []
    for root in config.SOURCE_ROOTS:
        for path in iter_documents(root):
            rel = str(Path(path).relative_to(root))
            files.append((str(root), path, rel))
    return files


def iter_batches(points: list[dict], batch_size: int):
    for start in range(0, len(points), batch_size):
        yield points[start : start + batch_size]


def upsert_batch(client, collection: str, batch: list[dict]) -> None:
    vectors = vs.embed_passages([p["text"] for p in batch])
    client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=p["id"],
                vector=vector,
                payload={
                    "text": p["text"],
                    "source": p["source"],
                    "chunk": p["chunk"],
                },
            )
            for p, vector in zip(batch, vectors)
        ],
        wait=True,
    )


def collect_chunks() -> list[dict]:
    points: list[dict] = []
    for root, path, rel in plan_files():
        text = load_file(path)
        if not text:
            print(f"skip empty: {path}")
            continue
        chunks = split_text(text)
        points.extend(chunk_points(rel, chunks))
        print(f"{rel}: {len(chunks)} chunks")
    return points


class ReindexJob:
    """Фоновая переиндексация со срезом прогресса, паузой и возобновлением."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._gate = threading.Event()
        self._gate.set()
        self._thread: threading.Thread | None = None
        self.state: dict = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "paused_at": None,
            "done": 0,
            "total": 0,
            "skipped_files": 0,
            "failed_files": [],
            "resumed": False,
            "collection": None,
            "error": None,
            "result": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _update(self, **fields) -> None:
        with self._lock:
            self.state.update(fields)

    def start(self) -> dict:
        with self._lock:
            if self.state["status"] in ("running", "paused") and self._thread is not None:
                if self._thread.is_alive():
                    started = self._gate.is_set()
                    if not started:
                        self._gate.set()
                        self._update(status="running", paused_at=None)
                    return {**self.state, "started": started, "action": "resume"}
                return {**self.state, "started": False, "action": "restart"}
            self.state = {
                **self._idle(),
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        self._gate.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return {**self.snapshot(), "started": True, "action": "start"}

    def pause(self) -> dict:
        with self._lock:
            running = self.state["status"] == "running" and self._thread is not None
        if not running:
            return {**self.snapshot(), "paused": False}
        self._gate.clear()
        self._update(
            status="paused",
            paused_at=datetime.now().isoformat(timespec="seconds"),
        )
        return {**self.snapshot(), "paused": True}

    def wait_paused(self, timeout: float | None = None) -> bool:
        """Дождаться фактической остановки на паузе (после дописывания батча)."""
        thread = self._thread
        if thread is None:
            return False
        thread.join(timeout=timeout)
        return not thread.is_alive() or self._gate.is_set()

    def _progress(self, done: int, total: int, skipped: int, **fields) -> None:
        self._update(done=done, total=total, skipped_files=skipped, **fields)

    def _failures(self, items: list[dict]) -> None:
        self._update(failed_files=items)

    def _run(self) -> None:
        try:
            result = reindex(
                progress=self._progress,
                on_failure=self._failures,
                gate=self._gate,
            )
            self._update(
                status="done",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                collection=result["collection"],
                result=result,
            )
            self._start_verification()
        except Exception as exc:
            self._gate.set()
            self._update(
                status="error",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                error=str(exc),
            )

    @staticmethod
    def _start_verification() -> None:
        try:
            import verify

            verify.job.start()
        except Exception as exc:
            print(f"сверка исправлений не запущена: {exc}")

    def mark_interrupted(self) -> None:
        with self._lock:
            collection = self.state["collection"]
            status = self.state["status"]
        if collection and status in ("running", "paused"):
            index_state.mark_run(collection, "interrupted")


job = ReindexJob()


def reindex(
    progress: Callable[..., None] | None = None,
    on_failure: Callable[[list[dict]], None] | None = None,
    gate: threading.Event | None = None,
) -> dict:
    """Полная переиндексация без простоя: новая коллекция + переключение алиаса.

    Прогресс сохраняется по ходу работы, поэтому после падения или перезапуска
    индексация продолжается с последнего среза, а не с нуля.
    """
    gate = gate or threading.Event()
    gate.set()

    open_run = index_state.get_open_run()
    orphans = [] if open_run else vs.orphan_collections()
    candidate = open_run["collection"] if open_run else (orphans[-1] if orphans else None)
    if candidate and vs.get_qdrant().collection_exists(candidate):
        collection = candidate
        vs.ensure_collection(collection)
        index_state.start_run(collection)
        resumed = True
        print(f"возобновление индексации: {collection}")
    else:
        collection = f"{config.QDRANT_COLLECTION}_v{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        vs.ensure_collection(collection, recreate=True)
        index_state.start_run(collection)
        resumed = False

    client = vs.get_qdrant()
    already: dict[str, set[int]] = vs.loaded_chunks(collection)
    files = plan_files()
    workers = max(1, config.INGEST_WORKERS)
    counters = {"done": 0, "total": 0, "skipped": 0}
    counter_lock = threading.Lock()
    failed: list[dict] = []

    def bump(field: str, amount: int) -> None:
        with counter_lock:
            counters[field] += amount

    def snapshot_counters() -> None:
        if progress:
            progress(
                counters["done"],
                counters["total"],
                counters["skipped"],
                phase="indexing",
            )

    def index_file(root: str, path: Path, rel: str) -> None:
        source_key = f"{root}|{rel}"
        try:
            stat = path.stat()
            prev = index_state.get_file_progress(collection, source_key)
            text = load_file(path)
            if not text:
                print(f"skip empty: {path}")
                return
            points = chunk_points(rel, split_text(text))

            start_at = 0
            have = already.get(rel, set())
            while start_at < len(points) and start_at in have:
                start_at += 1
            if prev and prev["size"] == stat.st_size and prev["mtime"] == stat.st_mtime:
                start_at = max(start_at, min(prev["done"], len(points)))

            bump("done", start_at)
            bump("total", len(points))
            if start_at >= len(points):
                bump("skipped", 1)
                index_state.save_file_progress(
                    collection,
                    source_key,
                    stat.st_size,
                    stat.st_mtime,
                    len(points),
                    len(points),
                )
                snapshot_counters()
                return

            since_snapshot = 0
            for batch in iter_batches(points[start_at:], config.INGEST_BATCH_SIZE):
                gate.wait()
                upsert_batch(client, collection, batch)
                start_at += len(batch)
                bump("done", len(batch))
                since_snapshot += len(batch)
                if since_snapshot >= config.INGEST_CHECKPOINT_EVERY:
                    index_state.save_file_progress(
                        collection,
                        source_key,
                        stat.st_size,
                        stat.st_mtime,
                        len(points),
                        start_at,
                    )
                    since_snapshot = 0
                snapshot_counters()

            index_state.save_file_progress(
                collection,
                source_key,
                stat.st_size,
                stat.st_mtime,
                len(points),
                start_at,
            )
            print(f"{rel}: {start_at}/{len(points)} chunks")
        except Exception as exc:
            failed.append({"source": rel, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {rel}: {type(exc).__name__}: {exc}")
            if on_failure:
                with counter_lock:
                    on_failure(list(failed))
            snapshot_counters()

    if progress:
        progress(0, 0, 0, resumed=resumed, collection=collection, phase="indexing")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda item: index_file(*item), files))

    done_total = counters["done"]
    total_chunks = counters["total"]
    skipped = counters["skipped"]

    total_known = total_chunks
    if progress:
        progress(done_total, total_chunks, skipped, phase="swapping")

    previous = vs.swap_target(collection)
    index_state.mark_run(collection, "done")
    index_state.forget_run(collection)
    return {
        "collection": collection,
        "points": total_known,
        "previous": previous,
        "skipped_files": skipped,
        "failed_files": failed,
        "resumed": resumed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексация документации в Qdrant")
    parser.add_argument("--recreate", action="store_true", help="пересоздать коллекцию")
    parser.add_argument("--dry-run", action="store_true", help="только разбивка, без эмбеддингов")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="писать в текущую коллекцию без переключения алиаса",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="не продолжать незавершённую индексацию, начать новую",
    )
    args = parser.parse_args()

    if args.dry_run:
        points = collect_chunks()
        print(f"всего чанков: {len(points)}")
        return

    if args.in_place:
        target = vs.target_name()
        vs.ensure_collection(target, recreate=args.recreate)
        upsert_batch(vs.get_qdrant(), target, points := collect_chunks())
        print("готово:", vs.collection_info(target))
        return

    if args.no_resume:
        open_run = index_state.get_open_run()
        if open_run:
            index_state.forget_run(open_run["collection"])
    result = reindex()
    print("готово:", result)


if __name__ == "__main__":
    main()