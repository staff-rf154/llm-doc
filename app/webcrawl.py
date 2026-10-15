"""Обход сайтов и индексация их страниц в общий индекс Qdrant.

Источники задаёт администратор в кабинете («Сайты»): домены или конкретные
адреса. Обход уважает robots.txt, ограничен тем же хостом, техническим лимитом
страниц и паузой между запросами. Текст страниц пишется в активную коллекцию
документации с источником вида `web/<host>/<path>` и URL в payload.
"""

import json
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

import config
import index_state
import settings
import vectorstore as vs

SETTING_KEY = "sites"
SOURCE_PREFIX = "web/"

SKIP_EXT = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".pdf", ".zip", ".rar",
    ".7z", ".mp4", ".mp3", ".avi", ".exe", ".dmg", ".css", ".js", ".woff", ".woff2",
    ".ttf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)

SKIP_MARKERS = (
    "/login", "/logout", "/register", "/reg/", "/cart", "/checkout", "/subscribe",
    "/thank", "/thanks", "/logout", "/search", "/vrednye-sovety",
)

REMOVE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form")


class Crawler:
    def __init__(self, sites: list[dict]) -> None:
        self.sites = sites
        self.client = None
        self.stats: dict = {"pages": 0, "chunks": 0, "errors": [], "skipped": 0, "done": 0}

    def _client(self):
        if self.client is None:
            import httpx

            self.client = httpx.Client(
                timeout=settings.int_value("WEB_CRAWL_TIMEOUT") or 20,
                follow_redirects=True,
                headers={
                    "User-Agent": settings.value(
                        "WEB_CRAWL_USER_AGENT",
                        "Mozilla/5.0 (compatible; DocsBot/1.0)",
                    ),
                    "Accept-Language": "ru-RU,ru;q=0.9",
                },
            )
        return self.client

    @staticmethod
    def _host(url: str) -> str:
        return (urlparse(url).netloc or "").lower()

    def _robots(self, base: str):
        parser = RobotFileParser()
        parser.set_url(urljoin(base, "/robots.txt"))
        try:
            response = self._client().get(parser.url)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser.allow_all = True
        except Exception:
            parser.allow_all = True
        return parser

    @staticmethod
    def _is_html(url: str) -> bool:
        path = urlparse(url).path.lower()
        return not any(path.endswith(ext) for ext in SKIP_EXT)

    def crawl_site(self, start_url: str, max_pages: int, progress) -> list[dict]:
        start_url = urldefrag(start_url)[0]
        host = self._host(start_url)
        robots = self._robots(f"{urlparse(start_url).scheme}://{host}")
        delay = (settings.int_value("WEB_CRAWL_DELAY_MS") or 500) / 1000
        respect = settings.bool_value("WEB_CRAWL_RESPECT_ROBOTS", True)

        queue: deque[str] = deque([start_url])
        seen: set[str] = set()
        documents: list[dict] = []
        failed = 0

        while queue and len(documents) < max_pages:
            url = urldefrag(queue.popleft())[0]
            if url in seen:
                continue
            seen.add(url)
            if self._host(url) != host or not self._is_html(url):
                continue
            path = urlparse(url).path.lower()
            if any(marker in path for marker in SKIP_MARKERS):
                continue
            if respect and not robots.can_fetch("*", url):
                continue

            try:
                response = self._client().get(url)
            except Exception as exc:
                failed += 1
                if len(self.stats["errors"]) < 20:
                    self.stats["errors"].append(f"{url}: {type(exc).__name__}")
                continue

            if response.status_code >= 400:
                failed += 1
                continue
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                continue

            parsed = self._extract(response.text, url, links=queue, host=host)
            documents.append({"url": url, **parsed})
            self.stats["pages"] += 1
            progress(url, self.stats["pages"])
            time.sleep(delay)

        if failed:
            self.stats["skipped"] += failed
        return documents

    @staticmethod
    def _extract(html: str, url: str, links: deque | None = None, host: str = "") -> dict:
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.get_text(strip=True) if soup.title else urlparse(url).path) or url
        for tag in soup(list(REMOVE_TAGS)):
            tag.decompose()
        if links is not None:
            for anchor in soup.find_all("a", href=True):
                target = urldefrag(urljoin(url, anchor["href"]))[0]
                if target.startswith(("http://", "https://")) and Crawler._host(target) == host:
                    links.append(target)

        description = soup.find("meta", attrs={"name": "description"})
        blocks = [title]
        if description and description.get("content"):
            blocks.append(description["content"].strip())
        heading = soup.find(["h1", "h2"])
        if heading:
            blocks.append(heading.get_text(" ", strip=True))
        body = soup.get_text("\n", strip=True)
        lines = [line for line in body.splitlines() if len(line) > 2]
        blocks.append("\n".join(lines))

        text = re.sub(r"\n{3,}", "\n\n", "\n".join(b for b in blocks if b))
        return {"title": title.strip(), "text": text}


def _source_for(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "/").strip("/")
    return f"{SOURCE_PREFIX}{parsed.netloc}/{path}".rstrip("/")


def index_sites(progress=None) -> dict:
    """Обойти все включённые сайты и записать их страницы в активный индекс."""
    import ingest

    client = vs.get_qdrant()
    target = vs.target_name()
    vs.ensure_collection(target)

    crawler = Crawler(sites())
    max_pages = settings.int_value("WEB_CRAWL_MAX_PAGES") or 2000
    documents: list[dict] = []

    for site in crawler.sites:
        if not site.get("enabled", True):
            continue
        start = site.get("url", "").strip()
        if not start:
            continue
        if progress:
            progress(site=start, stage="crawl", pages=crawler.stats["pages"])
        try:
            documents.extend(
                crawler.crawl_site(start, site.get("max_pages") or max_pages, progress)
            )
        except Exception as exc:
            crawler.stats["errors"].append(f"{start}: {type(exc).__name__}: {exc}")

    if not documents:
        return {
            "pages": 0,
            "chunks": 0,
            "errors": crawler.stats["errors"],
            "skipped": crawler.stats["skipped"],
            "collection": target,
        }

    # прежние веб-страницы этого набора заменяем полностью
    delete_web_points(client, target)

    points: list[dict] = []
    total_chunks = 0
    for document in documents:
        source = _source_for(document["url"])
        chunks = ingest.split_text(document["text"])
        total_chunks += len(chunks)
        for index, chunk in enumerate(chunks):
            points.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{index}")),
                    "text": chunk,
                    "source": source,
                    "chunk": index,
                    "url": document["url"],
                    "title": document["title"],
                    "kind": "web",
                }
            )

    upsert_points(client, target, points)
    index_state.mark_run(target, "web-indexed")
    if progress:
        progress(stage="done", pages=len(documents), chunks=total_chunks)
    return {
        "pages": len(documents),
        "chunks": total_chunks,
        "errors": crawler.stats["errors"],
        "skipped": crawler.stats["skipped"],
        "collection": target,
    }


def delete_web_points(client, collection: str) -> int:
    """Удалить ранее проиндексированные страницы сайтов из коллекции."""
    point_ids: list[int] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=2048,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            source = (point.payload or {}).get("source", "")
            if source.startswith(SOURCE_PREFIX):
                point_ids.append(int(point.id))
        if offset is None:
            break
    for start in range(0, len(point_ids), 1024):
        client.delete(
            collection_name=collection,
            points_selector=models_point_ids(point_ids[start : start + 1024]),
        )
    return len(point_ids)


def models_point_ids(ids: list[int]):
    from qdrant_client.http import models

    return models.PointIdsList(points=ids)


def upsert_points(client, collection: str, points: list[dict]) -> None:
    for start in range(0, len(points), 128):
        batch = points[start : start + 128]
        vectors = vs.embed_passages([item["text"] for item in batch])
        client.upsert(
            collection_name=collection,
            points=[
                models_point(
                    id=item["id"],
                    vector=vector,
                    payload={
                        "text": item["text"],
                        "source": item["source"],
                        "chunk": item["chunk"],
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "kind": item.get("kind", "web"),
                    },
                )
                for item, vector in zip(batch, vectors)
            ],
            wait=True,
        )


def models_point(**kwargs):
    from qdrant_client.http import models

    return models.PointStruct(**kwargs)


def sites() -> list[dict]:
    import appdb

    value = appdb.get_setting(SETTING_KEY, "")
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return []


def save_sites(items: list[dict]) -> list[dict]:
    import appdb

    clean = []
    for item in items:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        clean.append(
            {
                "url": url.rstrip("/"),
                "enabled": bool(item.get("enabled", True)),
                "max_pages": int(item.get("max_pages") or 0) or None,
            }
        )
    appdb.set_setting(SETTING_KEY, json.dumps(clean, ensure_ascii=False))
    return clean


class CrawlJob:
    """Фоновый обход сайтов с прогрессом."""

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
            "stage": "idle",
            "site": "",
            "pages": 0,
            "chunks": 0,
            "skipped": 0,
            "errors": [],
            "collection": None,
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
                "stage": "start",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return {**self.snapshot(), "started": True}

    def _run(self) -> None:
        def progress(*args, **fields):
            if args and "pages" not in fields:
                url, pages = args[0], args[1]
                fields.update(pages=pages, site=url)
            self._update(**fields)

        try:
            result = index_sites(progress=progress)
            self._update(
                status="done",
                stage="done",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                **result,
            )
        except Exception as exc:
            self._update(
                status="error",
                stage="error",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                error=f"{type(exc).__name__}: {exc}",
            )


job = CrawlJob()