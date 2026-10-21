"""Работа со страницами документов: список файлов, разбор диапазонов, извлечение текста."""

import re
from pathlib import Path

from pypdf import PdfReader

import config
import tenancy

PAGE_SEPARATORS = ("\f",)


def list_documents(tenant: str | None = None) -> list[dict]:
    """Документы, доступные администратору для ссылки на страницу."""
    items: list[dict] = []
    for root in config.SOURCE_ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in (".pdf", ".txt", ".md", ".html", ".htm"):
                continue
            try:
                rel = str(path.relative_to(root_path))
            except ValueError:
                rel = path.name
            if tenant is not None and not tenancy.in_tenant(rel, tenant):
                continue
            items.append(
                {
                    "root": root,
                    "name": rel,
                    "path": str(path),
                    "kind": "pdf" if suffix == ".pdf" else "text",
                }
            )
    return items


def find_document(name: str, tenant: str | None = None) -> dict | None:
    target = name.strip().lower()
    for item in list_documents(tenant):
        if item["name"].lower() == target or Path(item["name"]).name.lower() == target:
            return item
    return None


def parse_pages(value: str) -> list[tuple[int, int]]:
    """«7, 12-15» → [(7, 7), (12, 15)]. Страницы нумеруются с 1."""
    pages: list[tuple[int, int]] = []
    for chunk in re.split(r"[,;]", value or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.match(r"^(\d+)\s*[-–—]\s*(\d+)$", chunk)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start > end:
                start, end = end, start
            pages.append((start, end))
            continue
        match = re.match(r"^(\d+)$", chunk)
        if match:
            number = int(match.group(1))
            pages.append((number, number))
    return pages


def format_pages(pages: list[tuple[int, int]]) -> str:
    return ", ".join(str(start) if start == end else f"{start}-{end}" for start, end in pages)


def build_reference(doc_name: str, doc_pages: str) -> str:
    doc_name = doc_name.strip()
    doc_pages = doc_pages.strip()
    if not doc_name:
        return ""
    return f"{doc_name}, с. {doc_pages}" if doc_pages else doc_name


ACRONYMS = {
    "api", "csv", "dll", "dns", "faq", "gui", "http", "https", "icap", "id",
    "imap", "ip", "kib", "ldap", "os", "pdf", "pop3", "sdk", "siem", "smtp",
    "sql", "ssl", "tls", "ui", "url", "utf", "vpn", "xml",
}


def human_title(source: str, title: str = "", kind: str = "file") -> str:
    """Полное название документа по имени файла: siem_help.pdf → SIEM Help."""
    name = (source or "").strip()
    if not name:
        return title.strip()
    if name.startswith("web/"):
        host = name.split("/", 2)[1] if name.count("/") >= 2 else name[4:]
        clean = title.strip()
        if clean and len(clean) > 2 and clean.lower() not in host.lower():
            return clean
        return host or name
    stem = Path(name).stem or name
    words = [w for w in re.split(r"[\s_\-]+", stem) if w]
    parts = []
    for word in words:
        lower = word.lower()
        parts.append(lower.upper() if lower in ACRONYMS else word[:1].upper() + word[1:])
    return " ".join(parts) if parts else name


def _pdf_page_text(path: str, start: int, end: int) -> str:
    reader = PdfReader(path)
    total = len(reader.pages)
    blocks = []
    for number in range(max(1, start), min(total, end) + 1):
        try:
            text = reader.pages[number - 1].extract_text() or ""
        except Exception as exc:
            text = f"[страница {number} не читается: {type(exc).__name__}]"
        blocks.append(f"--- {doc_short(path)}, с. {number} ---\n{text.strip()}")
    header = f"Документ: {doc_short(path)} (страниц в файле: {total})"
    return header + "\n\n" + "\n\n".join(blocks)


def doc_short(path: str) -> str:
    return Path(path).name


def _text_page_text(path: str, start: int, end: int) -> str:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    chunks: list[str] = []
    for separator in PAGE_SEPARATORS:
        if separator in raw:
            chunks = raw.split(separator)
            break
    if not chunks:
        chunks = [raw]
    blocks = []
    for number in range(max(1, start), min(len(chunks), end) + 1):
        blocks.append(f"--- {doc_short(path)}, с. {number} ---\n{chunks[number - 1].strip()}")
    header = f"Документ: {doc_short(path)} (условных страниц: {len(chunks)})"
    return header + "\n\n" + "\n\n".join(blocks)


def extract_pages(doc_name: str, doc_pages: str, tenant: str | None = None) -> str:
    """Текст указанных страниц документа; пустая строка, если документ не найден."""
    document = find_document(doc_name, tenant)
    if not document:
        return ""
    pages = parse_pages(doc_pages)
    start = pages[0][0] if pages else 1
    end = pages[-1][1] if pages else 1
    if document["kind"] == "pdf":
        return _pdf_page_text(document["path"], start, end)
    return _text_page_text(document["path"], start, end)
