import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def _normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if line.strip() == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    soup = BeautifulSoup(xml, "xml")
    return "\n".join(t.get_text() for t in soup.find_all("w:t"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_html(path: Path) -> str:
    return BeautifulSoup(
        path.read_text(encoding="utf-8", errors="ignore"), "html.parser"
    ).get_text("\n")


READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_text,
    ".md": _read_text,
    ".html": _read_html,
    ".htm": _read_html,
}


def load_file(path: Path) -> str:
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        return ""
    return _normalize(reader(path))


def iter_documents(root: str):
    root_path = Path(root)
    if not root_path.exists():
        return
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED:
            yield path