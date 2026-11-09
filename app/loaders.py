import csv
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json",
    ".xml", ".yaml", ".yml", ".ini", ".conf", ".cfg", ".properties",
    ".sql", ".sh", ".bat", ".ps1", ".py", ".js", ".ts", ".java", ".go",
    ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".pl",
    ".css", ".svg", ".tex", ".srt", ".vtt",
}
HTML_SUFFIXES = {".html", ".htm", ".xhtml", ".epub"}
OFFICE_SUFFIXES = {".docx", ".pptx", ".odt", ".ods", ".odp"}
LEGACY_OFFICE_SUFFIXES = {".doc", ".xls", ".ppt"}
TABLE_SUFFIXES = {".xlsx"}
SUPPORTED = (
    {".pdf"}
    | TEXT_SUFFIXES
    | HTML_SUFFIXES
    | OFFICE_SUFFIXES
    | LEGACY_OFFICE_SUFFIXES
    | TABLE_SUFFIXES
)

FORMAT_LABELS = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".doc": "DOC",
    ".pptx": "PPTX",
    ".ppt": "PPT",
    ".odt": "ODT",
    ".ods": "ODS",
    ".odp": "ODP",
    ".xlsx": "XLSX",
    ".xls": "XLS",
    ".txt": "TXT",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".rst": "reStructuredText",
    ".log": "LOG",
    ".csv": "CSV",
    ".tsv": "TSV",
    ".json": "JSON",
    ".xml": "XML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".html": "HTML",
    ".htm": "HTML",
    ".xhtml": "HTML",
    ".epub": "EPUB",
    ".sql": "SQL",
    ".sh": "Shell",
    ".bat": "BAT",
    ".ps1": "PowerShell",
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".ini": "INI",
    ".conf": "CONF",
    ".cfg": "CFG",
    ".properties": "PROPERTIES",
    ".svg": "SVG",
    ".css": "CSS",
}


CONTENT_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "application/epub+zip": ".epub",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "text/markdown": ".md",
    "application/json": ".json",
    "application/xml": ".xml",
    "text/xml": ".xml",
    "text/html": ".html",
    "application/xhtml+xml": ".xhtml",
}


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
    paragraphs = []
    for paragraph in soup.find_all("w:p"):
        text = "".join(node.get_text() for node in paragraph.find_all("w:t"))
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _read_pptx(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        slides = sorted(
            (name for name in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
            key=lambda name: int(re.search(r"(\d+)", Path(name).stem).group(1)),
        )
        blocks = []
        for slide in slides:
            xml = zf.read(slide).decode("utf-8", errors="ignore")
            soup = BeautifulSoup(xml, "xml")
            texts = [node.get_text() for node in soup.find_all("a:t")]
            text = " ".join(item.strip() for item in texts if item.strip())
            if text:
                blocks.append(f"[{Path(slide).stem}] {text}")
    return "\n".join(blocks)


def _read_odf(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("content.xml").decode("utf-8", errors="ignore")
    soup = BeautifulSoup(xml, "xml")
    blocks = []
    for paragraph in soup.find_all(re.compile(r"^text:p$|^text:h$")):
        text = paragraph.get_text(" ", strip=True)
        if text:
            blocks.append(text)
    return "\n".join(blocks)


def _read_epub(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = sorted(name for name in zf.namelist() if name.lower().endswith((".html", ".xhtml", ".htm")))
        blocks = []
        for name in names:
            raw = zf.read(name).decode("utf-8", errors="ignore")
            text = BeautifulSoup(raw, "html.parser").get_text("\n")
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if text:
                blocks.append(text)
    return "\n".join(blocks)


def _read_xlsx(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError:
        return ""
    book = load_workbook(str(path), read_only=True, data_only=True)
    blocks = []
    try:
        for sheet in book.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if cell is None else str(cell).strip() for cell in row]
                if any(cells):
                    rows.append("\t".join(cells))
            if rows:
                blocks.append(f"[{sheet.title}]\n" + "\n".join(rows))
    finally:
        book.close()
    return "\n\n".join(blocks)


def _read_legacy(path: Path, tool: str) -> str:
    try:
        result = subprocess.run(
            [tool, "-d", "utf-8", str(path)],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", errors="ignore")


def _read_doc(path: Path) -> str:
    return _read_legacy(path, "catdoc")


def _read_xls(path: Path) -> str:
    return _read_legacy(path, "xls2csv")


def _read_ppt(path: Path) -> str:
    return _read_legacy(path, "catppt")


def _read_csv(path: Path, delimiter: str | None = None) -> str:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else None
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(raw[:8192], delimiters=",;\t|").delimiter
        except Exception:
            delimiter = ","
    reader = csv.reader(io.StringIO(raw), delimiter=delimiter or ",")
    lines = []
    for row in reader:
        cells = [cell.strip() for cell in row]
        if any(cells):
            lines.append("\t".join(cells))
    return "\n".join(lines)


def _read_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_html(path: Path) -> str:
    return BeautifulSoup(
        path.read_text(encoding="utf-8", errors="ignore"), "html.parser"
    ).get_text("\n")


READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".pptx": _read_pptx,
    ".odt": _read_odf,
    ".ods": _read_odf,
    ".odp": _read_odf,
    ".xlsx": _read_xlsx,
    ".doc": _read_doc,
    ".xls": _read_xls,
    ".ppt": _read_ppt,
    ".epub": _read_epub,
    ".csv": _read_csv,
    ".tsv": _read_csv,
    ".json": _read_json,
    ".html": _read_html,
    ".htm": _read_html,
    ".xhtml": _read_html,
    ".svg": _read_text,
}


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    reader = READERS.get(suffix)
    if reader is not None:
        return _normalize(reader(path))
    if suffix in TEXT_SUFFIXES:
        return _normalize(_read_text(path))
    return ""


def load_document(path: Path) -> dict:
    """Документ с разбивкой по страницам и заголовком из метаданных.

    Возвращает title, pages (текст по страницам, для не-PDF — одна условная
    страница) и text (весь текст сразу).
    """
    if path.suffix.lower() == ".pdf":
        return _read_pdf_document(path)
    text = load_file(path)
    return {
        "title": "",
        "pages": [text] if text else [],
        "page_labels": ["1"] if text else [],
        "text": text,
    }


ROMAN_PAGES = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19,
    "XX": 20,
}


def _printed_page_label(text: str, number: int, total: int) -> str:
    """Номер страницы, напечатанный в самом PDF, а не физический индекс."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return str(number)
    head = lines[:3]
    cleaned = []
    for line in head:
        line = re.sub(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", " ", line)
        line = re.sub(r"\b(?:19|20)\d{2}\b", " ", line)
        cleaned.append(line)
    candidates: list[tuple[int, int, int, str]] = []

    def add(value: int, label: str, priority: int) -> None:
        if 1 <= value <= total:
            candidates.append((abs(value - number), priority, value, label))

    first = cleaned[0]
    match = re.search(r"(\d{1,4})\s*$", first)
    if match:
        add(int(match.group(1)), match.group(1), 0)
    match = re.search(r"([IVXLCDM]{1,7})\s*$", first.upper())
    if match and match.group(1) in ROMAN_PAGES:
        add(ROMAN_PAGES[match.group(1)], match.group(1), 0)
    match = re.match(r"^(\d{1,4})\b", first)
    if match:
        add(int(match.group(1)), match.group(1), 1)
    for line in head:
        if re.fullmatch(r"\d{1,4}", line):
            add(int(line), line, 1)
        for item in re.finditer(r"(?<![\d.])(\d{1,4})(?![\d.])", line):
            add(int(item.group(1)), item.group(1), 2)
        upper = line.upper()
        if re.fullmatch(r"[IVXLCDM]{1,7}", upper) and upper in ROMAN_PAGES:
            add(ROMAN_PAGES[upper], upper, 2)
    candidates = [item for item in candidates if item[0] <= 30]
    if not candidates:
        return str(number)
    candidates.sort()
    return candidates[0][3]


def _read_pdf_document(path: Path) -> dict:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(_normalize(page.extract_text() or ""))
        except Exception:
            pages.append("")
    labels = [
        _printed_page_label(text, index, len(pages))
        for index, text in enumerate(pages, start=1)
    ]
    title = ""
    try:
        meta_title = (reader.metadata.title or "").strip()
        title = meta_title if meta_title.lower() not in (path.stem.lower(), "") else ""
    except Exception:
        title = ""
    if not title:
        for candidate in pages[:2]:
            for line in candidate.splitlines():
                clean = line.strip()
                if 4 <= len(clean) <= 120:
                    title = clean
                    break
            if title:
                break
    return {
        "title": title,
        "pages": pages,
        "page_labels": labels,
        "text": "\n".join(pages),
    }


def iter_documents(root: str):
    root_path = Path(root)
    if not root_path.exists():
        return
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED:
            yield path


def suffix_for_content_type(content_type: str) -> str:
    return CONTENT_TYPE_SUFFIXES.get((content_type or "").split(";")[0].strip().lower(), "")
