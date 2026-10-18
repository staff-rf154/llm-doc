"""Разбор логов по пути из вопроса: распаковка архивов и выборка важных строк."""

import bz2
import gzip
import lzma
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import config
import settings

PATH_RE = re.compile(r"(/[^\s\"'`<>|*?]+)")
WORD_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]{4,}")
SEVERITY_RE = re.compile(
    r"\b(error|err|fail|failed|failure|fatal|exception|traceback|timeout|timed ?out|"
    r"critical|panic|denied|refused|unable|crash|segfault|oom|killed|warning|warn)\b",
    re.IGNORECASE,
)
SEVERITY_RU_RE = re.compile(
    r"(ошибк|сбой|отказ|исключени|таймаут|не удалось|паник|не найден|недоступ)",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {
    ".log", ".txt", ".out", ".err", ".trace", ".dump",
    ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
    ".ini", ".conf", ".cfg", ".xml", ".md", ".rst",
}
ARCHIVE_SUFFIXES = {
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".tbz2", ".tbz", ".xz", ".txz",
}
STOPWORDS = {
    "логи", "лог", "файл", "файле", "файлы", "папке", "папка", "каталог",
    "путь", "почему", "работает", "работать", "изучи", "возьми", "дай",
    "ответ", "проанализируй", "разбери", "что", "какой", "какие", "зачем",
    "please", "logs", "log", "file", "folder", "path", "why", "error",
    "analyze", "look", "read",
}


class LogError(Exception):
    """Ошибка разбора логов, текст показывается пользователю."""


def _root() -> Path:
    return Path(settings.value("LOGS_ROOT", config.LOGS_ROOT)).resolve()


def _max_chars() -> int:
    return settings.int_value("LOGS_MAX_CHARS")


def _max_context() -> int:
    return settings.int_value("LOGS_CONTEXT_LINES")


def _max_files() -> int:
    return settings.int_value("LOGS_MAX_FILES")


def _max_file_bytes() -> int:
    return settings.int_value("LOGS_MAX_FILE_BYTES")


def _max_unpacked() -> int:
    return settings.int_value("LOGS_MAX_UNPACKED_BYTES")


def find_path(question: str) -> str | None:
    """Найти в вопросе абсолютный путь."""
    for raw in PATH_RE.findall(question or ""):
        candidate = raw.rstrip(".,;:!?»)")
        if candidate.startswith("//"):
            continue
        return candidate
    return None


def _within_root(path: Path) -> bool:
    root = _root()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve(candidate: str) -> Path:
    """Проверить, что путь существует и лежит внутри корня логов."""
    root = _root()
    if not root.exists():
        raise LogError(f"Каталог логов ещё не создан: {root}")
    target = Path(candidate)
    if not target.is_absolute():
        raise LogError("Нужен абсолютный путь к логам.")
    try:
        resolved = target.resolve()
    except OSError as exc:
        raise LogError(f"Не удалось разобрать путь: {exc}") from exc
    if not _within_root(resolved):
        raise LogError(f"Путь вне разрешённого каталога логов ({root}).")
    if not resolved.exists():
        raise LogError(f"Путь не найден: {resolved}")
    return resolved


def detect_kind(path: Path) -> str:
    """Определить тип: dir, zip, tar, gzip, bzip2, xz, plain."""
    if path.is_dir():
        return "dir"
    try:
        head = path.open("rb").read(6)
    except OSError:
        return "plain"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gzip" if not _is_tar(path) else "tar"
    if head.startswith(b"BZh"):
        return "bzip2" if not _is_tar(path) else "tar"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz" if not _is_tar(path) else "tar"
    if path.suffix.lower() == ".zip":
        return "zip"
    if _is_tar(path):
        return "tar"
    return "plain"


def _is_tar(path: Path) -> bool:
    try:
        return tarfile.is_tarfile(path)
    except OSError:
        return False


def _safe_name(dest: Path, name: str) -> Path | None:
    clean = name.replace("\\", "/").lstrip("/")
    if not clean or clean.startswith("../") or "/../" in clean:
        return None
    target = (dest / clean).resolve()
    try:
        target.relative_to(dest.resolve())
    except ValueError:
        return None
    return target


def _extract_zip(path: Path, dest: Path) -> list[Path]:
    written: list[Path] = []
    budget = _max_unpacked()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = _safe_name(dest, info.filename)
            if target is None or info.file_size > _max_file_bytes():
                continue
            budget -= info.file_size
            if budget < 0:
                break
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            written.append(target)
    return written


def _extract_tar(path: Path, dest: Path) -> list[Path]:
    written: list[Path] = []
    budget = _max_unpacked()
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            target = _safe_name(dest, member.name)
            if target is None or member.size > _max_file_bytes():
                continue
            budget -= member.size
            if budget < 0:
                break
            target.parent.mkdir(parents=True, exist_ok=True)
            src = archive.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            written.append(target)
    return written


def _decompress_stream(opener, path: Path, dest: Path) -> list[Path]:
    stem = path.name
    for suffix in (".gz", ".bz2", ".xz"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    target = _safe_name(dest, stem) or (dest / "decompressed")
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with opener(path) as src, target.open("wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > _max_file_bytes():
                break
            out.write(chunk)
    return [target]


def extract(path: Path, dest: Path) -> list[Path]:
    kind = detect_kind(path)
    if kind == "dir":
        return [item for item in sorted(path.rglob("*")) if item.is_file()]
    if kind == "zip":
        return _extract_zip(path, dest)
    if kind == "tar":
        return _extract_tar(path, dest)
    if kind == "gzip":
        return _decompress_stream(gzip.open, path, dest)
    if kind == "bzip2":
        return _decompress_stream(bz2.open, path, dest)
    if kind == "xz":
        return _decompress_stream(lzma.open, path, dest)
    return [path]


def _looks_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    try:
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    return True


def _keywords(question: str) -> set[str]:
    words = {word.lower() for word in WORD_RE.findall(question or "")}
    return {word for word in words if word not in STOPWORDS}


def _score(line: str, keywords: set[str]) -> int:
    total = 0
    if SEVERITY_RE.search(line) or SEVERITY_RU_RE.search(line):
        total += 2
    lower = line.lower()
    for word in keywords:
        if word in lower:
            total += 1
    return total


def _select_lines(lines: list[str], keywords: set[str], budget: int) -> tuple[list[str], bool]:
    scored = [(index, _score(line, keywords)) for index, line in enumerate(lines)]
    hits = [index for index, points in scored if points > 0]
    if not hits:
        tail = lines[-40:] if len(lines) > 40 else lines
        text = "\n".join(tail)
        if len(text) > budget:
            text = text[-budget:]
        return [text], len(text) >= budget
    points = dict(scored)
    context = _max_context()
    keep: set[int] = set()
    for index in hits:
        for offset in range(-context, context + 1):
            neighbour = index + offset
            if 0 <= neighbour < len(lines):
                keep.add(neighbour)
    ordered = sorted(keep, key=lambda i: (-points.get(i, 0), i))
    selected: list[str] = []
    used = 0
    truncated = False
    for index in ordered:
        if index not in keep:
            continue
        line = lines[index]
        cost = len(line) + 1
        if used + cost > budget:
            truncated = True
            break
        selected.append((index, line))
        used += cost
        keep.discard(index)
    selected.sort(key=lambda pair: pair[0])
    return [line for _, line in selected], truncated


def _read_lines(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(_max_file_bytes()).splitlines()
    except OSError:
        return []


def analyze(question: str) -> dict:
    """Разобрать указанный в вопросе путь и вернуть выдержки из логов."""
    candidate = find_path(question)
    if not candidate:
        raise LogError("В вопросе не указан путь к логам.")
    target = resolve(candidate)
    root = _root()
    keywords = _keywords(question)
    budget = _max_chars()

    workdir = tempfile.mkdtemp(prefix="logs-")
    try:
        extracted = extract(target, Path(workdir))
        text_files = [item for item in extracted if item.is_file() and _looks_text(item)]
        text_files = text_files[:_max_files()]
        if not text_files:
            raise LogError("В указанном пути не найдено текстовых логов для разбора.")

        reports: list[dict] = []
        best: list[tuple[int, str]] = []
        for path_file in text_files:
            lines = _read_lines(path_file)
            if not lines:
                continue
            top = max((_score(line, keywords) for line in lines), default=0)
            best.append((top, str(path_file)))
        order = {path_file: rank for rank, (_, path_file) in enumerate(sorted(best, reverse=True))}
        text_files.sort(key=lambda item: order.get(str(item), len(order)))

        remaining = budget
        total = 0
        for path_file in text_files:
            if remaining <= 0:
                break
            lines = _read_lines(path_file)
            if not lines:
                continue
            share = max(2000, remaining // 2)
            excerpt, truncated = _select_lines(lines, keywords, min(share, remaining))
            name = _display_name(path_file, root)
            text = "\n".join(excerpt) if excerpt else "(нет значимых строк)"
            reports.append({
                "name": name,
                "lines": len(lines),
                "text": text,
                "truncated": truncated,
            })
            remaining -= len(text)
            total += len(text)

        return {
            "root": str(root),
            "path": str(target),
            "kind": detect_kind(target),
            "files": reports,
            "truncated": remaining <= 0,
            "chars": total,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _display_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def build_prompt(question: str, report: dict) -> str:
    blocks = []
    for item in report["files"]:
        blocks.append(
            f"<log file=\"{item['name']}\" lines=\"{item['lines']}\">\n"
            f"{item['text']}\n</log>"
        )
    head = (
        "Ниже выдержки из логов, отобранные по проблеме. "
        "Путь: " + report["path"] + ". "
        "Разбери их: определи причину сбоя, приведи вероятные гипотезы по убыванию "
        "вероятности, укажи конкретные строки-подтверждения и шаги проверки.\n\n"
    )
    return head + "\n\n".join(blocks) + f"\n\nВопрос: {question}"


def sources_block(report: dict) -> str:
    if not report.get("files"):
        return ""
    lines = ["", "", "Источники (логи):"]
    for item in report["files"]:
        lines.append(f"— {item['name']} ({item['lines']} строк)")
    return "\n".join(lines)
