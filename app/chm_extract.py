import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup

SKIP_DIRS = {
    "files",
    "custom",
    "sys",
    "it",
    "_default",
    "_themes",
    "language",
    "lang",
}
SKIP_NAMES = {"style.css", "filelist.xml", "context.xml", "index.xml"}

SKIP_TAGS = ["script", "style", "nav"]


def archive_pages(archive: Path) -> list[tuple[str, bytes]]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["7z", "x", "-y", f"-o{tmp}", str(archive)],
            check=True,
            capture_output=True,
        )
        root = Path(tmp)
        pages: list[tuple[str, bytes]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0].lower() in SKIP_DIRS:
                continue
            if path.name.lower() in SKIP_NAMES or path.suffix.lower() not in {".htm", ".html"}:
                continue
            pages.append((str(relative), path.read_bytes()))
    return pages


def _strip_markup(raw: str) -> str:
    return BeautifulSoup(raw, "html.parser").get_text(" ").strip()


def html_to_text(raw: bytes) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()
    for tag in soup.find_all(["img", "meta", "link", "hr"]):
        tag.decompose()
    title = _strip_markup(soup.title.get_text()) if soup.title else ""
    body = soup.body or soup
    text = body.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    if title and not text.startswith(title):
        text = f"{title}\n{text}"
    return text.strip()


def extract(archive: Path, out_root: Path) -> tuple[int, int]:
    pages = archive_pages(archive)
    target = out_root / archive.stem
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for relative, raw in pages:
        text = html_to_text(raw)
        if len(text) < 40:
            skipped += 1
            continue
        destination = target / Path(relative).with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Распаковка CHM в текст для индексации")
    parser.add_argument("--docs-dir", default="/data/docs")
    parser.add_argument("--out-dir", default="/data/extracted")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    docs_root = Path(args.docs_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    archives = sorted(docs_root.rglob("*.chm"))
    if not archives:
        print("CHM-файлы не найдены")
        return

    for archive in archives:
        target = out_root / archive.stem
        if target.exists() and not args.rebuild:
            print(f"{archive.name}: уже распаковано в {target}")
            continue
        written, skipped = extract(archive, out_root)
        print(f"{archive.name}: {written} страниц, пропущено пустых: {skipped}")


if __name__ == "__main__":
    main()