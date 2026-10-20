"""Состояние резервных копий: последняя копия, размер, список."""

import json
from datetime import datetime
from pathlib import Path

import config


def _dirs() -> list[Path]:
    root = Path(config.BACKUP_DIR)
    if not root.exists():
        return []
    return sorted(
        (item for item in root.iterdir() if item.is_dir() and (item / "manifest.json").exists()),
        key=lambda item: item.name,
        reverse=True,
    )


def status(limit: int = 7) -> dict:
    root = Path(config.BACKUP_DIR)
    items = []
    for path in _dirs()[:limit]:
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        items.append(
            {
                "name": path.name,
                "created_at": manifest.get("created_at", ""),
                "bytes": int(manifest.get("bytes", 0)),
                "files": len(manifest.get("files", [])),
            }
        )
    latest = items[0] if items else None
    age_hours = None
    if latest:
        try:
            manifest_path = root / latest["name"] / "manifest.json"
            age_hours = round(
                (datetime.now().timestamp() - manifest_path.stat().st_mtime) / 3600, 1
            )
        except OSError:
            pass
    return {
        "dir": str(root),
        "exists": root.exists(),
        "count": len(_dirs()),
        "latest": latest,
        "age_hours": age_hours,
        "items": items,
    }


if __name__ == "__main__":
    print(json.dumps(status(), ensure_ascii=False, indent=2))
