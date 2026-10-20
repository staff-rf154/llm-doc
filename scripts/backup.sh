#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d-%H%M%S)
DIR="data/backups/$TS"
mkdir -p "$DIR"

echo "[1/5] SQLite"
python3 - "$DIR" <<'PY'
import os
import sqlite3
import sys

dest = sys.argv[1]
sources = [
    "data/app/app.db",
    "data/settings/settings.db",
    "data/synonyms/synonyms.db",
    "data/prompts/prompts.db",
    "data/index_state/index.db",
]
for src in sources:
    if not os.path.exists(src):
        continue
    name = src.replace("/", "_")
    source = sqlite3.connect(src)
    target = sqlite3.connect(os.path.join(dest, name))
    with target:
        source.backup(target)
    target.close()
    source.close()
    print(name)
PY

cp -a data/app/system_password.txt "$DIR/system_password.txt" 2>/dev/null || true

echo "[2/5] Qdrant snapshot"
QPORT=${QDRANT_PORT:-6333}
QKEY=$(grep -E '^QDRANT_API_KEY=' .env 2>/dev/null | cut -d= -f2- || true)
AUTH=()
if [ -n "${QKEY:-}" ]; then
  AUTH=(-H "api-key: $QKEY")
fi
COLL=$(curl -fsS "${AUTH[@]}" "http://localhost:$QPORT/aliases" | python3 -c 'import json,sys; d=json.load(sys.stdin)["result"]["aliases"]; a=[x["collection_name"] for x in d if x["alias_name"]=="docs"]; print(a[0] if a else "docs")')
SNAP=$(curl -fsS "${AUTH[@]}" -X POST "http://localhost:$QPORT/collections/$COLL/snapshots" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])')
docker compose cp "qdrant:/qdrant/snapshots/$COLL/$SNAP" "$DIR/qdrant.snapshot"
curl -fsS "${AUTH[@]}" -X DELETE "http://localhost:$QPORT/collections/$COLL/snapshots/$SNAP" >/dev/null || true
echo "$COLL" > "$DIR/qdrant_collection.txt"

echo "[3/5] Документация"
tar -czf "$DIR/docs.tar.gz" -C data docs

echo "[4/5] Манифест"
python3 - "$DIR" <<'PY'
import json
import os
import sys
from datetime import datetime

dest = sys.argv[1]
files = []
total = 0
for name in sorted(os.listdir(dest)):
    path = os.path.join(dest, name)
    if not os.path.isfile(path):
        continue
    size = os.path.getsize(path)
    total += size
    files.append({"name": name, "size": size})
manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "files": files,
    "bytes": total,
}
with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
print(f"{total} bytes")
PY

echo "[5/5] Ретенция (7 копий)"
ls -1dt data/backups/*/ 2>/dev/null | tail -n +8 | xargs -r rm -rf

echo "Готово: $DIR"
