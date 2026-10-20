#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
  echo "Использование: RESTORE=yes scripts/restore.sh data/backups/<копия> [--docs]" >&2
  exit 2
fi

DIR=${1%/}
RESTORE_DOCS=0
if [ "${2:-}" = "--docs" ]; then
  RESTORE_DOCS=1
fi

if [ "${RESTORE:-}" != "yes" ]; then
  echo "Восстановление необратимо. Запустите с RESTORE=yes." >&2
  exit 3
fi
if [ ! -d "$DIR" ] || [ ! -f "$DIR/manifest.json" ]; then
  echo "Копия не найдена или повреждена: $DIR" >&2
  exit 4
fi

echo "Проверка копии"
python3 - "$DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for item in manifest.get("files", []):
    path = root / item["name"]
    if not path.exists():
        raise SystemExit(f"нет файла: {path}")
    if path.stat().st_size != item["size"]:
        raise SystemExit(f"размер не совпадает: {path}")
print(f"копия от {manifest.get('created_at')}, файлов: {len(manifest.get('files', []))}")
PY

echo "Остановка приложения"
docker compose stop app

echo "Восстановление SQLite"
python3 - "$DIR" <<'PY'
import os
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
mapping = {
    "data_app_app.db": "data/app/app.db",
    "data_settings_settings.db": "data/settings/settings.db",
    "data_synonyms_synonyms.db": "data/synonyms/synonyms.db",
    "data_prompts_prompts.db": "data/prompts/prompts.db",
    "data_index_state_index.db": "data/index_state/index.db",
}
if (root / "system_password.txt").exists():
    Path("data/app").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "system_password.txt", "data/app/system_password.txt")

for name, target in mapping.items():
    source = root / name
    if not source.exists():
        print(f"пропуск: {name}")
        continue
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("-wal", "-shm"):
        extra = Path(str(target_path) + suffix)
        if extra.exists():
            extra.unlink()
    shutil.copy2(source, target_path)
    print(target)
PY

if [ "$RESTORE_DOCS" = "1" ] && [ -f "$DIR/docs.tar.gz" ]; then
  echo "Восстановление документации"
  tar -xzf "$DIR/docs.tar.gz" -C .
fi

SNAP="$DIR/qdrant.snapshot"
COLL=$(cat "$DIR/qdrant_collection.txt" 2>/dev/null || echo docs)
if [ -f "$SNAP" ]; then
  echo "Восстановление Qdrant: $COLL"
  docker compose cp "$SNAP" qdrant:/qdrant/snapshots/restore.snapshot
  QPORT=${QDRANT_PORT:-6333}
  QKEY=$(grep -E '^QDRANT_API_KEY=' .env 2>/dev/null | cut -d= -f2- || true)
  AUTH=()
  if [ -n "${QKEY:-}" ]; then
    AUTH=(-H "api-key: $QKEY")
  fi
  curl -fsS "${AUTH[@]}" -X PUT "http://localhost:$QPORT/collections/$COLL/snapshots/recover" \
    -H "Content-Type: application/json" \
    -d '{"location":"file:///qdrant/snapshots/restore.snapshot","priority":"snapshot"}' >/dev/null
  curl -fsS "${AUTH[@]}" -X POST "http://localhost:$QPORT/collections/aliases" \
    -H "Content-Type: application/json" \
    -d "{\"actions\":[{\"delete_alias\":{\"alias_name\":\"docs\"}}]}" >/dev/null || true
  curl -fsS "${AUTH[@]}" -X POST "http://localhost:$QPORT/collections/aliases" \
    -H "Content-Type: application/json" \
    -d "{\"actions\":[{\"create_alias\":{\"collection_name\":\"$COLL\",\"alias_name\":\"docs\"}}]}" >/dev/null
  docker compose exec -T qdrant rm -f /qdrant/snapshots/restore.snapshot
fi

echo "Запуск приложения"
docker compose start app
PORT=${APP_PORT:-8091}
ok=0
for _ in $(seq 1 30); do
  if curl -fsS --max-time 10 "http://localhost:$PORT/health?deep=1" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 2
done
if [ "$ok" != "1" ]; then
  echo "Приложение не прошло deep health за 60 с; проверьте docker compose logs app" >&2
  exit 5
fi
curl -fsS "http://localhost:$PORT/health?deep=1"
echo
echo "Восстановление завершено"
