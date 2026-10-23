# План переноса БД на PostgreSQL

Статус: согласован, выполняется поэтапно.
Объём: все 5 SQLite-баз, с переносом существующих данных.

## 1. Архитектура

- Новый сервис `postgres` в `docker-compose.yml` (`postgres:16`, том
  `./data/postgres`, healthcheck `pg_isready`, порт наружу не публикуется).
- Одна БД `llm` и 5 схем:
  - `app` — users, sessions, requests, ratings, corrections, lessons,
    domains, tenants, ...
  - `settings` — settings, meta
  - `prompts` — prompts
  - `synonyms` — synonyms
  - `index_state` — runs, file_progress
- Подключение через `psycopg 3` + `psycopg_pool`, `search_path` выставляется
  по модулю, пул общий.
- Отдельные БД не используем: ломается текущая миграция
  `settingsdb._read_legacy` (читает `settings` из `app.db`), нужны 5 пулов и
  сложнее бэкап.

## 2. Код

Затрагиваются: `app/appdb.py`, `app/settingsdb.py`, `app/promptdb.py`,
`app/synonymdb.py`, `app/index_state.py`, `app/config.py`, `app/settings.py`,
`app/main.py`, новый `app/db.py`.

Ключевые замены:

- `?` → `%s`, `row_factory` → `dict_row`.
- `INTEGER PRIMARY KEY AUTOINCREMENT` → `BIGSERIAL`, `BLOB` → `BYTEA`
  (`appdb.py:169`).
- `INSERT OR IGNORE` → `ON CONFLICT DO NOTHING`, `INSERT OR REPLACE` →
  `ON CONFLICT ... DO UPDATE`.
- `lastrowid` → `RETURNING id` (`appdb.py:418,636,715,800,1157,1623,1779,1864`).
- `datetime('now', ?)` → `now() - make_interval(days => ...)`
  (`appdb.py:1328,1447`).
- `PRAGMA table_info` / `sqlite_master` в `_migrate` (`appdb.py:213`) →
  идемпотентные `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` /
  `information_schema`.
- `executescript` → список DDL-выражений.
- `BEGIN IMMEDIATE` в health (`main.py:220`) → `SELECT 1`, ключ `sqlite` →
  `postgres`.
- Проверить `cur.rowcount` при UPDATE без изменений (`set_active`,
  `delete_*`, `upsert_*`).
- `np.frombuffer` в `autolearn.py:256` с `BYTEA` работает без изменений.

## 3. Миграция данных

Новый скрипт `scripts/migrate_sqlite_to_pg.py` (запускается в контейнере
`app`, где есть sqlite3 и psycopg):

- читает таблицы из 5 файлов и вставляет в соответствующие схемы с
  сохранением `id`;
- порядок с учётом FK: tenants → users → sessions/requests →
  ratings/request_evals → corrections/lessons/... → domains;
- после загрузки `setval` для всех последовательностей;
- отдельный шаг проверки FK и orphan-строк;
- сверка `COUNT(*)` по каждой таблице SQLite vs PG.

Исходные `.db` не удаляем, оставляем как fallback.

## 4. Compose, env, зависимости

- `docker-compose.yml`: сервис `postgres`, `app.depends_on:
  postgres: service_healthy`, `DATABASE_URL` в окружении, bind-mount
  `./data/postgres` (владелец — UID postgres).
- `.env` / `.env.example`: `POSTGRES_USER`, сгенерированный
  `POSTGRES_PASSWORD`, `POSTGRES_DB=llm`, `DATABASE_URL`.
- `app/requirements.txt`: `psycopg[binary]` + `psycopg-pool`, образ
  пересобирается, нужна сеть.
- `app/config.py`: `DATABASE_URL` вместо пяти путей; пути оставить для
  скрипта миграции.
- `app/settings.py:208-219`: параметры-пути заменить на `DATABASE_URL`.

## 5. Бэкапы

- `scripts/backup.sh`: вместо `sqlite3.backup` — `pg_dump -Fc` + manifest;
  на переходный период дополнительно копировать SQLite-файлы.
- `scripts/restore.sh`: `pg_restore --clean --if-exists`.
- `Makefile`: цель `migrate-db`.
- README: разделы «Хранение», «Переменные», «Резервное копирование».

## 6. Порядок выполнения и проверка

1. Бэкап рабочего дерева (`git diff` + tar, т.к. есть незакоммиченные
   правки) и бэкап БД в `data/backups/`.
2. Поднять PG, применить схему.
3. Остановить `app`, выполнить миграцию данных.
4. Сверить количество строк по всем таблицам, проверить FK.
5. Пересобрать и запустить `app`.
6. Проверить `/health?deep=1`, вход, настройки, пользователей/домены,
   журнал запросов, исправления, промты, синонимы, состояние индексации,
   логи, перезапуск с сохранением данных.
7. Оставить SQLite-файлы и старые бэкапы на период наблюдения.

Ориентировочный downtime: 5–15 минут.

## 7. Риски и откат

- Нарушение FK в старых данных → проверка и загрузка в правильном порядке.
- Ошибка в диалекте SQL → сверка строк и смоук-тесты.
- Откат: остановить `app`, вернуть прежние `docker-compose.yml` / `.env`,
  поднять SQLite-версию; данные SQLite не тронуты.

## 8. Результат выполнения (2026-10-22)

- Бэкап до изменений: `/root/llm-backups/pre-pg-20261022-094435`
  (5 SQLite-БД, исходники, `git diff`, манифест).
- Добавлен сервис `postgres` (`postgres:16`, `llm-postgres`, том
  `data/postgres`), приложение переведено на `DATABASE_URL`.
- Код переведён на `psycopg 3` + `psycopg_pool`: добавлен `app/db.py`,
  переписаны `appdb.py`, `settingsdb.py`, `promptdb.py`, `synonymdb.py`,
  `index_state.py`, `metrics.py`, health-проверка.
- Данные перенесены скриптом `app/migrate_sqlite_to_pg.py`. Сверка:
  17 таблиц, расхождений нет (app.requests 519, users 5, domains 20,
  prompts 41, synonyms 35 и т.д.).
- `pg_dump`/`pg_restore` проверены на временной базе, восстановление
  вернуло 519 запросов.
- `make backup` переведён на `pg_dump`, добавлен `make migrate-db`.
- Смоук-тесты: `/health?deep=1` — `postgres.ok=true`; чтение и запись
  (настройки, урок) работают; в логах ошибок нет.
- SQLite-файлы не удалены и остаются как резерв для отката.
