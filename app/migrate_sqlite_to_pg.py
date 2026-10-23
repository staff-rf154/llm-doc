"""Одноразовый перенос SQLite-баз в PostgreSQL."""

import os
import sqlite3
import sys

import psycopg
from psycopg import sql

import appdb
import config
import index_state
import promptdb
import settingsdb
import synonymdb

SOURCES = (
    (
        "app",
        config.APP_DB,
        (
            "tenants",
            "users",
            "sessions",
            "requests",
            "ratings",
            "corrections",
            "lessons",
            "training_log",
            "settings",
            "request_evals",
            "answer_issues",
            "question_topics",
            "topic_requests",
            "generated_questions",
            "domains",
            "doc_exclusions",
        ),
    ),
    ("settings", config.SETTINGS_DB, ("settings", "meta")),
    ("prompts", config.PROMPTS_DB, ("prompts",)),
    ("synonyms", config.SYNONYMS_DB, ("synonyms",)),
    ("index_state", config.INDEX_STATE_DB, ("runs", "file_progress")),
)


def ensure_target() -> None:
    appdb.ensure_schema()
    settingsdb.ensure_schema()
    promptdb.ensure_schema()
    synonymdb.ensure_schema()
    index_state.ensure_schema()


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sqlite_connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [row[0] for row in rows]


def sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({quote_ident(table)})")]


def sqlite_rows(conn: sqlite3.Connection, table: str, columns: list[str]) -> list[tuple]:
    names = ", ".join(quote_ident(column) for column in columns)
    return conn.execute(f"SELECT {names} FROM {quote_ident(table)}").fetchall()


def table_exists(pg: psycopg.Connection, schema: str, table: str) -> bool:
    return bool(
        pg.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, table),
        ).fetchone()
    )


def has_sequence(pg: psycopg.Connection, schema: str, table: str) -> bool:
    return bool(
        pg.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = 'id' "
            "AND column_default LIKE 'nextval%%'",
            (schema, table),
        ).fetchone()
    )


def reset_sequence(pg: psycopg.Connection, schema: str, table: str) -> None:
    qualified = f"{schema}.{table}"
    query = sql.SQL(
        "SELECT setval(pg_get_serial_sequence(%s, 'id'), "
        "COALESCE((SELECT MAX(id) FROM {table}), 1), "
        "(SELECT MAX(id) IS NOT NULL FROM {table}))"
    ).format(
        table=sql.SQL("{}.{}").format(
            sql.Identifier(schema), sql.Identifier(table)
        )
    )
    pg.execute(query, (qualified,))


def main() -> int:
    missing = [path for _, path, _ in SOURCES if not os.path.exists(path)]
    if missing:
        print("Нет исходных файлов: " + ", ".join(missing))
        return 2
    if not config.DATABASE_URL:
        print("DATABASE_URL не задан")
        return 2

    ensure_target()

    with psycopg.connect(config.DATABASE_URL) as pg:
        for schema, path, order in SOURCES:
            source = sqlite_connect(path)
            try:
                tables = sqlite_tables(source)
                present = [name for name in order if name in tables]
                present += [name for name in tables if name not in present]
                targets = [name for name in present if table_exists(pg, schema, name)]
                if targets:
                    pg.execute(
                        sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                            sql.SQL(", ").join(
                                sql.SQL("{}.{}").format(
                                    sql.Identifier(schema), sql.Identifier(name)
                                )
                                for name in targets
                            )
                        )
                    )
                for name in present:
                    if not table_exists(pg, schema, name):
                        print(f"{schema}.{name}: нет таблицы в PostgreSQL, пропуск")
                        continue
                    columns = sqlite_columns(source, name)
                    rows = sqlite_rows(source, name, columns)
                    query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
                        sql.Identifier(schema),
                        sql.Identifier(name),
                        sql.SQL(", ").join(sql.Identifier(c) for c in columns),
                        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                    )
                    with pg.cursor() as cur:
                        cur.executemany(query, rows)
                    print(f"{schema}.{name}: {len(rows)}")
            finally:
                source.close()

        for schema, _, order in SOURCES:
            for name in order:
                if table_exists(pg, schema, name) and has_sequence(pg, schema, name):
                    reset_sequence(pg, schema, name)
        pg.commit()

    verify = psycopg.connect(config.DATABASE_URL)
    mismatches = 0
    print("\nСверка строк SQLite -> PostgreSQL")
    for schema, path, order in SOURCES:
        source = sqlite_connect(path)
        try:
            tables = sqlite_tables(source)
            for name in order:
                if name not in tables or not table_exists(verify, schema, name):
                    continue
                before = source.execute(
                    f"SELECT COUNT(*) FROM {quote_ident(name)}"
                ).fetchone()[0]
                after = verify.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                        sql.Identifier(schema), sql.Identifier(name)
                    )
                ).fetchone()[0]
                if before != after:
                    mismatches += 1
                mark = "ok" if before == after else "MISMATCH"
                print(f"{mark:8} {schema}.{name}: {before} -> {after}")
        finally:
            source.close()
    verify.close()

    if mismatches:
        print(f"\nРасхождений: {mismatches}")
        return 1
    print("\nПеренос завершён без расхождений")
    return 0


if __name__ == "__main__":
    sys.exit(main())
