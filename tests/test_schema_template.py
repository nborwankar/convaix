import sqlite3

from convaix.backends.schema import render_schema, DIALECTS


def test_dialects_present():
    assert set(DIALECTS) == {"sqlite", "postgres"}


def test_sqlite_render_has_no_markers():
    sql = render_schema("sqlite")
    assert "{{" not in sql and "}}" not in sql


def test_sqlite_render_executes():
    sql = render_schema("sqlite")
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql)  # must not raise
    names = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"snapshots", "chunks", "discussions"} <= names
    conn.close()


def test_sqlite_omits_embedding_column():
    sql = render_schema("sqlite")
    assert "vector(768)" not in sql  # sqlite uses a separate vec0 virtual table


def test_postgres_render_has_pgvector():
    sql = render_schema("postgres")
    assert "{{" not in sql
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "embedding      vector(768)" in sql
    assert "USING hnsw" in sql
