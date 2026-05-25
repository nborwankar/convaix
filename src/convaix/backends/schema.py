"""Schema template rendering: one logical schema, two SQL dialects."""

from pathlib import Path

_TEMPLATE = (Path(__file__).parent / "schema.sql.template").read_text()

DIALECTS = {
    "sqlite": {
        "vector_extension": "",
        "timestamp_type": "TEXT",
        "array_type": "TEXT",
        "auto_id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "embedding_column": "",
        "vector_index": "-- sqlite-vec virtual table created at runtime",
    },
    "postgres": {
        "vector_extension": "CREATE EXTENSION IF NOT EXISTS vector;",
        "timestamp_type": "TIMESTAMPTZ",
        "array_type": "TEXT[]",
        "auto_id": "SERIAL PRIMARY KEY",
        "embedding_column": "\n    embedding      vector(768),",
        "vector_index": (
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
            "ON chunks USING hnsw (embedding vector_cosine_ops);"
        ),
    },
}


def render_schema(dialect):
    """Return executable DDL for 'sqlite' or 'postgres'."""
    if dialect not in DIALECTS:
        raise ValueError(
            f"Unknown dialect: {dialect!r} (expected one of {set(DIALECTS)})"
        )
    sql = _TEMPLATE
    for key, val in DIALECTS[dialect].items():
        sql = sql.replace("{{" + key + "}}", val)
    return sql
