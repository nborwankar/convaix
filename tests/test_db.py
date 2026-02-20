import pytest
from convaix.db import (
    init_db,
    load_snapshot,
    list_snapshots,
    get_snapshot,
    chunk_snapshot,
    get_chunks,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


def _sample_conv(convaix_id="cx_test-1234", conv_id="conv_abc123"):
    return {
        "schema_version": "1.0",
        "conversation": {
            "id": conv_id,
            "title": "Test Conversation",
            "source": "chatgpt",
            "source_id": "xyz789",
            "source_url": None,
            "model": "gpt-4",
            "created_at": "2026-02-18T10:00:00Z",
            "exported_at": "2026-02-18T12:00:00Z",
            "tags": ["test"],
            "metadata": {},
        },
        "turns": [
            {"turn_number": 1, "role": "user", "content": "Hello world"},
            {
                "turn_number": 2,
                "role": "assistant",
                "content": "Hi there! How can I help?",
            },
        ],
        "statistics": {
            "turn_count": 2,
            "user_turns": 1,
            "assistant_turns": 1,
            "total_chars": 37,
        },
        "x-convaix": {
            "convaix_id": convaix_id,
            "version": "0.1",
            "conv_id": conv_id,
            "author": {"handle": "testuser", "key_id": None},
            "published_at": "2026-02-18T14:00:00Z",
            "parent_refs": [],
            "annotations": [],
            "signature": None,
        },
    }


def test_init_db_creates_tables(db_path):
    conn = init_db(db_path)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert "snapshots" in tables
    assert "chunks" in tables
    conn.close()


def test_load_snapshot(db_path):
    conn = init_db(db_path)
    data = _sample_conv()
    result = load_snapshot(conn, data)
    assert result is True
    conn.close()


def test_load_snapshot_duplicate_rejected(db_path):
    conn = init_db(db_path)
    data = _sample_conv()
    load_snapshot(conn, data)
    result = load_snapshot(conn, data)
    assert result is False
    conn.close()


def test_list_snapshots(db_path):
    conn = init_db(db_path)
    load_snapshot(conn, _sample_conv("cx_001", "conv_aaa"))
    load_snapshot(conn, _sample_conv("cx_002", "conv_aaa"))
    load_snapshot(conn, _sample_conv("cx_003", "conv_bbb"))
    rows = list_snapshots(conn)
    assert len(rows) == 3
    conn.close()


def test_list_snapshots_filter_source(db_path):
    conn = init_db(db_path)
    load_snapshot(conn, _sample_conv("cx_001"))
    rows = list_snapshots(conn, source="chatgpt")
    assert len(rows) == 1
    rows = list_snapshots(conn, source="claude")
    assert len(rows) == 0
    conn.close()


def test_get_snapshot(db_path):
    conn = init_db(db_path)
    data = _sample_conv("cx_get_test")
    load_snapshot(conn, data)
    row = get_snapshot(conn, "cx_get_test")
    assert row is not None
    assert row["title"] == "Test Conversation"
    assert row["convaix_id"] == "cx_get_test"
    conn.close()


def test_chunk_snapshot_stores_chunks(db_path):
    conn = init_db(db_path)
    data = _sample_conv("cx_chunk_test")
    load_snapshot(conn, data)
    count = chunk_snapshot(conn, data, skip_embeddings=True)
    assert count > 0
    chunks = get_chunks(conn, "cx_chunk_test")
    assert len(chunks) > 0
    assert chunks[0]["role"] in ("user", "assistant")
    conn.close()
