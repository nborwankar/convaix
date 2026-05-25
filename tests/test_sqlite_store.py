import pytest

from convaix.backends import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "test.db"))
    yield s
    s.close()


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


def test_init_creates_tables(store):
    names = {
        r[0]
        for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"snapshots", "chunks"} <= names


def test_load_snapshot(store):
    assert store.load_snapshot(_sample_conv()) is True


def test_load_snapshot_duplicate_rejected(store):
    store.load_snapshot(_sample_conv())
    assert store.load_snapshot(_sample_conv()) is False


def test_list_snapshots(store):
    store.load_snapshot(_sample_conv("cx_001", "conv_aaa"))
    store.load_snapshot(_sample_conv("cx_002", "conv_aaa"))
    store.load_snapshot(_sample_conv("cx_003", "conv_bbb"))
    assert len(store.list_snapshots()) == 3


def test_list_snapshots_filter_source(store):
    store.load_snapshot(_sample_conv("cx_001"))
    assert len(store.list_snapshots(source="chatgpt")) == 1
    assert len(store.list_snapshots(source="claude")) == 0


def test_get_snapshot(store):
    store.load_snapshot(_sample_conv("cx_get_test"))
    row = store.get_snapshot("cx_get_test")
    assert row is not None and row["title"] == "Test Conversation"
    assert row["convaix_id"] == "cx_get_test"


def test_get_history(store):
    store.load_snapshot(_sample_conv("cx_001", "conv_aaa"))
    store.load_snapshot(_sample_conv("cx_002", "conv_aaa"))
    assert len(store.get_history("conv_aaa")) == 2


def test_chunk_and_embed_stores_chunks(store):
    store.load_snapshot(_sample_conv("cx_chunk_test"))
    count = store.chunk_and_embed(_sample_conv("cx_chunk_test"), skip_embeddings=True)
    assert count > 0
    chunks = store.get_chunks("cx_chunk_test")
    assert len(chunks) > 0 and chunks[0]["role"] in ("user", "assistant")


def test_export_snapshot_roundtrips(store):
    store.load_snapshot(_sample_conv("cx_exp"))
    exported = store.export_snapshot("cx_exp")
    assert exported["conversation"]["title"] == "Test Conversation"
    assert store.export_snapshot("cx_missing") is None


def test_keyword_search(store):
    store.load_snapshot(_sample_conv("cx_kw"))
    store.chunk_and_embed(_sample_conv("cx_kw"), skip_embeddings=True)
    rows = store.keyword_search("Hello world")
    assert any("Hello world" in r["chunk_text"] for r in rows)


def test_conversation_search(store):
    store.load_snapshot(_sample_conv("cx_cs"))
    store.chunk_and_embed(_sample_conv("cx_cs"), skip_embeddings=True)
    rows = store.conversation_search("Hello")
    assert rows and rows[0]["convaix_id"] == "cx_cs"
