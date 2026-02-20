import pytest
from convaix.db import init_db, load_snapshot, chunk_snapshot
from convaix.search import search_chunks, search_conversations


@pytest.fixture
def loaded_db(tmp_path):
    db_path = str(tmp_path / "search_test.db")
    conn = init_db(db_path)
    data = {
        "schema_version": "1.0",
        "conversation": {
            "id": "conv_search",
            "title": "Manifold Geometry Discussion",
            "source": "claude",
            "source_id": "search_test",
            "exported_at": "2026-02-18T00:00:00Z",
        },
        "turns": [
            {
                "turn_number": 1,
                "role": "user",
                "content": "Tell me about Riemannian manifolds and their applications in machine learning",
            },
            {
                "turn_number": 2,
                "role": "assistant",
                "content": "Riemannian manifolds are smooth manifolds equipped with a Riemannian metric that defines inner products on tangent spaces.",
            },
        ],
        "statistics": {
            "turn_count": 2,
            "user_turns": 1,
            "assistant_turns": 1,
            "total_chars": 100,
        },
        "x-convaix": {
            "convaix_id": "cx_search_test",
            "version": "0.1",
            "conv_id": "conv_search",
            "author": {"handle": "testuser"},
            "published_at": "2026-02-18T00:00:00Z",
            "parent_refs": [],
            "annotations": [],
            "signature": None,
        },
    }
    load_snapshot(conn, data)
    chunk_snapshot(conn, data, skip_embeddings=True)
    return conn


def test_keyword_search(loaded_db):
    results = search_chunks(loaded_db, "Riemannian", mode="keyword")
    assert len(results) > 0
    assert any("Riemannian" in r["chunk_text"] for r in results)


def test_keyword_search_title_match(loaded_db):
    results = search_chunks(loaded_db, "Manifold Geometry", mode="keyword")
    assert len(results) > 0


def test_keyword_search_no_results(loaded_db):
    results = search_chunks(loaded_db, "zzznonexistent", mode="keyword")
    assert len(results) == 0


def test_conversation_search(loaded_db):
    results = search_conversations(loaded_db, "Riemannian")
    assert len(results) > 0
    assert results[0]["title"] == "Manifold Geometry Discussion"
