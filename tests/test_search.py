import pytest

from convaix.backends import SQLiteStore
from convaix.search import search_chunks, search_conversations


@pytest.fixture
def loaded_store(tmp_path):
    s = SQLiteStore(str(tmp_path / "search_test.db"))
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
    s.load_snapshot(data)
    s.chunk_and_embed(data, skip_embeddings=True)
    yield s
    s.close()


def test_keyword_search(loaded_store):
    results = search_chunks(loaded_store, "Riemannian", mode="keyword")
    assert len(results) > 0
    assert any("Riemannian" in r["chunk_text"] for r in results)


def test_keyword_search_title_match(loaded_store):
    results = search_chunks(loaded_store, "Manifold Geometry", mode="keyword")
    assert len(results) > 0


def test_keyword_search_no_results(loaded_store):
    results = search_chunks(loaded_store, "zzznonexistent", mode="keyword")
    assert len(results) == 0


def test_conversation_search(loaded_store):
    results = search_conversations(loaded_store, "Riemannian")
    assert len(results) > 0
    assert results[0]["title"] == "Manifold Geometry Discussion"
