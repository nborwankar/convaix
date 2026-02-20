import pytest
from convaix.embeddings import embed_texts, EMBEDDING_DIM


@pytest.mark.slow
def test_embed_single_text():
    result = embed_texts(["Hello world"])
    assert len(result) == 1
    assert len(result[0]) == EMBEDDING_DIM


@pytest.mark.slow
def test_embed_multiple():
    result = embed_texts(["Hello", "World", "Test"])
    assert len(result) == 3
    for emb in result:
        assert len(emb) == EMBEDDING_DIM


@pytest.mark.slow
def test_embed_query():
    from convaix.embeddings import embed_query

    result = embed_query("test query")
    assert len(result) == EMBEDDING_DIM
