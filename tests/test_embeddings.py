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


def test_force_st_backend(monkeypatch):
    monkeypatch.setenv("CONVAIX_EMBED_BACKEND", "sentence-transformers")
    from convaix import embeddings as e

    assert e._select_backend() == "sentence-transformers"


def test_auto_backend_resolves(monkeypatch):
    monkeypatch.setenv("CONVAIX_EMBED_BACKEND", "auto")
    from convaix import embeddings as e

    # auto defaults to the safe ST backend and must NOT import MLX
    # (a broken MLX install can abort() the process) — see bug convaix-fkk
    assert e._select_backend() == "sentence-transformers"


def test_force_mlx_backend(monkeypatch):
    monkeypatch.setenv("CONVAIX_EMBED_BACKEND", "mlx")
    from convaix import embeddings as e

    assert e._select_backend() == "mlx"
