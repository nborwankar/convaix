"""Embedding utilities for conversation search.

Backend-selectable via CONVAIX_EMBED_BACKEND: 'auto' (default), 'mlx', or
'sentence-transformers'. 'auto' tries MLX (Apple Silicon GPU) and falls back to
sentence-transformers if MLX is unavailable. Lazy-loads the model on first use.
"""

import logging
import os

os.environ.setdefault("USE_TF", "0")

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768
_MLX_MODEL = "nomic-text-v1.5"
_ST_MODEL = "nomic-ai/nomic-embed-text-v1.5"
# Public constant kept for backward compatibility (tests import EMBEDDING_MODEL).
EMBEDDING_MODEL = _ST_MODEL

_model = None
_backend = None


def _select_backend():
    """Resolve the embedding backend name without loading a model."""
    pref = os.environ.get("CONVAIX_EMBED_BACKEND", "auto").lower()
    if pref == "mlx":
        return "mlx"
    if pref in ("sentence-transformers", "st"):
        return "sentence-transformers"
    # auto: prefer MLX, fall back to sentence-transformers on import failure
    try:
        import mlx_embedding_models.embedding  # noqa: F401

        return "mlx"
    except ImportError as e:
        logger.info("MLX unavailable (%s); using sentence-transformers", e)
        return "sentence-transformers"


def _load_mlx():
    import mlx_embedding_models.embedding as _mlx_emb

    extended = sorted(
        set(_mlx_emb.SEQ_LENS + [640, 768, 896, 1024, 1280, 1536, 1792, 2048])
    )
    _mlx_emb.SEQ_LENS = extended
    from mlx_embedding_models.embedding import EmbeddingModel

    logger.info("Loading %s (MLX)...", _MLX_MODEL)
    return EmbeddingModel.from_registry(_MLX_MODEL)


def _load_st():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading %s (sentence-transformers)...", _ST_MODEL)
    return SentenceTransformer(_ST_MODEL, trust_remote_code=True)


def get_model():
    """Load the embedding model for the resolved backend on first use."""
    global _model, _backend
    if _model is None:
        _backend = _select_backend()
        _model = _load_mlx() if _backend == "mlx" else _load_st()
        logger.info("Embedding backend: %s (dim=%d)", _backend, EMBEDDING_DIM)
    return _model


def _encode(prefixed):
    model = get_model()
    if _backend == "mlx":
        return model.encode(prefixed, batch_size=64, show_progress=False).tolist()
    return model.encode(
        prefixed, show_progress_bar=False, normalize_embeddings=True
    ).tolist()


def embed_texts(texts, batch_size=64):
    """Embed documents (with the nomic 'search_document:' prefix) for storage."""
    return _encode([f"search_document: {t}" for t in texts])


def embed_query(text):
    """Embed a single query (with the nomic 'search_query:' prefix) for retrieval."""
    return _encode([f"search_query: {text}"])[0]
