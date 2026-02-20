"""Embedding utilities for conversation search.

Uses mlx-embedding-models for native Apple Silicon GPU acceleration.
Lazy-loads nomic-embed-text-v1.5 on first use.
"""

import logging
import os

os.environ.setdefault("USE_TF", "0")

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-text-v1.5"
EMBEDDING_DIM = 768

_model = None


def _patch_seq_lens():
    """Extend mlx-embedding-models SEQ_LENS for nomic's full 2048-token context."""
    import mlx_embedding_models.embedding as _mlx_emb

    _EXTENDED = sorted(
        set(_mlx_emb.SEQ_LENS + [640, 768, 896, 1024, 1280, 1536, 1792, 2048])
    )
    _mlx_emb.SEQ_LENS = _EXTENDED
    logger.debug(f"Patched SEQ_LENS: max={_EXTENDED[-1]} ({len(_EXTENDED)} buckets)")


def get_model():
    """Load nomic-embed-text-v1.5 via MLX on first use."""
    global _model
    if _model is None:
        _patch_seq_lens()
        from mlx_embedding_models.embedding import EmbeddingModel

        logger.info(f"Loading {EMBEDDING_MODEL} (MLX)...")
        _model = EmbeddingModel.from_registry(EMBEDDING_MODEL)
        logger.info(f"Model loaded (dim={EMBEDDING_DIM})")
    return _model


def embed_texts(texts, batch_size=64):
    """Generate embeddings with search_document: prefix for storage."""
    model = get_model()
    prefixed = [f"search_document: {t}" for t in texts]
    embeddings = model.encode(prefixed, batch_size=batch_size, show_progress=False)
    return embeddings.tolist()


def embed_query(text):
    """Generate a single embedding with search_query: prefix for retrieval."""
    model = get_model()
    embeddings = model.encode([f"search_query: {text}"], show_progress=False)
    return embeddings.tolist()[0]
