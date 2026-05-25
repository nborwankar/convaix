"""Hybrid search over a ConversationStore.

Backend-agnostic: the store provides raw keyword/semantic/conversation queries;
this module embeds the query, merges, dedups, and ranks.
"""

import logging

logger = logging.getLogger(__name__)


def search_chunks(store, query, source=None, limit=10, mode="hybrid"):
    """Hybrid chunk search. Returns ranked dicts with a 'match_type' field."""
    results = {}

    if mode in ("keyword", "hybrid"):
        for r in store.keyword_search(query, source=source, limit=limit):
            results[(r["convaix_id"], r["chunk_text"][:100])] = {
                **r,
                "match_type": "kw",
            }

    if mode in ("semantic", "hybrid"):
        try:
            from .embeddings import embed_query

            vector = embed_query(query)
        except Exception as e:  # embeddings optional / model load failure — log
            logger.debug("semantic search unavailable: %s", e)
            vector = None
        if vector is not None:
            for r in store.semantic_search(vector, source=source, limit=limit):
                key = (r["convaix_id"], r["chunk_text"][:100])
                if key in results:
                    results[key]["match_type"] = "both"
                    results[key]["similarity"] = max(
                        results[key]["similarity"], r["similarity"]
                    )
                else:
                    results[key] = {**r, "match_type": "sem"}

    ranked = sorted(results.values(), key=lambda r: r["similarity"], reverse=True)
    return ranked[:limit]


def search_conversations(store, query, source=None, limit=20):
    """Conversation-level keyword search with hit counts."""
    return store.conversation_search(query, source=source, limit=limit)
