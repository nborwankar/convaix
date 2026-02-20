"""Hybrid search across conversation snapshots.

Combines semantic similarity (sqlite-vec cosine) with keyword matching
(LIKE on chunk_text and title).
"""

import json
import logging

logger = logging.getLogger(__name__)


def search_chunks(conn, query, source=None, limit=10, mode="hybrid"):
    """Search chunks by keyword, semantic, or hybrid mode.

    Returns list of dicts with: chunk_text, role, title, source,
    convaix_id, similarity, match_type.
    """
    results = {}

    # Keyword search
    if mode in ("keyword", "hybrid"):
        kw_results = _keyword_search(conn, query, source, limit)
        for r in kw_results:
            key = (r["convaix_id"], r["chunk_text"][:100])
            results[key] = {**r, "match_type": "kw"}

    # Semantic search
    if mode in ("semantic", "hybrid"):
        sem_results = _semantic_search(conn, query, source, limit)
        for r in sem_results:
            key = (r["convaix_id"], r["chunk_text"][:100])
            if key in results:
                results[key]["match_type"] = "both"
                results[key]["similarity"] = max(
                    results[key]["similarity"], r["similarity"]
                )
            else:
                results[key] = {**r, "match_type": "sem"}

    # Sort by similarity descending
    sorted_results = sorted(
        results.values(), key=lambda r: r["similarity"], reverse=True
    )
    return sorted_results[:limit]


def search_conversations(conn, query, source=None, limit=20):
    """Conversation-level keyword search with hit counts.

    Returns list of dicts with: title, source, convaix_id, hits, turn_count.
    """
    kw_pattern = f"%{query}%"
    params = [kw_pattern, kw_pattern]
    source_filter = ""

    if source:
        source_filter = "AND s.source = ?"
        params.append(source)

    params.append(limit)

    rows = conn.execute(
        f"""SELECT s.title, s.source, s.convaix_id,
                   COUNT(*) AS hits, s.turn_count
            FROM chunks c
            JOIN snapshots s ON c.convaix_id = s.convaix_id
            WHERE (c.chunk_text LIKE ? OR s.title LIKE ?)
            {source_filter}
            GROUP BY s.convaix_id
            ORDER BY hits DESC
            LIMIT ?""",
        params,
    ).fetchall()

    return [dict(r) for r in rows]


def _keyword_search(conn, query, source, limit):
    """LIKE-based keyword search on chunk_text and title."""
    kw_pattern = f"%{query}%"
    params = [kw_pattern, kw_pattern]
    source_filter = ""

    if source:
        source_filter = "AND s.source = ?"
        params.append(source)

    params.append(limit)

    rows = conn.execute(
        f"""SELECT c.role, c.chunk_text, s.title, s.source, s.convaix_id,
                   1.0 AS similarity
            FROM chunks c
            JOIN snapshots s ON c.convaix_id = s.convaix_id
            WHERE (c.chunk_text LIKE ? OR s.title LIKE ?)
            {source_filter}
            LIMIT ?""",
        params,
    ).fetchall()

    return [dict(r) for r in rows]


def _semantic_search(conn, query, source, limit):
    """sqlite-vec cosine similarity search."""
    try:
        from .embeddings import embed_query
    except ImportError:
        logger.debug("Embeddings not available, skipping semantic search")
        return []

    # Check if chunks_vec exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    )
    if not cur.fetchone():
        return []

    # Check if there are any embeddings
    cur = conn.execute("SELECT COUNT(*) FROM chunks_vec")
    if cur.fetchone()[0] == 0:
        return []

    query_emb = embed_query(query)

    rows = conn.execute(
        """SELECT v.rowid, v.distance
           FROM chunks_vec v
           WHERE embedding MATCH ?
           ORDER BY distance
           LIMIT ?""",
        (json.dumps(query_emb), limit),
    ).fetchall()

    results = []
    for row in rows:
        chunk = conn.execute(
            """SELECT c.role, c.chunk_text, c.convaix_id, s.title, s.source
               FROM chunks c
               JOIN snapshots s ON c.convaix_id = s.convaix_id
               WHERE c.id = ?""",
            (row["rowid"],),
        ).fetchone()

        if chunk:
            results.append(
                {
                    "role": chunk["role"],
                    "chunk_text": chunk["chunk_text"],
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "convaix_id": chunk["convaix_id"],
                    "similarity": 1.0 - row["distance"],
                }
            )

    return results
