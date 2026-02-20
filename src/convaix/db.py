"""SQLite3 + sqlite-vec database for conversation storage.

Handles schema creation, snapshot loading, and query helpers.
"""

import json
import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.expanduser("~/.convaix/convaix.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    convaix_id  TEXT PRIMARY KEY,
    conv_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    source_id   TEXT,
    model       TEXT,
    created_at  TEXT,
    published_at TEXT,
    author      TEXT,
    tags        TEXT DEFAULT '[]',
    raw         TEXT NOT NULL,
    turn_count  INTEGER NOT NULL DEFAULT 0,
    total_chars INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_snapshots_conv_id ON snapshots(conv_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_author ON snapshots(author);
CREATE INDEX IF NOT EXISTS idx_snapshots_source ON snapshots(source);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    convaix_id   TEXT NOT NULL REFERENCES snapshots(convaix_id),
    turn_number  INTEGER NOT NULL,
    chunk_number INTEGER NOT NULL,
    role         TEXT NOT NULL,
    chunk_text   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(convaix_id, turn_number, chunk_number)
);

CREATE TABLE IF NOT EXISTS discussions (
    discussion_id TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    created_by    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discussion_refs (
    discussion_id TEXT NOT NULL REFERENCES discussions(discussion_id),
    convaix_id    TEXT NOT NULL REFERENCES snapshots(convaix_id),
    PRIMARY KEY (discussion_id, convaix_id)
);

CREATE TABLE IF NOT EXISTS discussion_messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    discussion_id TEXT NOT NULL REFERENCES discussions(discussion_id),
    author        TEXT NOT NULL,
    content       TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""


def init_db(db_path=None):
    """Create database and tables. Returns connection.

    Also initializes sqlite-vec virtual table if available.
    """
    db_path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)

    # Try to load sqlite-vec for vector search
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                embedding float[768]
            )
        """
        )
        logger.debug("sqlite-vec loaded, chunks_vec table ready")
    except (ImportError, Exception) as e:
        logger.debug(f"sqlite-vec not available: {e} (keyword search only)")

    conn.commit()
    return conn


def load_snapshot(conn, conv_data):
    """Load a conversation snapshot into the database.

    Expects conv_data with x-convaix block containing convaix_id.
    Returns True if loaded, False if duplicate convaix_id.
    """
    ext = conv_data.get("x-convaix", {})
    convaix_id = ext.get("convaix_id")
    if not convaix_id:
        logger.warning("No convaix_id in x-convaix block, skipping")
        return False

    conv = conv_data["conversation"]
    stats = conv_data.get("statistics", {})
    author = ext.get("author", {}).get("handle", "")

    try:
        conn.execute(
            """INSERT INTO snapshots
               (convaix_id, conv_id, title, source, source_id, model,
                created_at, published_at, author, tags, raw,
                turn_count, total_chars)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                convaix_id,
                conv.get("id", ""),
                conv.get("title", "Untitled"),
                conv.get("source", "unknown"),
                conv.get("source_id"),
                conv.get("model"),
                conv.get("created_at"),
                ext.get("published_at"),
                author,
                json.dumps(conv.get("tags", [])),
                json.dumps(conv_data),
                stats.get("turn_count", 0),
                stats.get("total_chars", 0),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def list_snapshots(conn, source=None, author=None, limit=1000):
    """List snapshots. Returns list of Row objects."""
    query = "SELECT convaix_id, conv_id, title, source, author, published_at, turn_count FROM snapshots"
    params = []
    conditions = []

    if source:
        conditions.append("source = ?")
        params.append(source)
    if author:
        conditions.append("author = ?")
        params.append(author)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY title LIMIT ?"
    params.append(limit)

    return conn.execute(query, params).fetchall()


def get_snapshot(conn, convaix_id):
    """Get a single snapshot by convaix_id. Returns Row or None."""
    return conn.execute(
        "SELECT * FROM snapshots WHERE convaix_id = ?", (convaix_id,)
    ).fetchone()


def get_snapshot_history(conn, conv_id):
    """Get all snapshots for a conv_id lineage, ordered by published_at."""
    return conn.execute(
        """SELECT convaix_id, conv_id, title, source, author, published_at, turn_count
           FROM snapshots WHERE conv_id = ? ORDER BY published_at""",
        (conv_id,),
    ).fetchall()


def chunk_snapshot(conn, conv_data, skip_embeddings=False):
    """Split snapshot turns into paragraph chunks and optionally embed them.

    Returns number of chunks stored.
    """
    import hashlib

    from .chunking import split_into_chunks

    ext = conv_data.get("x-convaix", {})
    convaix_id = ext.get("convaix_id")
    title = conv_data["conversation"].get("title", "")
    turns = conv_data.get("turns", [])

    chunk_data = []
    for turn in turns:
        content = turn.get("content", "")
        role = turn.get("role", "user")
        turn_number = turn.get("turn_number", 0)
        paragraphs = split_into_chunks(content)
        for j, paragraph in enumerate(paragraphs):
            chunk_number = j + 1
            content_hash = hashlib.sha256(paragraph.encode()).hexdigest()
            chunk_data.append(
                (convaix_id, turn_number, chunk_number, role, paragraph, content_hash)
            )

    if not chunk_data:
        return 0

    stored = 0
    for row in chunk_data:
        try:
            conn.execute(
                """INSERT INTO chunks
                   (convaix_id, turn_number, chunk_number, role, chunk_text, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                row,
            )
            stored += 1
        except sqlite3.IntegrityError:
            pass  # already exists

    # Embed if requested and sqlite-vec is available
    if not skip_embeddings and stored > 0:
        _embed_chunks(conn, convaix_id, title, chunk_data)

    conn.commit()
    return stored


def _embed_chunks(conn, convaix_id, title, chunk_data):
    """Generate and store embeddings for chunks."""
    try:
        from .embeddings import embed_texts
    except ImportError:
        logger.debug("Embeddings not available (install convaix[embeddings])")
        return

    # Check if chunks_vec table exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    )
    if not cur.fetchone():
        logger.debug("chunks_vec table not found, skipping embeddings")
        return

    # Build prefixed texts
    texts = []
    chunk_ids = []
    for convaix_id_val, turn_number, chunk_number, role, paragraph, _ in chunk_data:
        prefixed = f"[{title}] {role}: {paragraph}"
        texts.append(prefixed)
        # Get the chunk row id
        row = conn.execute(
            "SELECT id FROM chunks WHERE convaix_id=? AND turn_number=? AND chunk_number=?",
            (convaix_id_val, turn_number, chunk_number),
        ).fetchone()
        if row:
            chunk_ids.append(row["id"])

    if not texts:
        return

    embeddings = embed_texts(texts)
    for chunk_id, emb in zip(chunk_ids, embeddings):
        conn.execute(
            "INSERT OR REPLACE INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
            (chunk_id, json.dumps(emb)),
        )


def get_chunks(conn, convaix_id):
    """Get all chunks for a snapshot, ordered by turn and chunk number."""
    return conn.execute(
        """SELECT * FROM chunks
           WHERE convaix_id = ?
           ORDER BY turn_number, chunk_number""",
        (convaix_id,),
    ).fetchall()
