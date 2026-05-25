"""SQLite + sqlite-vec implementation of ConversationStore."""

import hashlib
import json
import logging
import os
import sqlite3

from .base import ConversationStore
from .schema import render_schema
from ..chunking import split_into_chunks

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = os.path.expanduser("~/.convaix/convaix.db")


class SQLiteStore(ConversationStore):
    def __init__(self, db_path=None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.conn = None
        self._vec = False
        self.init_db()

    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(render_schema("sqlite"))
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[768])"
            )
            self._vec = True
        except Exception as e:  # sqlite-vec optional — log, don't crash
            logger.debug("sqlite-vec unavailable: %s (keyword-only)", e)
            self._vec = False
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    # ── write ──
    def load_snapshot(self, conv_data):
        ext = conv_data.get("x-convaix", {})
        convaix_id = ext.get("convaix_id")
        if not convaix_id:
            logger.warning("No convaix_id in x-convaix block, skipping")
            return False
        conv = conv_data["conversation"]
        stats = conv_data.get("statistics", {})
        author = ext.get("author", {}).get("handle", "")
        try:
            self.conn.execute(
                """INSERT INTO snapshots
                   (convaix_id, conv_id, title, source, source_id, model,
                    created_at, published_at, author, tags, raw, turn_count, total_chars)
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
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def chunk_and_embed(self, conv_data, skip_embeddings=False):
        ext = conv_data.get("x-convaix", {})
        convaix_id = ext.get("convaix_id")
        title = conv_data["conversation"].get("title", "")
        rows = []
        for turn in conv_data.get("turns", []):
            content = turn.get("content", "")
            role = turn.get("role", "user")
            tn = turn.get("turn_number", 0)
            for j, para in enumerate(split_into_chunks(content)):
                rows.append(
                    (
                        convaix_id,
                        tn,
                        j + 1,
                        role,
                        para,
                        hashlib.sha256(para.encode()).hexdigest(),
                    )
                )
        if not rows:
            return 0
        stored = 0
        for r in rows:
            try:
                self.conn.execute(
                    """INSERT INTO chunks
                       (convaix_id, turn_number, chunk_number, role, chunk_text, content_hash)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    r,
                )
                stored += 1
            except sqlite3.IntegrityError:
                pass
        if not skip_embeddings and stored and self._vec:
            self._embed_chunks(convaix_id, title, rows)
        self.conn.commit()
        return stored

    def _embed_chunks(self, convaix_id, title, rows):
        from ..embeddings import embed_texts

        texts, ids = [], []
        for cid, tn, cn, role, para, _ in rows:
            texts.append(f"[{title}] {role}: {para}")
            row = self.conn.execute(
                "SELECT id FROM chunks WHERE convaix_id=? AND turn_number=? AND chunk_number=?",
                (cid, tn, cn),
            ).fetchone()
            if row:
                ids.append(row["id"])
        if not texts:
            return
        for cid_, emb in zip(ids, embed_texts(texts)):
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                (cid_, json.dumps(emb)),
            )

    # ── read ──
    def get_snapshot(self, snapshot_id):
        row = self.conn.execute(
            "SELECT * FROM snapshots WHERE convaix_id = ?", (snapshot_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_history(self, conv_id):
        rows = self.conn.execute(
            """SELECT convaix_id, conv_id, title, source, author, published_at, turn_count
               FROM snapshots WHERE conv_id = ? ORDER BY published_at""",
            (conv_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_snapshots(self, source=None, author=None, limit=1000):
        q = (
            "SELECT convaix_id, conv_id, title, source, author, published_at, turn_count "
            "FROM snapshots"
        )
        conds, params = [], []
        if source:
            conds.append("source = ?")
            params.append(source)
        if author:
            conds.append("author = ?")
            params.append(author)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY title LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def get_chunks(self, snapshot_id):
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE convaix_id = ? ORDER BY turn_number, chunk_number",
            (snapshot_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── raw search ──
    def keyword_search(self, query_text, source=None, limit=10):
        like = f"%{query_text}%"
        params = [like, like]
        sf = ""
        if source:
            sf = "AND s.source = ?"
            params.append(source)
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT c.role, c.chunk_text, s.title, s.source, s.convaix_id, 1.0 AS similarity
                FROM chunks c JOIN snapshots s ON c.convaix_id = s.convaix_id
                WHERE (c.chunk_text LIKE ? OR s.title LIKE ?) {sf} LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def semantic_search(self, query_vector, source=None, limit=10):
        if not self._vec:
            return []
        if self.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0:
            return []
        hits = self.conn.execute(
            """SELECT v.rowid, v.distance FROM chunks_vec v
               WHERE embedding MATCH ? ORDER BY distance LIMIT ?""",
            (json.dumps(query_vector), limit),
        ).fetchall()
        out = []
        for h in hits:
            c = self.conn.execute(
                """SELECT c.role, c.chunk_text, c.convaix_id, s.title, s.source
                   FROM chunks c JOIN snapshots s ON c.convaix_id = s.convaix_id
                   WHERE c.id = ?""",
                (h["rowid"],),
            ).fetchone()
            if c and (source is None or c["source"] == source):
                d = dict(c)
                d["similarity"] = 1.0 - h["distance"]
                out.append(d)
        return out

    def conversation_search(self, query_text, source=None, limit=20):
        like = f"%{query_text}%"
        params = [like, like]
        sf = ""
        if source:
            sf = "AND s.source = ?"
            params.append(source)
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT s.title, s.source, s.convaix_id, COUNT(*) AS hits, s.turn_count
                FROM chunks c JOIN snapshots s ON c.convaix_id = s.convaix_id
                WHERE (c.chunk_text LIKE ? OR s.title LIKE ?) {sf}
                GROUP BY s.convaix_id ORDER BY hits DESC LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ── export ──
    def export_snapshot(self, snapshot_id):
        row = self.conn.execute(
            "SELECT raw FROM snapshots WHERE convaix_id = ?", (snapshot_id,)
        ).fetchone()
        return json.loads(row["raw"]) if row else None
