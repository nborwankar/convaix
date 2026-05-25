"""Abstract storage interface implemented by each backend."""

from abc import ABC, abstractmethod


class ConversationStore(ABC):
    """Backend-agnostic conversation storage + raw search queries.

    Search methods return raw, per-backend result rows (list[dict]); the
    `convaix.search` module performs merge/dedup/ranking on top.
    """

    # ── lifecycle ──
    @abstractmethod
    def init_db(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    # ── write ──
    @abstractmethod
    def load_snapshot(self, conv_data: dict) -> bool: ...

    @abstractmethod
    def chunk_and_embed(
        self, conv_data: dict, skip_embeddings: bool = False
    ) -> int: ...

    # ── read ──
    @abstractmethod
    def get_snapshot(self, snapshot_id: str) -> dict | None: ...

    @abstractmethod
    def get_history(self, conv_id: str) -> list[dict]: ...

    @abstractmethod
    def list_snapshots(
        self, source=None, author=None, limit: int = 1000
    ) -> list[dict]: ...

    @abstractmethod
    def get_chunks(self, snapshot_id: str) -> list[dict]: ...

    # ── raw search ──
    @abstractmethod
    def keyword_search(
        self, query_text: str, source=None, limit: int = 10
    ) -> list[dict]: ...

    @abstractmethod
    def semantic_search(
        self, query_vector: list, source=None, limit: int = 10
    ) -> list[dict]: ...

    @abstractmethod
    def conversation_search(
        self, query_text: str, source=None, limit: int = 20
    ) -> list[dict]: ...

    # ── export ──
    @abstractmethod
    def export_snapshot(self, snapshot_id: str) -> dict | None: ...
