"""convaix storage backends."""

from .base import ConversationStore
from .sqlite import SQLiteStore, DEFAULT_DB_PATH


def open_store(db_path=None, backend="sqlite"):
    """Factory: open a ConversationStore. (postgres added in 2b)"""
    if backend == "sqlite":
        return SQLiteStore(db_path)
    raise ValueError(f"Unknown backend: {backend!r}")


__all__ = ["ConversationStore", "SQLiteStore", "open_store", "DEFAULT_DB_PATH"]
