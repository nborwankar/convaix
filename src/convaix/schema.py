"""Conversation schema utilities.

Provides slugification, ID generation, role normalization, schema
conversion, and x-convaix extension support.
"""

import hashlib
import re
import uuid
from datetime import datetime, timezone

ROLE_MAP = {
    "model": "assistant",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "human": "user",
}


def slugify(text, max_len=60):
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


def generate_conv_id(source, source_id):
    """Generate a stable, platform-agnostic conversation ID."""
    raw = f"{source}:{source_id}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"conv_{digest}"


def generate_convaix_id():
    """Generate a globally unique convaix snapshot ID."""
    return f"cx_{uuid.uuid4()}"


def convert_to_schema(
    source,
    source_id,
    title,
    turns,
    source_url=None,
    model=None,
    created_at=None,
    extra_metadata=None,
):
    """Convert provider-specific data to schema-compliant format."""
    schema_turns = []
    for i, turn in enumerate(turns):
        raw_role = turn.get("role", "unknown")
        role = ROLE_MAP.get(raw_role, raw_role)
        schema_turns.append(
            {
                "turn_number": i + 1,
                "role": role,
                "content": turn.get("content", ""),
                "timestamp": turn.get("timestamp"),
                "attachments": turn.get("attachments", []),
                "metadata": turn.get("metadata", {}),
            }
        )

    user_turns = sum(1 for t in schema_turns if t["role"] == "user")
    assistant_turns = sum(1 for t in schema_turns if t["role"] == "assistant")
    total_chars = sum(len(t["content"]) for t in schema_turns)

    effective_source_id = source_id or hashlib.sha256(title.encode()).hexdigest()[:16]
    conv_id = generate_conv_id(source, effective_source_id)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    metadata = extra_metadata or {}

    return {
        "schema_version": "1.0",
        "conversation": {
            "id": conv_id,
            "title": title,
            "source": source,
            "source_id": source_id,
            "source_url": source_url,
            "model": model,
            "created_at": created_at,
            "exported_at": now_iso,
            "tags": [],
            "metadata": metadata,
        },
        "turns": schema_turns,
        "statistics": {
            "turn_count": len(schema_turns),
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "total_chars": total_chars,
        },
    }


def add_convaix_extension(conv_data, author_handle, parent_refs=None):
    """Add x-convaix extension block to a schema v1.0 conversation.

    Returns the modified conv_data (mutates in place and returns).
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conv_data["x-convaix"] = {
        "convaix_id": generate_convaix_id(),
        "version": "0.1",
        "conv_id": conv_data["conversation"]["id"],
        "author": {
            "handle": author_handle,
            "key_id": None,
        },
        "published_at": now_iso,
        "parent_refs": parent_refs or [],
        "annotations": [],
        "signature": None,
    }
    return conv_data
