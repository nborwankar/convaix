"""Schema validation for conversation JSON files.

Default `validate_conversation(data)` is lenient/structural (backward compatible).
`strict=True` additionally enforces turn_number sequencing, statistics cross-check,
conv_id format, source enum, and rejects content-quality artifacts.
"""

import re

VALID_ROLES = {"user", "assistant", "system"}
VALID_SOURCES = {"gemini", "chatgpt", "claude"}
REQUIRED_CONV_FIELDS = {"id", "title", "source", "exported_at"}
REQUIRED_TURN_FIELDS = {"turn_number", "role", "content"}
REQUIRED_TOP_LEVEL = {"schema_version", "conversation", "turns", "statistics"}
REQUIRED_STATS_FIELDS = {"turn_count", "user_turns", "assistant_turns", "total_chars"}

CONV_ID_RE = re.compile(r"^conv_[a-f0-9]{12}$")

DIRTY_CONTENT_PATTERNS = [
    (
        re.compile(r"^Show thinking\nGemini said\n", re.MULTILINE),
        "Combined Gemini UI artifact",
    ),
    (
        re.compile(r"^Show thinking\n", re.MULTILINE),
        "Gemini 'Show thinking' UI artifact",
    ),
    (re.compile(r"^Gemini said\n", re.MULTILINE), "Gemini 'Gemini said' UI artifact"),
    (re.compile(r"^You said\n", re.MULTILINE), "Gemini 'You said' UI artifact"),
    (re.compile(r"^ChatGPT said\n", re.MULTILINE), "ChatGPT UI artifact"),
]


class ValidationError(Exception):
    """Raised when a conversation fails schema validation."""

    pass


def check_content_quality(data):
    """Return a list of content-quality warning strings (extraction artifacts).

    At most one warning per turn (first matching pattern).
    """
    warnings = []
    for i, turn in enumerate(data.get("turns", [])):
        content = turn.get("content", "")
        for pattern, description in DIRTY_CONTENT_PATTERNS:
            if pattern.search(content):
                preview = content[:80].replace("\n", "\\n")
                warnings.append(
                    f"turns[{i}] (turn {turn.get('turn_number', '?')}): "
                    f'{description} — "{preview}..."'
                )
                break
    return warnings


def validate_conversation(data, strict=False):
    """Validate a conversation dict against schema v1.0.

    Raises ValidationError on failure. Unknown top-level keys (x-convaix, etc.)
    are ignored. See module docstring for strict-mode behavior.
    """
    # Top-level required fields
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            raise ValidationError(f"Missing required top-level field: {field}")

    if data["schema_version"] != "1.0":
        raise ValidationError(f"Unsupported schema_version: {data['schema_version']}")

    # Conversation block
    conv = data["conversation"]
    for field in REQUIRED_CONV_FIELDS:
        if field not in conv:
            raise ValidationError(f"Missing required conversation field: {field}")

    # Turns
    turns = data["turns"]
    if not isinstance(turns, list):
        raise ValidationError("turns must be a list")

    for i, turn in enumerate(turns):
        for field in REQUIRED_TURN_FIELDS:
            if field not in turn:
                raise ValidationError(f"Turn {i + 1}: missing required field: {field}")
        if turn["role"] not in VALID_ROLES:
            raise ValidationError(
                f"Turn {i + 1}: invalid role '{turn['role']}' "
                f"(must be one of {VALID_ROLES})"
            )

    if not strict:
        return

    # ── strict-only checks ──
    if not CONV_ID_RE.match(conv["id"]):
        raise ValidationError(
            f"Invalid conversation.id format: '{conv['id']}' (expected conv_ + 12 hex)"
        )
    if conv["source"] not in VALID_SOURCES:
        raise ValidationError(
            f"Invalid conversation.source: '{conv['source']}' "
            f"(must be one of {VALID_SOURCES})"
        )

    for i, turn in enumerate(turns):
        if turn["turn_number"] != i + 1:
            raise ValidationError(
                f"Turn {i + 1}: turn_number is {turn['turn_number']}, expected {i + 1}"
            )

    stats = data["statistics"]
    for field in REQUIRED_STATS_FIELDS:
        if field not in stats:
            raise ValidationError(f"Missing required statistics field: {field}")
    checks = {
        "turn_count": len(turns),
        "user_turns": sum(1 for t in turns if t.get("role") == "user"),
        "assistant_turns": sum(1 for t in turns if t.get("role") == "assistant"),
        "total_chars": sum(len(t.get("content", "")) for t in turns),
    }
    for key, actual in checks.items():
        if stats[key] != actual:
            raise ValidationError(
                f"statistics.{key} ({stats[key]}) != actual ({actual})"
            )

    quality = check_content_quality(data)
    if quality:
        raise ValidationError("Content-quality issues: " + "; ".join(quality))
