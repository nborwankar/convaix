"""Schema validation for conversation JSON files."""

VALID_ROLES = {"user", "assistant", "system"}
REQUIRED_CONV_FIELDS = {"id", "title", "source", "exported_at"}
REQUIRED_TURN_FIELDS = {"turn_number", "role", "content"}
REQUIRED_TOP_LEVEL = {"schema_version", "conversation", "turns", "statistics"}


class ValidationError(Exception):
    """Raised when a conversation fails schema validation."""

    pass


def validate_conversation(data):
    """Validate a conversation dict against schema v1.0.

    Raises ValidationError with a descriptive message on failure.
    Unknown top-level keys (like x-convaix, x-future) are ignored.
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
