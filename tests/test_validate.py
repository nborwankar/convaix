import pytest
from convaix.validate import validate_conversation, ValidationError


def _minimal_valid():
    return {
        "schema_version": "1.0",
        "conversation": {
            "id": "conv_abc123",
            "title": "Test",
            "source": "chatgpt",
            "source_id": "xyz",
            "exported_at": "2026-02-18T00:00:00Z",
        },
        "turns": [
            {"turn_number": 1, "role": "user", "content": "Hello"},
        ],
        "statistics": {
            "turn_count": 1,
            "user_turns": 1,
            "assistant_turns": 0,
            "total_chars": 5,
        },
    }


def test_valid_minimal():
    validate_conversation(_minimal_valid())  # should not raise


def test_missing_schema_version():
    data = _minimal_valid()
    del data["schema_version"]
    with pytest.raises(ValidationError, match="schema_version"):
        validate_conversation(data)


def test_missing_conversation_id():
    data = _minimal_valid()
    del data["conversation"]["id"]
    with pytest.raises(ValidationError, match="id"):
        validate_conversation(data)


def test_invalid_role():
    data = _minimal_valid()
    data["turns"][0]["role"] = "alien"
    with pytest.raises(ValidationError, match="role"):
        validate_conversation(data)


def test_valid_with_x_convaix():
    data = _minimal_valid()
    data["x-convaix"] = {
        "convaix_id": "cx_test",
        "version": "0.1",
        "conv_id": "conv_abc123",
        "author": {"handle": "test"},
        "published_at": "2026-02-18T00:00:00Z",
        "parent_refs": [],
        "annotations": [],
        "signature": None,
    }
    validate_conversation(data)  # should not raise


def test_unknown_top_level_keys_ignored():
    data = _minimal_valid()
    data["x-future-extension"] = {"foo": "bar"}
    validate_conversation(data)  # should not raise
