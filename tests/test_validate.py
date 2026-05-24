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


from convaix.validate import check_content_quality


def _strict_valid():
    return {
        "schema_version": "1.0",
        "conversation": {
            "id": "conv_0123456789ab",  # 12 hex
            "title": "Test",
            "source": "chatgpt",
            "exported_at": "2026-02-18T00:00:00Z",
        },
        "turns": [
            {"turn_number": 1, "role": "user", "content": "Hello"},
            {"turn_number": 2, "role": "assistant", "content": "Hi!"},
        ],
        "statistics": {
            "turn_count": 2,
            "user_turns": 1,
            "assistant_turns": 1,
            "total_chars": 8,
        },
    }


def test_strict_accepts_clean():
    validate_conversation(_strict_valid(), strict=True)  # no raise


def test_strict_rejects_bad_id_format():
    data = _strict_valid()
    data["conversation"]["id"] = "conv_abc123"  # too short
    with pytest.raises(ValidationError, match="id format"):
        validate_conversation(data, strict=True)


def test_strict_rejects_bad_source():
    data = _strict_valid()
    data["conversation"]["source"] = "bard"
    with pytest.raises(ValidationError, match="source"):
        validate_conversation(data, strict=True)


def test_strict_rejects_stats_mismatch():
    data = _strict_valid()
    data["statistics"]["total_chars"] = 999
    with pytest.raises(ValidationError, match="total_chars"):
        validate_conversation(data, strict=True)


def test_strict_rejects_turn_number_gap():
    data = _strict_valid()
    data["turns"][1]["turn_number"] = 5
    with pytest.raises(ValidationError, match="turn_number"):
        validate_conversation(data, strict=True)


def test_strict_rejects_dirty_content():
    data = _strict_valid()
    data["turns"][1]["content"] = "Gemini said\nHi!"
    data["statistics"]["total_chars"] = len("Hello") + len("Gemini said\nHi!")
    with pytest.raises(ValidationError, match="[Cc]ontent"):
        validate_conversation(data, strict=True)


def test_check_content_quality_flags_artifacts():
    data = _strict_valid()
    data["turns"][0]["content"] = "You said\nHello"
    warnings = check_content_quality(data)
    assert len(warnings) == 1
    assert "You said" in warnings[0]


def test_check_content_quality_clean_is_empty():
    assert check_content_quality(_strict_valid()) == []


def test_default_validator_still_lenient():
    # 6-hex id, no strict — must NOT raise (backward compat)
    validate_conversation(_minimal_valid())
