import uuid
from convaix.schema import (
    slugify,
    generate_conv_id,
    generate_convaix_id,
    convert_to_schema,
    add_convaix_extension,
    ROLE_MAP,
)


def test_slugify_basic():
    assert slugify("Hello World!") == "hello-world"


def test_slugify_truncates():
    assert len(slugify("a" * 100, max_len=60)) == 60


def test_generate_conv_id_stable():
    id1 = generate_conv_id("chatgpt", "abc123")
    id2 = generate_conv_id("chatgpt", "abc123")
    assert id1 == id2
    assert id1.startswith("conv_")


def test_generate_conv_id_different_sources():
    id1 = generate_conv_id("chatgpt", "abc123")
    id2 = generate_conv_id("claude", "abc123")
    assert id1 != id2


def test_generate_convaix_id():
    cid = generate_convaix_id()
    assert cid.startswith("cx_")
    # Should be valid UUID after prefix
    uuid.UUID(cid[3:])


def test_role_map():
    assert ROLE_MAP["model"] == "assistant"
    assert ROLE_MAP["human"] == "user"


def test_convert_to_schema():
    turns = [
        {"role": "human", "content": "Hello"},
        {"role": "model", "content": "Hi there"},
    ]
    result = convert_to_schema(
        source="gemini",
        source_id="test123",
        title="Test Conversation",
        turns=turns,
    )
    assert result["schema_version"] == "1.0"
    assert result["conversation"]["source"] == "gemini"
    assert result["turns"][0]["role"] == "user"
    assert result["turns"][1]["role"] == "assistant"
    assert result["statistics"]["turn_count"] == 2
    assert "x-convaix" not in result


def test_add_convaix_extension():
    conv = convert_to_schema(
        source="claude",
        source_id="test456",
        title="Test",
        turns=[{"role": "user", "content": "Hi"}],
    )
    extended = add_convaix_extension(conv, author_handle="nborwankar")
    assert "x-convaix" in extended
    ext = extended["x-convaix"]
    assert ext["convaix_id"].startswith("cx_")
    assert ext["author"]["handle"] == "nborwankar"
    assert ext["conv_id"] == conv["conversation"]["id"]
    assert ext["parent_refs"] == []
    assert ext["signature"] is None


from convaix.schema import clean_turn_text


def test_clean_user_you_said():
    assert clean_turn_text("You said\nHello there", "user") == "Hello there"


def test_clean_assistant_combined_prefix():
    assert (
        clean_turn_text("Show thinking\nGemini said\nThe answer", "assistant")
        == "The answer"
    )


def test_clean_assistant_gemini_said():
    assert clean_turn_text("Gemini said\nHi", "assistant") == "Hi"


def test_clean_assistant_chatgpt_said():
    assert clean_turn_text("ChatGPT said\nResponse", "assistant") == "Response"


def test_clean_user_attachment_prefixed():
    assert (
        clean_turn_text("photo.png\nPNG\nYou said\nLook at this", "user")
        == "Look at this"
    )


def test_clean_noop_when_clean():
    assert clean_turn_text("Just normal content", "assistant") == "Just normal content"
