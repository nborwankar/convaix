from convaix.chunking import split_into_chunks


def test_single_paragraph():
    assert split_into_chunks("Hello world") == ["Hello world"]


def test_double_newline_split():
    text = (
        "This is the first paragraph with enough content to exceed the threshold easily.\n\n"
        "This is the second paragraph which also has enough content to stand alone."
    )
    result = split_into_chunks(text)
    assert len(result) == 2
    assert "first paragraph" in result[0]
    assert "second paragraph" in result[1]


def test_short_paragraphs_merged():
    text = (
        "This is a long enough first paragraph that exceeds the minimum character threshold.\n\n"
        "OK\n\n"
        "This is a long enough third paragraph that also exceeds the minimum character threshold."
    )
    result = split_into_chunks(text, min_chars=50)
    # "OK" (2 chars) should merge into previous
    assert len(result) == 2
    assert "OK" in result[0]  # merged into first


def test_empty_input():
    assert split_into_chunks("") == []
    assert split_into_chunks("   ") == []
    assert split_into_chunks(None) == []


def test_whitespace_only_paragraphs():
    text = (
        "Real content that is long enough to exceed the default minimum threshold.\n\n"
        "   \n\n"
        "More content that is also long enough to exceed the default minimum threshold."
    )
    result = split_into_chunks(text)
    assert len(result) == 2
