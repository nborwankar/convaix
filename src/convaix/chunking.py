"""Paragraph-level chunking for conversation turns."""


def split_into_chunks(content, min_chars=50):
    """Split content on double newlines, merge short paragraphs.

    Returns list of paragraph strings. Empty/None input returns [].
    """
    if not content or not content.strip():
        return []

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    merged = [paragraphs[0]]
    for p in paragraphs[1:]:
        if len(p) < min_chars and merged:
            merged[-1] = merged[-1] + "\n\n" + p
        else:
            merged.append(p)

    return merged
