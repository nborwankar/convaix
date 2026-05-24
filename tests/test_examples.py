import json
from pathlib import Path

import pytest

from convaix.validate import validate_conversation, check_content_quality

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.json"))


def test_examples_exist():
    assert len(EXAMPLE_FILES) == 3


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_validates(path):
    data = json.loads(path.read_text())
    validate_conversation(data)  # default (structural) — must not raise


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_is_clean(path):
    data = json.loads(path.read_text())
    assert check_content_quality(data) == []
