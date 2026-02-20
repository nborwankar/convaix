import json
from click.testing import CliRunner
from convaix.cli import main


def _write_sample_conv(path, convaix_id="cx_cli_test"):
    data = {
        "schema_version": "1.0",
        "conversation": {
            "id": "conv_cli",
            "title": "CLI Test Conv",
            "source": "chatgpt",
            "source_id": "cli_test",
            "exported_at": "2026-02-18T00:00:00Z",
        },
        "turns": [
            {"turn_number": 1, "role": "user", "content": "Hello from CLI test"},
            {
                "turn_number": 2,
                "role": "assistant",
                "content": "Hi! This is the CLI test response.",
            },
        ],
        "statistics": {
            "turn_count": 2,
            "user_turns": 1,
            "assistant_turns": 1,
            "total_chars": 42,
        },
        "x-convaix": {
            "convaix_id": convaix_id,
            "version": "0.1",
            "conv_id": "conv_cli",
            "author": {"handle": "testuser"},
            "published_at": "2026-02-18T00:00:00Z",
            "parent_refs": [],
            "annotations": [],
            "signature": None,
        },
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def test_load_command(tmp_path):
    conv_dir = tmp_path / "convs"
    conv_dir.mkdir()
    _write_sample_conv(str(conv_dir / "test.json"))

    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    result = runner.invoke(
        main, ["load", str(conv_dir), "--db", db_path, "--skip-embeddings"]
    )
    assert result.exit_code == 0
    assert "Loaded" in result.output or "1" in result.output


def test_list_command(tmp_path):
    conv_dir = tmp_path / "convs"
    conv_dir.mkdir()
    _write_sample_conv(str(conv_dir / "test.json"))

    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    runner.invoke(main, ["load", str(conv_dir), "--db", db_path, "--skip-embeddings"])
    result = runner.invoke(main, ["list", "--db", db_path])
    assert result.exit_code == 0
    assert "CLI Test Conv" in result.output


def test_search_keyword(tmp_path):
    conv_dir = tmp_path / "convs"
    conv_dir.mkdir()
    _write_sample_conv(str(conv_dir / "test.json"))

    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    runner.invoke(main, ["load", str(conv_dir), "--db", db_path, "--skip-embeddings"])
    result = runner.invoke(main, ["search", "CLI test", "--db", db_path])
    assert result.exit_code == 0


def test_validate_command(tmp_path):
    path = str(tmp_path / "valid.json")
    _write_sample_conv(path)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", path])
    assert result.exit_code == 0
    assert "Valid" in result.output or "valid" in result.output
