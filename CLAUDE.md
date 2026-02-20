# CLAUDE.md — convaix

**Project**: convaix — AI conversation exchange
**Status**: Initial development (v0.1.0)
**Created**: 2026-02-19

## Overview

convaix is a standalone tool for storing, searching, and sharing AI conversations.
It provides a standard schema, local search (SQLite + sqlite-vec), and a P2P exchange
layer starting with git-based shared repos.

## Development Commands

```bash
# Setup
conda activate convaix
pip install -e ".[all]"

# Testing
pytest
pytest -v                    # verbose
pytest -m "not slow"         # skip embedding tests

# Code formatting
black src/ tests/
flake8 src/ tests/
```

## Architecture

```
convaix/
├── src/convaix/
│   ├── schema.py          # v1.0 format + x-convaix extensions
│   ├── validate.py        # JSON schema validation
│   ├── db.py              # SQLite3 + sqlite-vec
│   ├── chunking.py        # Paragraph splitting
│   ├── embeddings.py      # MLX + nomic-embed-text-v1.5
│   ├── search.py          # Hybrid semantic + keyword
│   ├── cli.py             # Click CLI entry point
│   ├── providers/         # ChatGPT, Claude, Gemini adapters
│   └── exchange/          # Git-based P2P sharing
└── tests/
```

## Key Concepts

- **Immutable snapshots**: Each export is a frozen snapshot with unique `convaix_id`
- **Lineage**: `conv_id` groups snapshots of the same LLM conversation
- **x-convaix extension**: P2P metadata in `"x-convaix"` namespace, backward compatible
- **Hybrid search**: Semantic (sqlite-vec cosine) + keyword (LIKE) combined
