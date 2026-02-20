"""convaix CLI — AI conversation exchange."""

import json
import logging
import os

import click
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)

DEFAULT_DB = os.path.expanduser("~/.convaix/convaix.db")


@click.group()
@click.version_option(package_name="convaix")
def main():
    """convaix — store, search, and share AI conversations."""
    pass


@main.command()
@click.argument("path")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option(
    "--skip-embeddings", is_flag=True, help="Load without generating embeddings"
)
def load(path, db, skip_embeddings):
    """Load conversation JSON files into local database."""
    from .db import init_db, load_snapshot, chunk_snapshot
    from .schema import add_convaix_extension
    from .validate import ValidationError, validate_conversation

    conn = init_db(db)
    loaded = 0
    skipped = 0
    errors = 0

    files = []
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.endswith(".json"):
                files.append(os.path.join(path, f))
    elif os.path.isfile(path):
        files.append(path)
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        return

    for filepath in files:
        basename = os.path.basename(filepath)
        try:
            with open(filepath) as f:
                data = json.load(f)

            validate_conversation(data)

            # Add x-convaix extension if not present
            if "x-convaix" not in data:
                add_convaix_extension(data, author_handle="local")

            if load_snapshot(conn, data):
                chunk_snapshot(conn, data, skip_embeddings=skip_embeddings)
                console.print(f"  [green]Loaded[/green]: {basename}")
                loaded += 1
            else:
                skipped += 1
        except (ValidationError, json.JSONDecodeError) as e:
            console.print(f"  [red]Error[/red]: {basename}: {e}")
            errors += 1

    conn.close()

    table = Table(title="Load Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    table.add_row("Loaded", str(loaded))
    table.add_row("Skipped", str(skipped))
    table.add_row("Errors", str(errors))
    console.print(table)


@main.command("list")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--source", "-s", help="Filter by source")
def list_cmd(db, source):
    """List loaded conversation snapshots."""
    from .db import init_db, list_snapshots

    conn = init_db(db)
    rows = list_snapshots(conn, source=source)
    conn.close()

    if not rows:
        console.print("[yellow]No snapshots found.[/yellow]")
        return

    table = Table(title="Snapshots")
    table.add_column("#", style="dim")
    table.add_column("Title", style="blue", ratio=2)
    table.add_column("Source", style="magenta")
    table.add_column("Author", style="cyan")
    table.add_column("Turns", style="green")
    table.add_column("convaix_id", style="dim")

    for i, row in enumerate(rows, 1):
        table.add_row(
            str(i),
            row["title"],
            row["source"],
            row["author"] or "",
            str(row["turn_count"]),
            row["convaix_id"][:16] + "...",
        )

    console.print(table)


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--limit", "-l", type=int, default=10, help="Max results")
@click.option("--source", "-s", help="Filter by source")
@click.option(
    "--conversations",
    "-c",
    "conv_mode",
    is_flag=True,
    help="Conversation-level results",
)
def search(query, db, limit, source, conv_mode):
    """Hybrid search across loaded conversations."""
    from .db import init_db
    from .search import search_chunks, search_conversations

    query_str = " ".join(query)
    conn = init_db(db)

    if conv_mode:
        results = search_conversations(conn, query_str, source=source, limit=limit)
        conn.close()
        if not results:
            console.print("[yellow]No conversations found.[/yellow]")
            return

        table = Table(title=f'Conversations: "{query_str}"', show_lines=True)
        table.add_column("#", style="dim")
        table.add_column("Title", style="blue", ratio=2)
        table.add_column("Src", style="magenta")
        table.add_column("Hits", style="green")
        table.add_column("Turns", style="cyan")

        for i, r in enumerate(results, 1):
            table.add_row(
                str(i), r["title"], r["source"], str(r["hits"]), str(r["turn_count"])
            )
        console.print(table)
        return

    results = search_chunks(conn, query_str, source=source, limit=limit)
    conn.close()

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f'Search: "{query_str}"', show_lines=True)
    table.add_column("Sim", style="green", no_wrap=True)
    table.add_column("Match", style="yellow", no_wrap=True)
    table.add_column("Src", style="magenta")
    table.add_column("Role", style="cyan")
    table.add_column("Conversation", style="blue", max_width=30)
    table.add_column("Content", ratio=2)

    for r in results:
        preview = r["chunk_text"][:300]
        if len(r["chunk_text"]) > 300:
            preview += "..."
        table.add_row(
            f"{r['similarity']:.3f}",
            r["match_type"],
            r["source"],
            r["role"],
            r["title"][:30],
            preview,
        )
    console.print(table)


@main.command()
@click.argument("file_path")
def validate(file_path):
    """Validate a conversation JSON file against schema v1.0."""
    from .validate import ValidationError, validate_conversation

    try:
        with open(file_path) as f:
            data = json.load(f)
        validate_conversation(data)
        console.print(f"[green]Valid[/green]: {file_path}")
    except ValidationError as e:
        console.print(f"[red]Invalid[/red]: {e}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]JSON error[/red]: {e}")
        raise SystemExit(1)


@main.command()
@click.argument("conv_id")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def history(conv_id, db):
    """Show all snapshots of a conversation lineage."""
    from .db import get_snapshot_history, init_db

    conn = init_db(db)
    rows = get_snapshot_history(conn, conv_id)
    conn.close()

    if not rows:
        console.print(f"[yellow]No snapshots found for {conv_id}[/yellow]")
        return

    table = Table(title=f"History: {conv_id}")
    table.add_column("convaix_id", style="dim")
    table.add_column("Published", style="cyan")
    table.add_column("Turns", style="green")
    table.add_column("Author", style="blue")

    for row in rows:
        table.add_row(
            row["convaix_id"][:16] + "...",
            row["published_at"] or "",
            str(row["turn_count"]),
            row["author"] or "",
        )
    console.print(table)


@main.command()
@click.argument("convaix_id")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--output", "-o", help="Output file path (default: stdout)")
def export(convaix_id, db, output):
    """Export a snapshot back to JSON."""
    from .db import get_snapshot, init_db

    conn = init_db(db)
    row = get_snapshot(conn, convaix_id)
    conn.close()

    if not row:
        console.print(f"[red]Snapshot not found: {convaix_id}[/red]")
        raise SystemExit(1)

    raw = json.loads(row["raw"])
    formatted = json.dumps(raw, indent=2, ensure_ascii=False)

    if output:
        with open(output, "w") as f:
            f.write(formatted)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        click.echo(formatted)
