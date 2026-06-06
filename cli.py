"""
turbovec CLI — manage vector databases from the command line.

Usage:
  python cli.py serve                     # start the MCP server
  python cli.py index <path>              # index a file or directory
  python cli.py search "query"            # semantic search
  python cli.py deploy <name>             # export/deploy a database
  python cli.py list-dbs                  # show all databases
  python cli.py stats                     # show active db stats
  python cli.py remove <file>             # remove a file from the index
  python cli.py use <db-name>             # print the switch command (use tv_use_db in Claude Code)
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Optional

import typer

sys.path.insert(0, str(Path(__file__).parent))
from indexer import VectorDB

BASE_DIR = Path.home() / ".turbovec"
app = typer.Typer(
    name="turbovec",
    help="TurboVec CLI — deploy and query vector databases from the command line.",
    add_completion=False,
)


def _get_db(db: str) -> VectorDB:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return VectorDB(str(BASE_DIR / db))


# ------------------------------------------------------------------ #
# Commands                                                             #
# ------------------------------------------------------------------ #

@app.command()
def serve():
    """Start the TurboVec MCP server (used by Claude Code)."""
    import server as srv
    srv.mcp.run()


@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory to index"),
    db: str = typer.Option("default", "--db", "-d", help="Target database name"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    extensions: Optional[str] = typer.Option(
        None, "--ext", help="Comma-separated extensions, e.g. .md,.py"
    ),
):
    """Index files into a vector database."""
    vdb = _get_db(db)
    p = Path(path).expanduser()
    exts = set(extensions.split(",")) if extensions else None

    if p.is_file():
        count = vdb.index_file(str(p))
        typer.echo(f"✓ {p.name}: {count} chunks")
        return

    if p.is_dir():
        results = vdb.index_directory(str(p), recursive=recursive, extensions=exts)
        ok = {k: v for k, v in results.items() if isinstance(v, int)}
        errors = {k: v for k, v in results.items() if isinstance(v, str)}
        total = sum(ok.values())
        typer.echo(f"\n✓ {len(ok)} files · {total} chunks total")
        for f, n in ok.items():
            typer.echo(f"  {Path(f).name}: {n} chunks")
        if errors:
            typer.echo(f"\n✗ {len(errors)} error(s):", err=True)
            for f, e in errors.items():
                typer.echo(f"  {Path(f).name}: {e}", err=True)
        return

    typer.echo(f"✗ Path not found: {path}", err=True)
    raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(...),
    db: str = typer.Option("default", "--db", "-d"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    filter_files: Optional[str] = typer.Option(
        None, "--files", help="Comma-separated file paths to restrict search"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Semantic search over an indexed database."""
    vdb = _get_db(db)
    files = filter_files.split(",") if filter_files else None
    results = vdb.search(query, top_k=top_k, filter_files=files)

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not results:
        typer.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        typer.echo(f"\n[{i}] {r['file']}  line {r['start_line']+1}–{r['end_line']+1}  score={r['score']:.3f}")
        typer.echo("─" * 60)
        typer.echo(r["text"].strip()[:400])


@app.command()
def deploy(
    name: str = typer.Argument(..., help="Name for the exported database"),
    source: str = typer.Option("default", "--source", "-s", help="Source database"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Export a vector database to a named copy or path."""
    src_base = BASE_DIR / source
    dst_dir = Path(output) if output else BASE_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for ext in (".tvim", ".json"):
        src = src_base.with_suffix(ext)
        if src.exists():
            dst = dst_dir / (name + ext)
            shutil.copy2(src, dst)
            copied.append(str(dst))

    if not copied:
        typer.echo(f"✗ Source database '{source}' not found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"✓ Deployed '{source}' → '{name}'")
    for f in copied:
        typer.echo(f"  {f}")


@app.command("list-dbs")
def list_dbs():
    """List all available vector databases."""
    if not BASE_DIR.exists():
        typer.echo("No databases found.")
        return
    dbs = sorted({f.stem for f in BASE_DIR.glob("*.tvim")})
    if not dbs:
        typer.echo("No databases found. Run: turbovec index <path>")
        return
    for name in dbs:
        typer.echo(f"  • {name}")


@app.command()
def stats(db: str = typer.Option("default", "--db", "-d")):
    """Show statistics for a vector database."""
    vdb = _get_db(db)
    s = vdb.stats()
    typer.echo(json.dumps(s, indent=2))


@app.command()
def remove(
    file_path: str = typer.Argument(..., help="File to remove from the index"),
    db: str = typer.Option("default", "--db", "-d"),
):
    """Remove a file's chunks from the index."""
    vdb = _get_db(db)
    vdb.remove_file(file_path)
    vdb.save()
    typer.echo(f"✓ Removed '{file_path}' from '{db}'")


@app.command()
def use(
    db_name: str = typer.Argument(..., help="Database name to switch to"),
):
    """
    Switch the active database for future CLI commands.
    Verifies the database exists and prints its stats.
    (Inside Claude Code, use the tv_use_db MCP tool instead.)
    """
    db_path = BASE_DIR / db_name
    if not db_path.with_suffix(".tvim").exists():
        typer.echo(f"✗ Database '{db_name}' not found in {BASE_DIR}", err=True)
        typer.echo("Available databases:")
        dbs = sorted({f.stem for f in BASE_DIR.glob("*.tvim")})
        for d in dbs:
            typer.echo(f"  • {d}")
        raise typer.Exit(1)

    vdb = _get_db(db_name)
    s = vdb.stats()
    typer.echo(f"✓ Database '{db_name}' is ready")
    typer.echo(f"  Files: {s['files_indexed']}  Chunks: {s['total_chunks']}")
    typer.echo(f"  Index: {s['index_file']}")


if __name__ == "__main__":
    app()
