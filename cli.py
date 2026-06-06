"""
turbovec CLI — manage vector databases from the command line.

Usage:
  python cli.py create-db <name>          # create db with chosen embedding provider
  python cli.py index <path>              # index a file or directory
  python cli.py search "query"            # semantic search
  python cli.py deploy <name>             # local copy
  python cli.py deploy --target archive   # export as .tar.gz
  python cli.py deploy --target s3://...  # upload to S3 / GCS / Azure
  python cli.py import-db <source>        # import from archive or cloud
  python cli.py serve-http                # start REST HTTP server
  python cli.py list-dbs                  # show all databases + their providers
  python cli.py providers                 # list available embedding providers
  python cli.py stats                     # show active db stats
  python cli.py remove <file>             # remove a file from the index
  python cli.py use <db-name>             # verify a database and show its stats
  python cli.py serve                     # start the MCP server
"""

import sys
import json
from pathlib import Path
from typing import Optional

import typer

sys.path.insert(0, str(Path(__file__).parent))
from indexer import VectorDB
from embeddings import make_provider, PROVIDERS
from deploy import export_archive, import_archive, upload_cloud, download_cloud, copy_local

BASE_DIR = Path.home() / ".turbovec"
app = typer.Typer(
    name="turbovec",
    help="TurboVec CLI — deploy and query vector databases from the command line.",
    add_completion=False,
)

PROVIDER_HELP = "Embedding provider: sentence-transformers (local) or openai (API)"
MODEL_HELP = "Model name for the provider (uses provider default if omitted)"


def _get_db(db: str, provider_name: Optional[str] = None, model: Optional[str] = None) -> VectorDB:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    provider = make_provider(provider_name, model) if provider_name else None
    return VectorDB(str(BASE_DIR / db), provider=provider)


# ------------------------------------------------------------------ #
# Commands                                                             #
# ------------------------------------------------------------------ #

@app.command()
def serve():
    """Start the TurboVec MCP server (used by Claude Code)."""
    import server as srv
    srv.mcp.run()


@app.command("create-db")
def create_db(
    name: str = typer.Argument(..., help="Name for the new database"),
    provider: str = typer.Option("sentence-transformers", "--provider", "-p", help=PROVIDER_HELP),
    model: Optional[str] = typer.Option(None, "--model", "-m", help=MODEL_HELP),
):
    """
    Create a new named vector database with a specific embedding provider.

    Examples:
      python cli.py create-db myproject
      python cli.py create-db myproject --provider openai --model text-embedding-3-large
      python cli.py create-db multilingual --model paraphrase-multilingual-MiniLM-L12-v2
    """
    db_path = BASE_DIR / name
    if db_path.with_suffix(".tvim").exists():
        typer.echo(f"✗ Database '{name}' already exists. Use: python cli.py use {name}", err=True)
        raise typer.Exit(1)

    try:
        emb_provider = make_provider(provider, model)
    except (ValueError, ImportError) as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    vdb = VectorDB(str(db_path), provider=emb_provider)
    vdb.save()
    s = vdb.stats()
    emb = s["embedding"]
    typer.echo(f"✓ Created '{name}'")
    typer.echo(f"  Provider : {emb['provider']}")
    typer.echo(f"  Model    : {emb['model']}")
    typer.echo(f"  Dim      : {emb['dim']}")


@app.command()
def providers():
    """List all available embedding providers and their models."""
    typer.echo("\nsentence-transformers  (local, offline)\n")
    models = [
        ("all-MiniLM-L6-v2",                        "384d",  "fast, general-purpose  [default]"),
        ("all-MiniLM-L12-v2",                        "384d",  "slightly better recall than L6"),
        ("all-mpnet-base-v2",                         "768d",  "higher quality, slower"),
        ("BAAI/bge-small-en-v1.5",                   "384d",  "fast, strong retrieval"),
        ("BAAI/bge-base-en-v1.5",                    "768d",  "strong retrieval, balanced"),
        ("BAAI/bge-large-en-v1.5",                   "1024d", "best retrieval, heavier"),
        ("intfloat/e5-small-v2",                     "384d",  "lightweight"),
        ("intfloat/e5-base-v2",                      "768d",  "balanced"),
        ("intfloat/e5-large-v2",                     "1024d", "best quality"),
        ("paraphrase-multilingual-MiniLM-L12-v2",    "384d",  "multilingual"),
    ]
    for m, d, note in models:
        typer.echo(f"  {m:<48} {d:<6}  {note}")

    typer.echo("\nopenai  (API — requires OPENAI_API_KEY)\n")
    oai_models = [
        ("text-embedding-3-small",  "1536d", "fast, cheap  [default]"),
        ("text-embedding-3-large",  "3072d", "highest quality"),
        ("text-embedding-ada-002",  "1536d", "legacy"),
    ]
    for m, d, note in oai_models:
        typer.echo(f"  {m:<48} {d:<6}  {note}")
    typer.echo()


@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory to index"),
    db: str = typer.Option("default", "--db", "-d", help="Target database name"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    extensions: Optional[str] = typer.Option(
        None, "--ext", help="Comma-separated extensions, e.g. .md,.py"
    ),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help=PROVIDER_HELP),
    model: Optional[str] = typer.Option(None, "--model", "-m", help=MODEL_HELP),
):
    """
    Index files into a vector database.

    If the database doesn't exist yet, it will be created with the specified
    provider (defaults to sentence-transformers/all-MiniLM-L6-v2).
    If it already exists, the stored provider is used automatically.
    """
    vdb = _get_db(db, provider, model)
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
    name: Optional[str] = typer.Argument(None, help="Destination name (required for local target)"),
    source: str = typer.Option("default", "--source", "-s", help="Source database name"),
    target: str = typer.Option("local", "--target", "-t",
        help="Deployment target: 'local', 'archive', 's3://bucket/prefix', 'gs://...', 'azure://...'"),
    output: Optional[str] = typer.Option(None, "--output", "-o",
        help="Output path for local copy dir or archive file"),
):
    """
    Deploy a database to a local copy, archive, or cloud storage.

    Examples:
      python cli.py deploy backup --source mydb
      python cli.py deploy --source mydb --target archive --output /tmp/mydb.tar.gz
      python cli.py deploy --source mydb --target s3://my-bucket/indexes
      python cli.py deploy --source mydb --target gs://my-bucket/indexes
      python cli.py deploy --source mydb --target azure://my-container/indexes
    """
    try:
        if target == "local":
            if not name:
                typer.echo("✗ Provide a destination name: python cli.py deploy <name>", err=True)
                raise typer.Exit(1)
            files = copy_local(source, name, output)
            typer.echo(f"✓ Copied '{source}' → '{name}'")
            for f in files:
                typer.echo(f"  {f}")

        elif target == "archive":
            out = export_archive(source, output)
            typer.echo(f"✓ Archive created: {out}")

        else:
            # cloud
            uris = upload_cloud(source, target)
            typer.echo(f"✓ Uploaded '{source}' to {target}")
            for u in uris:
                typer.echo(f"  {u}")

    except (FileNotFoundError, ValueError, ImportError) as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1)


@app.command("import-db")
def import_db(
    source: str = typer.Argument(...,
        help="Archive path or cloud URI (s3://, gs://, azure://)"),
    db_name: Optional[str] = typer.Option(None, "--name", "-n",
        help="Override the database name (default: inferred from archive/URI)"),
):
    """
    Import a database from a local archive or cloud storage.

    Examples:
      python cli.py import-db /tmp/mydb.tar.gz
      python cli.py import-db /tmp/mydb.tar.gz --name newname
      python cli.py import-db s3://my-bucket/indexes/mydb.tar.gz
      python cli.py import-db gs://my-bucket/indexes/mydb
      python cli.py import-db azure://my-container/indexes/mydb
    """
    try:
        is_cloud = any(source.startswith(p) for p in ("s3://", "gs://", "azure://"))

        if is_cloud and source.endswith(".tar.gz"):
            import tempfile
            from deploy import _parse_uri, _download_single_cloud_file
            provider, bucket, key = _parse_uri(source)
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name
            _download_single_cloud_file(provider, bucket, key, tmp_path)
            name = import_archive(tmp_path, db_name)
            Path(tmp_path).unlink(missing_ok=True)
        elif is_cloud:
            name = download_cloud(source, db_name)
        else:
            name = import_archive(source, db_name)

        vdb = VectorDB(str(BASE_DIR / name))
        s = vdb.stats()
        emb = s["embedding"]
        typer.echo(f"✓ Imported '{name}'")
        typer.echo(f"  Files: {s['files_indexed']}  Chunks: {s['total_chunks']}")
        typer.echo(f"  Embedding: {emb['provider']} / {emb['model']}  dim={emb['dim']}")

    except (FileNotFoundError, ValueError, ImportError) as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1)


@app.command("serve-http")
def serve_http(
    db: str = typer.Option("default", "--db", "-d", help="Database to serve"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
    write_key: Optional[str] = typer.Option(None, "--write-key",
        help="Bearer token required for write endpoints (index, delete). "
             "If not set, write endpoints are open."),
):
    """
    Start the TurboVec HTTP REST server.

    Exposes the vector index over HTTP so any process or machine can query it.
    Interactive API docs available at http://localhost:<port>/docs once running.

    Examples:
      python cli.py serve-http --db myproject
      python cli.py serve-http --db myproject --port 9000 --write-key mysecret
    """
    try:
        import http_server as hs
    except ImportError:
        typer.echo("✗ HTTP server requires: pip install fastapi uvicorn[standard]", err=True)
        raise typer.Exit(1)

    hs.serve(db_name=db, host=host, port=port, write_key=write_key)


@app.command("list-dbs")
def list_dbs():
    """List all available vector databases with their embedding providers."""
    if not BASE_DIR.exists():
        typer.echo("No databases found.")
        return
    dbs = sorted({f.stem for f in BASE_DIR.glob("*.tvim")})
    if not dbs:
        typer.echo("No databases found. Run: python cli.py create-db <name>")
        return
    for name in dbs:
        meta_file = (BASE_DIR / name).with_suffix(".json")
        emb_info = ""
        if meta_file.exists():
            try:
                raw = json.loads(meta_file.read_text(encoding="utf-8"))
                emb = raw.get("embedding", {})
                if emb:
                    emb_info = f"  [{emb.get('provider','?')} / {emb.get('model','?')}  dim={emb.get('dim','?')}]"
            except Exception:
                pass
        typer.echo(f"  • {name}{emb_info}")


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
