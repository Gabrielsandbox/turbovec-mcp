"""
TurboVec MCP Server
Exposes vector-search tools to Claude Code via the Model Context Protocol.
"""

import sys
import json
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent))
from indexer import VectorDB
from embeddings import make_provider, PROVIDERS
from deploy import export_archive, import_archive, upload_cloud, download_cloud, copy_local

# ------------------------------------------------------------------ #
# Database management                                                  #
# ------------------------------------------------------------------ #

BASE_DIR = Path.home() / ".turbovec"
BASE_DIR.mkdir(parents=True, exist_ok=True)

_active_db: Optional[VectorDB] = None
_active_name: str = "default"


def get_db() -> VectorDB:
    global _active_db, _active_name
    if _active_db is None:
        _active_db = VectorDB(str(BASE_DIR / _active_name))
    return _active_db


# ------------------------------------------------------------------ #
# MCP tools                                                            #
# ------------------------------------------------------------------ #

mcp = FastMCP("turbovec")


@mcp.tool()
def tv_index(path: str, recursive: bool = True) -> str:
    """
    Index a file or directory into the active vector database.
    Supports .md .txt .py .js .ts .json .yaml .csv .sql and 20+ more.
    Returns a summary of files and chunks added.
    """
    db = get_db()
    p = Path(path).expanduser()

    if p.is_file():
        count = db.index_file(str(p))
        return f"Indexed {p.name}: {count} chunks added."

    if p.is_dir():
        results = db.index_directory(str(p), recursive=recursive)
        ok = {k: v for k, v in results.items() if isinstance(v, int)}
        errors = {k: v for k, v in results.items() if isinstance(v, str)}
        total_chunks = sum(ok.values())
        lines = [f"Indexed {len(ok)} files · {total_chunks} chunks total."]
        for f, n in list(ok.items())[:30]:
            lines.append(f"  ✓ {Path(f).name}: {n} chunks")
        if errors:
            lines.append(f"\n{len(errors)} error(s):")
            for f, e in list(errors.items())[:10]:
                lines.append(f"  ✗ {Path(f).name}: {e}")
        return "\n".join(lines)

    return f"Path not found: {path}"


@mcp.tool()
def tv_search(query: str, top_k: int = 5, filter_files: Optional[list[str]] = None) -> str:
    """
    Semantic search over indexed files.
    Returns the most relevant chunks with file paths, line numbers, and similarity scores.
    Use filter_files to restrict search to specific file paths.
    """
    db = get_db()
    results = db.search(query, top_k=top_k, filter_files=filter_files)

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        file_ref = f"{r['file']}:{r['start_line'] + 1}"
        lines.append(f"### {i}. [{Path(r['file']).name}:{r['start_line']+1}]({file_ref})  score={r['score']:.3f}")
        lines.append("```")
        lines.append(r["text"].strip()[:500])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def tv_remove_file(file_path: str) -> str:
    """Remove all indexed chunks for a specific file."""
    db = get_db()
    db.remove_file(file_path)
    db.save()
    return f"Removed '{file_path}' from index."


@mcp.tool()
def tv_stats() -> str:
    """Return statistics about the active vector database, including the embedding provider and model."""
    db = get_db()
    return json.dumps(db.stats(), indent=2)


@mcp.tool()
def tv_use_db(name: str) -> str:
    """
    Switch the active vector database by name.
    All subsequent operations will use this database.
    The embedding provider is restored automatically from the stored index config.
    Available databases are stored in ~/.turbovec/<name>.
    """
    global _active_db, _active_name
    _active_name = name
    _active_db = VectorDB(str(BASE_DIR / name))
    s = _active_db.stats()
    emb = s["embedding"]
    return (
        f"Switched to database '{name}'. "
        f"{s['files_indexed']} files · {s['total_chunks']} chunks. "
        f"Embedding: {emb['provider']} / {emb['model']} (dim={emb['dim']})"
    )


@mcp.tool()
def tv_create_db(name: str, provider: str = "sentence-transformers", model: Optional[str] = None) -> str:
    """
    Create a new named vector database with a specific embedding provider.
    Fails if a database with that name already exists (use tv_use_db to switch to it).

    provider options:
      - "sentence-transformers"  (local, offline)
          models: all-MiniLM-L6-v2 (default, 384d), all-mpnet-base-v2 (768d),
                  BAAI/bge-base-en-v1.5 (768d), BAAI/bge-large-en-v1.5 (1024d),
                  intfloat/e5-base-v2 (768d), paraphrase-multilingual-MiniLM-L12-v2 (384d)
      - "openai"                 (API, requires OPENAI_API_KEY)
          models: text-embedding-3-small (1536d, default), text-embedding-3-large (3072d)
    """
    global _active_db, _active_name
    db_path = BASE_DIR / name

    if db_path.with_suffix(".tvim").exists():
        return (
            f"Database '{name}' already exists. "
            f"Use tv_use_db('{name}') to switch to it, or choose a different name."
        )

    try:
        emb_provider = make_provider(provider, model)
    except (ValueError, ImportError) as e:
        return f"Error: {e}"

    _active_name = name
    _active_db = VectorDB(str(db_path), provider=emb_provider)
    _active_db.save()

    s = _active_db.stats()
    emb = s["embedding"]
    return (
        f"Created database '{name}' with {emb['provider']} / {emb['model']} (dim={emb['dim']}). "
        f"Now active. Use tv_index to add files."
    )


@mcp.tool()
def tv_list_dbs() -> str:
    """List all available vector databases in ~/.turbovec/ with their embedding providers."""
    dbs = sorted({f.stem for f in BASE_DIR.glob("*.tvim")})
    if not dbs:
        return "No databases found. Run tv_create_db or tv_index to create one."

    lines = []
    for n in dbs:
        marker = " ◀ active" if n == _active_name else ""
        meta_file = (BASE_DIR / n).with_suffix(".json")
        emb_info = ""
        if meta_file.exists():
            try:
                raw = json.loads(meta_file.read_text(encoding="utf-8"))
                emb = raw.get("embedding", {})
                if emb:
                    emb_info = f"  [{emb.get('provider','?')} / {emb.get('model','?')}]"
            except Exception:
                pass
        lines.append(f"  • {n}{emb_info}{marker}")
    return "\n".join(lines)


@mcp.tool()
def tv_list_providers() -> str:
    """List all available embedding providers and their supported models."""
    lines = [
        "**sentence-transformers** (local, offline — no API key needed)",
        "  Models:",
        "    all-MiniLM-L6-v2              384d  fast, good general-purpose (default)",
        "    all-MiniLM-L12-v2             384d  slightly better recall than L6",
        "    all-mpnet-base-v2             768d  higher quality, slower",
        "    BAAI/bge-small-en-v1.5        384d  fast, strong retrieval",
        "    BAAI/bge-base-en-v1.5         768d  strong retrieval, balanced",
        "    BAAI/bge-large-en-v1.5        1024d best retrieval, heavier",
        "    intfloat/e5-small-v2          384d  lightweight",
        "    intfloat/e5-base-v2           768d  balanced",
        "    intfloat/e5-large-v2          1024d best quality",
        "    paraphrase-multilingual-MiniLM-L12-v2  384d  multilingual",
        "",
        "**openai** (API — requires OPENAI_API_KEY env var)",
        "  Models:",
        "    text-embedding-3-small        1536d  fast, cheap (default)",
        "    text-embedding-3-large        3072d  highest quality",
        "    text-embedding-ada-002        1536d  legacy",
    ]
    return "\n".join(lines)


@mcp.tool()
def tv_deploy(
    source: str = "default",
    name: Optional[str] = None,
    target: str = "local",
    output_path: Optional[str] = None,
) -> str:
    """
    Deploy a vector database to a local copy, archive, or cloud storage.

    target options:
      "local"           Copy files to a new name in ~/.turbovec/ (default).
                        Requires: name
      "archive"         Bundle into a portable .tar.gz file.
                        output_path sets the file location (default: ~/.turbovec/<source>.tar.gz)
      "s3://bucket/prefix"    Upload to Amazon S3       (requires: pip install boto3)
      "gs://bucket/prefix"    Upload to Google Cloud Storage  (requires: pip install google-cloud-storage)
      "azure://container/prefix"  Upload to Azure Blob   (requires: pip install azure-storage-blob;
                                                            AZURE_STORAGE_CONNECTION_STRING env var)

    Examples:
      tv_deploy(source="mydb", name="mydb-backup")
      tv_deploy(source="mydb", target="archive", output_path="/tmp/mydb.tar.gz")
      tv_deploy(source="mydb", target="s3://my-bucket/indexes")
      tv_deploy(source="mydb", target="gs://my-bucket/indexes")
    """
    try:
        if target == "local":
            if not name:
                return "Error: 'name' is required for local deployment."
            files = copy_local(source, name, output_path)
            return f"Copied '{source}' → '{name}'\n" + "\n".join(f"  {f}" for f in files)

        elif target == "archive":
            out = export_archive(source, output_path)
            return f"Archive created: {out}"

        else:
            # cloud target
            uris = upload_cloud(source, target)
            return f"Uploaded '{source}' to {target}\n" + "\n".join(f"  {u}" for u in uris)

    except (FileNotFoundError, ValueError, ImportError) as e:
        return f"Error: {e}"


@mcp.tool()
def tv_import(
    source: str,
    db_name: Optional[str] = None,
) -> str:
    """
    Import a database from an archive file or cloud storage.

    source can be:
      /path/to/file.tar.gz              local archive
      s3://bucket/prefix/name.tar.gz    S3 archive object
      gs://bucket/prefix/name.tar.gz    GCS archive object
      azure://container/prefix/name.tar.gz  Azure archive blob
      s3://bucket/prefix/name           S3 raw files (no .tar.gz)
      gs://bucket/prefix/name           GCS raw files
      azure://container/prefix/name     Azure raw files

    db_name overrides the name embedded in the archive or URI.
    """
    try:
        is_cloud = any(source.startswith(p) for p in ("s3://", "gs://", "azure://"))

        if is_cloud and source.endswith(".tar.gz"):
            # download archive to temp, then import
            import tempfile
            provider, bucket, key = _parse_cloud_uri(source)
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name
            _download_single_file(provider, bucket, key, tmp_path)
            name = import_archive(tmp_path, db_name)
            Path(tmp_path).unlink(missing_ok=True)

        elif is_cloud:
            name = download_cloud(source, db_name)

        else:
            name = import_archive(source, db_name)

        db = VectorDB(str(BASE_DIR / name))
        s = db.stats()
        emb = s["embedding"]
        return (
            f"Imported database '{name}'. "
            f"{s['files_indexed']} files · {s['total_chunks']} chunks. "
            f"Embedding: {emb['provider']} / {emb['model']}"
        )
    except (FileNotFoundError, ValueError, ImportError) as e:
        return f"Error: {e}"


@mcp.tool()
def tv_http_server_info(db_name: Optional[str] = None) -> str:
    """
    Show how to start the TurboVec HTTP server for a given database.
    The HTTP server exposes a REST API so other processes or machines can query the index.
    """
    db = db_name or _active_name
    lines = [
        f"To serve database '{db}' over HTTP, run in a terminal:",
        "",
        f"  python http_server.py --db {db} --port 8000",
        "",
        "Or via the CLI:",
        f"  python cli.py serve-http --db {db} --port 8000",
        "",
        "To require a write key for indexing endpoints:",
        f"  python cli.py serve-http --db {db} --port 8000 --write-key mysecret",
        "",
        "Endpoints once running:",
        "  GET  http://localhost:8000/health",
        "  GET  http://localhost:8000/stats",
        "  GET  http://localhost:8000/dbs",
        "  POST http://localhost:8000/search   {query, top_k, filter_files}",
        "  POST http://localhost:8000/index    {path, recursive}  [write key]",
        "  DELETE http://localhost:8000/file   {file_path}        [write key]",
        "",
        "Interactive docs: http://localhost:8000/docs",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Internal helpers for tv_import cloud archive                         #
# ------------------------------------------------------------------ #

def _parse_cloud_uri(uri: str):
    for scheme, provider in [("s3://", "s3"), ("gs://", "gcs"), ("azure://", "azure")]:
        if uri.startswith(scheme):
            rest = uri[len(scheme):]
            bucket, _, key = rest.partition("/")
            return provider, bucket, key
    raise ValueError(f"Unsupported URI: {uri}")


def _download_single_file(provider: str, bucket: str, key: str, dest: str):
    if provider == "s3":
        import boto3
        boto3.client("s3").download_file(bucket, key, dest)
    elif provider == "gcs":
        from google.cloud import storage as gcs
        gcs.Client().bucket(bucket).blob(key).download_to_filename(dest)
    elif provider == "azure":
        import os
        from azure.storage.blob import BlobServiceClient
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        svc = BlobServiceClient.from_connection_string(conn)
        Path(dest).write_bytes(
            svc.get_container_client(bucket).get_blob_client(key).download_blob().readall()
        )


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    mcp.run()
