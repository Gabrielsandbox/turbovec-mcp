"""
TurboVec MCP Server
Exposes vector-search tools to Claude Code via the Model Context Protocol.
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent))
from indexer import VectorDB
from embeddings import make_provider, PROVIDERS

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
def tv_deploy(name: str, source: str = "default", output_path: Optional[str] = None) -> str:
    """
    Deploy (export/copy) a named vector database to a destination.
    Useful for sharing indexes or deploying to another machine.
    source: name of the database to export (default: 'default')
    output_path: optional absolute path; defaults to ~/.turbovec/<name>
    """
    src_base = BASE_DIR / source
    dst_dir = Path(output_path) if output_path else BASE_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for ext in (".tvim", ".json"):
        src = src_base.with_suffix(ext)
        if src.exists():
            dst = dst_dir / (name + ext)
            shutil.copy2(src, dst)
            copied.append(str(dst))

    if not copied:
        return f"Source database '{source}' not found."

    return f"Deployed '{source}' → '{name}'\n" + "\n".join(f"  {f}" for f in copied)


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    mcp.run()
