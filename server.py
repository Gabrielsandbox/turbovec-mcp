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
# MCP server                                                           #
# ------------------------------------------------------------------ #

mcp = FastMCP("turbovec")


@mcp.tool()
def tv_index(path: str, recursive: bool = True) -> str:
    """
    Index a file or directory into the active vector database.
    Supports .md .txt .py .js .ts .json .yaml .csv .sql and more.
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
    """Return statistics about the active vector database."""
    db = get_db()
    return json.dumps(db.stats(), indent=2)


@mcp.tool()
def tv_use_db(name: str) -> str:
    """
    Switch the active vector database by name.
    All subsequent operations will use this database.
    Available databases are stored in ~/.turbovec/<name>.
    """
    global _active_db, _active_name
    _active_name = name
    _active_db = VectorDB(str(BASE_DIR / name))
    s = _active_db.stats()
    return (
        f"Switched to database '{name}'. "
        f"{s['files_indexed']} files · {s['total_chunks']} chunks indexed."
    )


@mcp.tool()
def tv_list_dbs() -> str:
    """List all available vector databases in ~/.turbovec/."""
    dbs = sorted({f.stem for f in BASE_DIR.glob("*.tvim")})
    if not dbs:
        return "No databases found. Run tv_index to create one."
    active_marker = lambda n: " ◀ active" if n == _active_name else ""
    return "\n".join(f"  • {n}{active_marker(n)}" for n in dbs)


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
