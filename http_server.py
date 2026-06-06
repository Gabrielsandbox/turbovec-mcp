"""
TurboVec HTTP Server
Serves a vector database over a REST API so any process or machine can query it.

Start:
  python http_server.py --db myproject --port 8000
  python cli.py serve-http --db myproject --port 8000

Endpoints:
  GET  /health              liveness check
  GET  /stats               index statistics (files, chunks, provider)
  GET  /dbs                 list all databases in ~/.turbovec/
  POST /search              semantic search
  POST /index               index a file or directory  (write key required)
  DELETE /file              remove a file from the index  (write key required)

Authentication:
  Write endpoints (POST /index, DELETE /file) require a bearer token when
  --write-key is set on startup. Pass it as:  Authorization: Bearer <key>
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from indexer import VectorDB

BASE_DIR = Path.home() / ".turbovec"

# ------------------------------------------------------------------ #
# FastAPI app                                                          #
# ------------------------------------------------------------------ #

try:
    from fastapi import FastAPI, HTTPException, Depends, Security, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise ImportError(
        "HTTP server requires: pip install fastapi uvicorn[standard]"
    )

app = FastAPI(
    title="TurboVec HTTP Server",
    description="REST API for querying a TurboVec vector database.",
    version="0.1.0",
)

# runtime state — set by serve()
_db: Optional[VectorDB] = None
_db_name: str = "default"
_write_key: Optional[str] = None

security = HTTPBearer(auto_error=False)


def _require_write(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    if _write_key is None:
        return  # no key configured → write endpoints are open
    if credentials is None or credentials.credentials != _write_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing write key",
        )


def _get_db() -> VectorDB:
    if _db is None:
        raise HTTPException(status_code=503, detail="No database loaded")
    return _db


# ------------------------------------------------------------------ #
# Schemas                                                              #
# ------------------------------------------------------------------ #

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_files: Optional[list[str]] = None


class SearchResult(BaseModel):
    score: float
    file: str
    text: str
    start_line: int
    end_line: int


class IndexRequest(BaseModel):
    path: str
    recursive: bool = True


class RemoveRequest(BaseModel):
    file_path: str


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.get("/health", tags=["status"])
def health():
    return {"status": "ok", "db": _db_name}


@app.get("/stats", tags=["status"])
def stats():
    return _get_db().stats()


@app.get("/dbs", tags=["status"])
def list_dbs():
    dbs = sorted({f.stem for f in BASE_DIR.glob("*.tvim")})
    return {"databases": dbs, "active": _db_name}


@app.post("/search", response_model=list[SearchResult], tags=["search"])
def search(req: SearchRequest):
    """Semantic search. Returns chunks with file path, line range, and score."""
    db = _get_db()
    results = db.search(req.query, top_k=req.top_k, filter_files=req.filter_files)
    return results


@app.post("/index", tags=["write"])
def index_path(req: IndexRequest, _: None = Depends(_require_write)):
    """Index a file or directory. Requires write key if one is configured."""
    db = _get_db()
    p = Path(req.path).expanduser()

    if p.is_file():
        count = db.index_file(str(p))
        return {"indexed": 1, "chunks": count}

    if p.is_dir():
        results = db.index_directory(str(p), recursive=req.recursive)
        ok = {k: v for k, v in results.items() if isinstance(v, int)}
        errors = {k: v for k, v in results.items() if isinstance(v, str)}
        return {
            "indexed": len(ok),
            "chunks": sum(ok.values()),
            "errors": errors or None,
        }

    raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")


@app.delete("/file", tags=["write"])
def remove_file(req: RemoveRequest, _: None = Depends(_require_write)):
    """Remove a file's chunks from the index."""
    db = _get_db()
    db.remove_file(req.file_path)
    db.save()
    return {"removed": req.file_path}


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def serve(
    db_name: str = "default",
    host: str = "0.0.0.0",
    port: int = 8000,
    write_key: Optional[str] = None,
    reload: bool = False,
):
    global _db, _db_name, _write_key
    _db_name = db_name
    _write_key = write_key

    db_path = BASE_DIR / db_name
    if not db_path.with_suffix(".tvim").exists():
        print(f"✗ Database '{db_name}' not found in {BASE_DIR}")
        raise SystemExit(1)

    _db = VectorDB(str(db_path))
    s = _db.stats()
    emb = s["embedding"]
    print(f"✓ Loaded '{db_name}'  {s['files_indexed']} files · {s['total_chunks']} chunks")
    print(f"  Embedding: {emb['provider']} / {emb['model']}  dim={emb['dim']}")
    if write_key:
        print(f"  Write key: set (pass as 'Authorization: Bearer <key>')")
    else:
        print(f"  Write key: not set (write endpoints are open)")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Docs: http://{'localhost' if host == '0.0.0.0' else host}:{port}/docs\n")

    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboVec HTTP server")
    parser.add_argument("--db",        default="default",   help="Database name to serve")
    parser.add_argument("--host",      default="0.0.0.0",   help="Bind address")
    parser.add_argument("--port",      default=8000, type=int)
    parser.add_argument("--write-key", default=None,        help="Bearer token for write endpoints")
    args = parser.parse_args()

    serve(
        db_name=args.db,
        host=args.host,
        port=args.port,
        write_key=args.write_key,
    )
