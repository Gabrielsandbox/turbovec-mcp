# turbovec-mcp

> **High-performance RAG for Claude Code** — semantic file search with line-level precision, powered by Google's TurboQuant algorithm.

[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-blueviolet?logo=anthropic)](https://github.com/anthropics/claude-plugins-official)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

`turbovec-mcp` connects Claude Code to a local vector database built on [TurboVec](https://github.com/RyanCodrai/turbovec) — a Rust-native vector index using Google Research's **TurboQuant** algorithm (ICLR 2026). Instead of reading every file from scratch, Claude can semantically search your codebase, notes, or documents and retrieve the exact chunks it needs — with file paths and line numbers.

**Key properties:**
- 🦀 **Rust core** — TurboVec beats FAISS IndexPQFastScan by 12–20% on ARM; matches it on x86
- 🧠 **87% less memory** — 10M vectors fit in 4 GB instead of 31 GB
- 🔌 **No training** — TurboQuant is data-oblivious; no codebook training, no separate train phase
- 🔒 **Fully local** — embeddings run offline via `sentence-transformers`
- 💾 **Persistent** — indexes survive sessions, stored in `~/.turbovec/`

---

## Demo

```
You: search for how the authentication middleware works
Claude: [calls tv_search("authentication middleware")]

### 1. server/middleware/auth.ts:42  score=0.91
```typescript
export async function authMiddleware(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1]
  if (!token) return res.status(401).json({ error: 'No token' })
  ...
```

### 2. docs/ARCHITECTURE.md:118  score=0.87
The auth layer validates JWTs issued by Supabase Auth...
```

---

## Installation

```bash
git clone https://github.com/Gabrielsandbox/turbovec-mcp
cd turbovec-mcp
pip install -r requirements.txt
```

Add the server to your project's `.mcp.json` (replace the path with where you cloned):

```json
{
  "mcpServers": {
    "turbovec": {
      "command": "python",
      "args": ["/path/to/turbovec-mcp/server.py"]
    }
  }
}
```

Restart Claude Code. The `turbovec` server will appear in your MCP tool list automatically.

---

## MCP Tools (available inside Claude Code)

| Tool | Description |
|------|-------------|
| `tv_index` | Index a file or directory (supports `.md` `.py` `.ts` `.json` `.sql` and 20+ more) |
| `tv_search` | Semantic search → returns chunks with file path, line range, and similarity score |
| `tv_remove_file` | Remove a file's chunks from the index |
| `tv_stats` | Show index statistics (files, chunks, db path) |
| `tv_use_db` | Switch the active database by name |
| `tv_list_dbs` | List all databases in `~/.turbovec/` |
| `tv_deploy` | Export/copy a named database (for sharing or deployment) |

---

## CLI

The bundled CLI lets you manage vector databases directly from the terminal — no Claude Code required.

```bash
# Index your entire project
python cli.py index ./my-project --db myproject

# Semantic search from the terminal
python cli.py search "JWT token validation" --db myproject --top-k 10

# JSON output for scripting
python cli.py search "database schema" --json

# Export a database by name
python cli.py deploy production --source myproject --output ./exports/

# List all databases
python cli.py list-dbs

# Show index stats
python cli.py stats --db myproject

# Verify a database exists and check its stats
python cli.py use myproject
```

---

## Architecture

```
turbovec-mcp/
├── server.py      # FastMCP server — 7 tools exposed to Claude Code
├── indexer.py     # VectorDB class: chunk → embed → IdMapIndex → persist
├── cli.py         # Typer CLI for terminal use
├── requirements.txt
├── pyproject.toml
└── .claude-plugin/
    └── plugin.json  # Anthropic plugin manifest
```

**Data flow:**

```
File  →  chunk(400 chars, 60 overlap)  →  all-MiniLM-L6-v2 (dim=384)
     →  IdMapIndex (4-bit TurboQuant)  →  ~/.turbovec/<name>.tvim
                                           ~/.turbovec/<name>.json  (metadata)
```

Each chunk stores: `file path`, `text preview`, `start_line`, `end_line`.  
Metadata sidecar (JSON) maps vector IDs → chunk info. Supports O(1) deletion by file.

---

## Multiple databases

```bash
# Create separate indexes for different projects
python cli.py index ~/code/api --db api
python cli.py index ~/vault --db notes
python cli.py index ~/docs --db docs

# Switch active database inside Claude Code (MCP tool)
# Ask Claude: "switch to the api database"
# Claude calls: tv_use_db with name="api"
```

---

## Requirements

- Python 3.11+
- `turbovec` (Rust extension with Python bindings)
- `sentence-transformers>=3.0`
- `mcp[cli]>=1.0`
- `typer>=0.12`
- `numpy>=1.26`

---

## How TurboQuant works

TurboVec is built on Google Research's **TurboQuant** algorithm (ICLR 2026). Unlike PQ (Product Quantization) or other vector compression methods, TurboQuant:

- Matches the **Shannon lower bound on distortion** for uniform sources
- Requires **no training data** — the quantizer is data-oblivious
- Uses hand-written **NEON** (ARM) and **AVX-512BW** (x86) kernels for maximum throughput

This means you get FAISS-level recall with dramatically lower memory, and no offline training pipeline.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

Issues and PRs welcome. To add support for a new file type, add its extension to `SUPPORTED_EXTENSIONS` in [indexer.py](indexer.py).

---

Built on [TurboVec](https://github.com/RyanCodrai/turbovec) by Ryan Codrai — Rust vector index using Google Research's TurboQuant (ICLR 2026).
