# turbovec-mcp

> **High-performance RAG for Claude Code** — semantic file search with line-level precision, pluggable embedding providers, powered by Google's TurboQuant algorithm.

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
- 🔒 **Fully local** — default embeddings run offline via `sentence-transformers`
- 🔀 **Pluggable embeddings** — swap providers per database; choice is stored and auto-restored
- 💾 **Persistent** — indexes survive sessions, stored in `~/.turbovec/`

---

## Demo

```
You: search for how the authentication middleware works
Claude: [calls tv_search("authentication middleware")]

### 1. server/middleware/auth.ts:42  score=0.91
export async function authMiddleware(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1]
  if (!token) return res.status(401).json({ error: 'No token' })
  ...

### 2. docs/ARCHITECTURE.md:118  score=0.87
The auth layer validates JWTs issued by Supabase Auth...
```

---

## Installation

```bash
git clone https://github.com/Gabrielsandbox/turbovec-mcp
cd turbovec-mcp
pip install -r requirements.txt        # sentence-transformers included
# pip install openai                   # only if using the OpenAI provider
```

Add the server to your project's `.mcp.json`:

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

Restart Claude Code — the `turbovec` tools will appear automatically.

---

## Embedding providers

Each database is created with a specific embedding provider. The choice is stored in the index metadata and restored automatically on every subsequent open — you never have to re-specify it.

### sentence-transformers *(local, offline, default)*

| Model | Dim | Notes |
|-------|-----|-------|
| `all-MiniLM-L6-v2` | 384 | Fast, good general-purpose **[default]** |
| `all-MiniLM-L12-v2` | 384 | Slightly better recall than L6 |
| `all-mpnet-base-v2` | 768 | Higher quality, slower |
| `BAAI/bge-small-en-v1.5` | 384 | Fast, strong retrieval |
| `BAAI/bge-base-en-v1.5` | 768 | Strong retrieval, balanced |
| `BAAI/bge-large-en-v1.5` | 1024 | Best retrieval, heavier |
| `intfloat/e5-base-v2` | 768 | Balanced |
| `intfloat/e5-large-v2` | 1024 | Best quality |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual |

### openai *(API — requires `OPENAI_API_KEY`)*

| Model | Dim | Notes |
|-------|-----|-------|
| `text-embedding-3-small` | 1536 | Fast, cheap **[default]** |
| `text-embedding-3-large` | 3072 | Highest quality |
| `text-embedding-ada-002` | 1536 | Legacy |

### Adding a custom provider

Subclass `EmbeddingProvider` in [`embeddings.py`](embeddings.py), implement `embed()`, `dim`, `to_dict()`, and `from_dict()`, then register in the `PROVIDERS` dict.

---

## Deployment

Three deployment modes are supported. All are available from both the CLI and from within Claude Code via MCP tools.

### 1. Archive (.tar.gz)

Bundle the index into a single portable file — easy to share, attach to releases, or copy between machines.

```bash
# Export
python cli.py deploy --source mydb --target archive --output /tmp/mydb.tar.gz

# Import on another machine
python cli.py import-db /tmp/mydb.tar.gz
python cli.py import-db /tmp/mydb.tar.gz --name newname
```

### 2. Cloud storage

Upload and download to S3, Google Cloud Storage, or Azure Blob Storage. Install only the SDK you need:

```bash
pip install boto3                 # Amazon S3
pip install google-cloud-storage  # Google Cloud Storage
pip install azure-storage-blob    # Azure Blob Storage
```

```bash
# Upload to S3
python cli.py deploy --source mydb --target s3://my-bucket/indexes

# Upload to GCS
python cli.py deploy --source mydb --target gs://my-bucket/indexes

# Upload to Azure (requires AZURE_STORAGE_CONNECTION_STRING env var)
python cli.py deploy --source mydb --target azure://my-container/indexes

# Download from any cloud
python cli.py import-db s3://my-bucket/indexes/mydb
python cli.py import-db gs://my-bucket/indexes/mydb
python cli.py import-db s3://my-bucket/indexes/mydb.tar.gz  # archive object
```

Cloud credentials are read from the environment automatically:
- **S3**: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, or `~/.aws/credentials`
- **GCS**: `GOOGLE_APPLICATION_CREDENTIALS`
- **Azure**: `AZURE_STORAGE_CONNECTION_STRING`

### 3. HTTP server

Serve a database over REST so any process or machine can query it — no local index required on the client.

```bash
pip install fastapi uvicorn[standard]

# Start the server
python cli.py serve-http --db mydb --port 8000

# Protect write endpoints with a bearer token
python cli.py serve-http --db mydb --port 8000 --write-key mysecret
```

**Endpoints:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | — | Liveness check |
| `GET` | `/stats` | — | Index stats + provider info |
| `GET` | `/dbs` | — | List all databases |
| `POST` | `/search` | — | `{query, top_k, filter_files}` |
| `POST` | `/index` | write key | `{path, recursive}` |
| `DELETE` | `/file` | write key | `{file_path}` |

Interactive docs at `http://localhost:8000/docs` once running.

**Query from anywhere:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "JWT authentication", "top_k": 5}'
```

---

## MCP tools

All tools are available inside Claude Code once the server is running.

| Tool | Description |
|------|-------------|
| `tv_create_db` | Create a new database and choose its embedding provider/model |
| `tv_index` | Index a file or directory (`.md` `.py` `.ts` `.json` `.sql` and 20+ more) |
| `tv_search` | Semantic search → chunks with file path, line range, similarity score |
| `tv_remove_file` | Remove a file's chunks from the index |
| `tv_stats` | Show index stats including the active embedding provider |
| `tv_use_db` | Switch to a different database (provider auto-restored) |
| `tv_list_dbs` | List all databases with their embedding provider and model |
| `tv_list_providers` | List all available providers and models |
| `tv_deploy` | Deploy to local copy, `.tar.gz` archive, S3, GCS, or Azure |
| `tv_import` | Import from a local archive or cloud URI |
| `tv_http_server_info` | Show how to start the HTTP server for a database |

---

## CLI

Full database management from the terminal — no Claude Code required.

```bash
# See available providers and models
python cli.py providers

# Create a database with a specific provider
python cli.py create-db myproject
python cli.py create-db research --provider openai --model text-embedding-3-large
python cli.py create-db multilingual --model paraphrase-multilingual-MiniLM-L12-v2

# Index files (provider is read from the db; --provider/--model only needed for new dbs)
python cli.py index ./my-project --db myproject
python cli.py index ./my-project --db myproject --recursive

# Semantic search
python cli.py search "JWT token validation" --db myproject --top-k 10
python cli.py search "database schema" --db myproject --json   # raw JSON output

# Inspect databases
python cli.py list-dbs          # shows provider + model for each db
python cli.py stats --db myproject
python cli.py use myproject     # verify db exists and print its stats

# Manage files
python cli.py remove ./file.py --db myproject

# Deploy — local copy
python cli.py deploy backup --source myproject

# Deploy — archive
python cli.py deploy --source myproject --target archive --output /tmp/myproject.tar.gz

# Deploy — cloud
python cli.py deploy --source myproject --target s3://my-bucket/indexes
python cli.py deploy --source myproject --target gs://my-bucket/indexes
python cli.py deploy --source myproject --target azure://my-container/indexes

# Import
python cli.py import-db /tmp/myproject.tar.gz
python cli.py import-db s3://my-bucket/indexes/myproject

# HTTP server
python cli.py serve-http --db myproject --port 8000
python cli.py serve-http --db myproject --port 8000 --write-key mysecret

# Start the MCP server manually
python cli.py serve
```

---

## Architecture

```
turbovec-mcp/
├── embeddings.py    # EmbeddingProvider ABC + SentenceTransformer + OpenAI providers
├── indexer.py       # VectorDB: chunk → embed → IdMapIndex → persist
├── deploy.py        # Archive export/import + S3 / GCS / Azure upload/download
├── http_server.py   # FastAPI REST server (serve-http)
├── server.py        # FastMCP server — 11 MCP tools exposed to Claude Code
├── cli.py           # Typer CLI for terminal use
├── requirements.txt
├── pyproject.toml
└── .claude-plugin/
    └── plugin.json  # Anthropic plugin manifest
```

**Data flow:**

```
File  →  chunk(400 chars, 60 overlap)  →  EmbeddingProvider.embed()
     →  IdMapIndex (4-bit TurboQuant)  →  ~/.turbovec/<name>.tvim
                                           ~/.turbovec/<name>.json  (chunks + provider config)
```

The metadata sidecar stores the provider name, model, and dim. On reload, `load_provider(config)` reconstructs the exact provider — the index and its embedding model are always in sync.

---

## Multiple databases

```bash
# Different providers for different use cases
python cli.py create-db code     --model BAAI/bge-base-en-v1.5          # strong code retrieval
python cli.py create-db notes                                             # fast local default
python cli.py create-db research --provider openai --model text-embedding-3-large  # best quality

python cli.py index ~/code       --db code
python cli.py index ~/vault      --db notes
python cli.py index ~/papers     --db research

python cli.py list-dbs
#  • code      [sentence-transformers / BAAI/bge-base-en-v1.5  dim=768]
#  • notes     [sentence-transformers / all-MiniLM-L6-v2  dim=384]
#  • research  [openai / text-embedding-3-large  dim=3072]
```

Inside Claude Code, ask: *"switch to the research database"* — Claude calls `tv_use_db("research")` and the right provider loads automatically.

---

## Requirements

- Python 3.11+
- `turbovec` — Rust vector index with Python bindings
- `sentence-transformers>=3.0` — local embeddings (default provider)
- `mcp[cli]>=1.0` — MCP server framework
- `typer>=0.12` — CLI
- `numpy>=1.26`
- `openai` *(optional)* — only needed for the OpenAI provider

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

Issues and PRs welcome.

- **New file type:** add its extension to `SUPPORTED_EXTENSIONS` in [`indexer.py`](indexer.py)
- **New embedding provider:** subclass `EmbeddingProvider` in [`embeddings.py`](embeddings.py) and add to `PROVIDERS`

---

Built on [TurboVec](https://github.com/RyanCodrai/turbovec) by Ryan Codrai — Rust vector index using Google Research's TurboQuant (ICLR 2026).
