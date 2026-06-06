import json
import numpy as np
from pathlib import Path
from typing import Optional

from embeddings import EmbeddingProvider, DEFAULT_PROVIDER, load_provider

SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".html", ".css",
    ".rs", ".go", ".java", ".c", ".cpp", ".h", ".sh",
    ".csv", ".xml", ".sql",
}

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


class VectorDB:
    """
    Wraps a TurboVec IdMapIndex with a JSON metadata sidecar.

    The embedding provider is chosen at creation time and stored in the
    metadata sidecar so the index can be reopened without re-specifying it.
    Mixing providers on the same index is rejected at index time.

    Files: <db_path>.tvim (vectors) + <db_path>.json (metadata + config)
    """

    def __init__(self, db_path: str, provider: Optional[EmbeddingProvider] = None, bit_width: int = 4):
        self.db_path = Path(db_path)
        self.bit_width = bit_width
        self.index_file = self.db_path.with_suffix(".tvim")
        self.meta_file = self.db_path.with_suffix(".json")

        if self.index_file.exists() and self.meta_file.exists():
            from turbovec import IdMapIndex
            self.index = IdMapIndex.load(str(self.index_file))
            raw = json.loads(self.meta_file.read_text(encoding="utf-8"))
            self.metadata: dict[str, dict] = raw["chunks"]
            self.file_ids: dict[str, list[int]] = raw["file_ids"]
            self.next_id: int = raw["next_id"]
            # restore the provider that was used when this index was created
            stored = raw.get("embedding")
            if stored:
                self.provider = load_provider(stored)
            else:
                # legacy index without stored provider — default to sentence-transformers
                self.provider = provider or DEFAULT_PROVIDER
        else:
            from turbovec import IdMapIndex
            self.provider = provider or DEFAULT_PROVIDER
            self.index = IdMapIndex(dim=self.provider.dim, bit_width=bit_width)
            self.metadata = {}
            self.file_ids = {}
            self.next_id = 1

    # ------------------------------------------------------------------ #
    # Chunking                                                             #
    # ------------------------------------------------------------------ #

    def _chunk(self, text: str) -> list[tuple[str, int, int]]:
        """Return list of (chunk_text, start_line, end_line)."""
        lines = text.split("\n")
        chunks = []
        buf: list[str] = []
        buf_len = 0
        start = 0

        for i, line in enumerate(lines):
            buf.append(line)
            buf_len += len(line) + 1
            if buf_len >= CHUNK_SIZE:
                chunks.append(("\n".join(buf), start, i))
                tail: list[str] = []
                tail_len = 0
                for l in reversed(buf):
                    if tail_len + len(l) > CHUNK_OVERLAP:
                        break
                    tail.insert(0, l)
                    tail_len += len(l) + 1
                buf = tail
                buf_len = tail_len
                start = i - len(tail) + 1

        if buf:
            chunks.append(("\n".join(buf), start, len(lines) - 1))

        return chunks

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def index_file(self, file_path: str) -> int:
        """Index one file. Returns chunk count added."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(file_path)

        self.remove_file(str(path))

        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = self._chunk(text)
        if not chunks:
            return 0

        vectors = self.provider.embed([c[0] for c in chunks])
        new_ids = list(range(self.next_id, self.next_id + len(chunks)))
        self.next_id += len(chunks)

        self.index.add_with_ids(vectors, np.array(new_ids, dtype=np.uint64))

        for vid, (chunk_text, start, end) in zip(new_ids, chunks):
            self.metadata[str(vid)] = {
                "file": str(path),
                "text": chunk_text[:600],
                "start_line": start,
                "end_line": end,
            }

        self.file_ids[str(path)] = new_ids
        self.save()
        return len(chunks)

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        extensions: Optional[set] = None,
    ) -> dict[str, int | str]:
        exts = extensions or SUPPORTED_EXTENSIONS
        pattern = "**/*" if recursive else "*"
        results: dict[str, int | str] = {}
        for f in Path(dir_path).glob(pattern):
            if f.is_file() and f.suffix.lower() in exts:
                try:
                    results[str(f)] = self.index_file(str(f))
                except Exception as e:
                    results[str(f)] = f"error: {e}"
        return results

    def remove_file(self, file_path: str):
        key = str(Path(file_path).resolve())
        if key in self.file_ids:
            for vid in self.file_ids[key]:
                self.index.remove(vid)
                self.metadata.pop(str(vid), None)
            del self.file_ids[key]

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_files: Optional[list[str]] = None,
    ) -> list[dict]:
        vec = self.provider.embed([query])[0]

        if filter_files:
            allowed_ids: list[int] = []
            for f in filter_files:
                allowed_ids.extend(self.file_ids.get(str(Path(f).resolve()), []))
            if not allowed_ids:
                return []
            scores, ids = self.index.search(
                vec, k=top_k, allowlist=np.array(allowed_ids, dtype=np.uint64)
            )
        else:
            scores, ids = self.index.search(vec, k=top_k)

        results = []
        for score, vid in zip(scores, ids):
            meta = self.metadata.get(str(int(vid)), {})
            results.append({
                "score": float(score),
                "file": meta.get("file", ""),
                "text": meta.get("text", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
            })
        return results

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.write(str(self.index_file))
        self.meta_file.write_text(
            json.dumps(
                {
                    "next_id": self.next_id,
                    "chunks": self.metadata,
                    "file_ids": self.file_ids,
                    "embedding": self.provider.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def stats(self) -> dict:
        return {
            "files_indexed": len(self.file_ids),
            "total_chunks": len(self.metadata),
            "embedding": self.provider.to_dict(),
            "bit_width": self.bit_width,
            "index_file": str(self.index_file),
            "meta_file": str(self.meta_file),
        }
