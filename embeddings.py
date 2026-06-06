"""
Pluggable embedding providers for turbovec-mcp.

Supported:
  - sentence-transformers  (local, offline, any SBERT-compatible model)
  - openai                 (API, requires OPENAI_API_KEY env var)

Adding a new provider:
  1. Subclass EmbeddingProvider
  2. Set `name` class attribute
  3. Implement embed(), dim property, to_dict(), from_dict()
  4. Register in PROVIDERS dict at the bottom
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class EmbeddingProvider(ABC):
    name: ClassVar[str]

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vector dimension for this provider/model."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns float32 array of shape (len(texts), dim)."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialise config to dict (stored in index metadata)."""

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict) -> "EmbeddingProvider":
        """Reconstruct provider from stored config dict."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


# ------------------------------------------------------------------ #
# sentence-transformers (local, offline)                              #
# ------------------------------------------------------------------ #

_ST_KNOWN_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
}


class SentenceTransformerProvider(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2", _dim: int | None = None):
        self.model = model
        self._dim = _dim or _ST_KNOWN_DIMS.get(model)
        self._model_obj = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim

    def _load(self):
        if self._model_obj is None:
            from sentence_transformers import SentenceTransformer
            self._model_obj = SentenceTransformer(self.model)
            self._dim = self._model_obj.get_sentence_embedding_dimension()
        return self._model_obj

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._load().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def to_dict(self) -> dict:
        return {"provider": self.name, "model": self.model, "dim": self.dim}

    @classmethod
    def from_dict(cls, config: dict) -> "SentenceTransformerProvider":
        return cls(model=config["model"], _dim=config.get("dim"))


# ------------------------------------------------------------------ #
# OpenAI embeddings (API)                                             #
# ------------------------------------------------------------------ #

_OAI_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._dim = _OAI_DIMS.get(model, 1536)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI provider requires: pip install openai")
        if not self.api_key:
            raise ValueError("OpenAI provider requires OPENAI_API_KEY environment variable")

        client = openai.OpenAI(api_key=self.api_key)
        # batch in chunks of 2048 (OpenAI limit)
        all_vecs = []
        for i in range(0, len(texts), 2048):
            resp = client.embeddings.create(input=texts[i : i + 2048], model=self.model)
            all_vecs.extend(r.embedding for r in resp.data)
        return np.array(all_vecs, dtype=np.float32)

    def to_dict(self) -> dict:
        return {"provider": self.name, "model": self.model, "dim": self.dim}

    @classmethod
    def from_dict(cls, config: dict) -> "OpenAIProvider":
        return cls(model=config["model"])


# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    SentenceTransformerProvider.name: SentenceTransformerProvider,
    OpenAIProvider.name: OpenAIProvider,
}

DEFAULT_PROVIDER = SentenceTransformerProvider()


def make_provider(provider: str, model: str | None = None) -> EmbeddingProvider:
    """
    Construct a provider by name.

    Examples:
        make_provider("sentence-transformers")
        make_provider("sentence-transformers", "BAAI/bge-base-en-v1.5")
        make_provider("openai", "text-embedding-3-large")
    """
    cls = PROVIDERS.get(provider)
    if cls is None:
        supported = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown provider '{provider}'. Supported: {supported}")

    if model:
        return cls(model=model)
    return cls()


def load_provider(config: dict) -> EmbeddingProvider:
    """Reconstruct a provider from a stored config dict."""
    cls = PROVIDERS.get(config.get("provider", "sentence-transformers"))
    if cls is None:
        raise ValueError(f"Cannot load unknown provider: {config}")
    return cls.from_dict(config)
