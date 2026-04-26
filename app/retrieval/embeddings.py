"""Embedding model — local HuggingFace via sentence-transformers (free, no API key).

Default: ``BAAI/bge-small-en-v1.5`` — free, runs locally on CPU, 384-dim vectors.

Cost comparison:
  Model                         Dim    Cost      Speed       Quality
  BAAI/bge-small-en-v1.5        384    $0.00     ~fast       Better
  BAAI/bge-base-en-v1.5         768    $0.00     ~moderate   Very good
  sentence-transformers/all-MiniLM-L6-v2  384  $0.00  ~fast  Good
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

HF_MODEL_NAME = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
HF_DEVICE = os.getenv("HF_DEVICE", "cpu")

_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


def get_embedding_dim() -> int:
    dim = _MODEL_DIMENSIONS.get(HF_MODEL_NAME)
    if dim is None:
        logger.warning(
            "Unknown embedding model '%s' — defaulting to 384 dims.", HF_MODEL_NAME
        )
        return 384
    return dim


@runtime_checkable
class Embeddings(Protocol):
    """Minimal protocol for embedding models used in retrieval indexes."""

    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class HFEmbeddings:
    """Thin synchronous wrapper around sentence-transformers SentenceTransformer."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading HuggingFace embedding model: %s on %s (FREE)",
            HF_MODEL_NAME,
            HF_DEVICE,
        )
        self._model = SentenceTransformer(HF_MODEL_NAME, device=HF_DEVICE)

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def get_embeddings() -> HFEmbeddings:
    """Return the configured embedding model instance."""
    return HFEmbeddings()
