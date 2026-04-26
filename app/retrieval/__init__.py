"""Retrieval subpackage — import indices from their modules to avoid import cycles."""

from .embeddings import get_embedding_dim, get_embeddings

__all__ = ["get_embedding_dim", "get_embeddings"]
