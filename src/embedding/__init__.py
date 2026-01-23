"""Embedding and index management module."""

from src.embedding.embedder import TextEmbedder
from src.embedding.index import FAISSIndex

__all__ = ["TextEmbedder", "FAISSIndex"]
