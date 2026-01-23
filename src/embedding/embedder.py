"""Text embedding using sentence-transformers."""

from typing import Optional

import numpy as np

from src.config import get_settings


class TextEmbedder:
    """Generate text embeddings using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedder with specified model.

        Args:
            model_name: Name of sentence-transformers model (default from settings)
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self._model = None

    @property
    def model(self):
        """Lazily load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding

        Returns:
            2D numpy array of embeddings (n_texts x dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings
