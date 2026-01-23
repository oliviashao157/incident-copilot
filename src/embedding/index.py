"""FAISS index management for similarity search."""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.embedding.embedder import TextEmbedder
from src.schema import Incident


class FAISSIndex:
    """FAISS-based vector index for incident similarity search."""

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        index_path: Optional[Path] = None,
    ):
        """Initialize FAISS index.

        Args:
            embedder: TextEmbedder instance (created if not provided)
            index_path: Path to load existing index from
        """
        self.embedder = embedder or TextEmbedder()
        self.index: Optional[faiss.Index] = None
        self.id_map: dict[int, str] = {}  # vector_id -> incident_id
        self.incident_map: dict[str, int] = {}  # incident_id -> vector_id

        if index_path and index_path.exists():
            self.load(index_path)

    def build(self, incidents: list[Incident]) -> None:
        """Build the index from a list of incidents.

        Args:
            incidents: List of incidents to index
        """
        if not incidents:
            raise ValueError("Cannot build index with empty incident list")

        # Generate embeddings
        texts = [incident.to_embedding_text() for incident in incidents]
        embeddings = self.embedder.embed_batch(texts)

        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Add vectors to index
        self.index.add(embeddings.astype(np.float32))

        # Build ID mappings
        self.id_map = {i: incident.id for i, incident in enumerate(incidents)}
        self.incident_map = {incident.id: i for i, incident in enumerate(incidents)}

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search for similar incidents.

        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (incident_id, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")

        # Generate query embedding
        query_embedding = self.embedder.embed(query).reshape(1, -1).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Filter by threshold and map to incident IDs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                incident_id = self.id_map.get(int(idx))
                if incident_id:
                    results.append((incident_id, float(score)))

        return results

    def search_by_incident(
        self,
        incident: Incident,
        k: int = 5,
        threshold: float = 0.0,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Search for incidents similar to a given incident.

        Args:
            incident: Incident to find similar ones for
            k: Number of results to return
            threshold: Minimum similarity score
            exclude_self: Whether to exclude the query incident from results

        Returns:
            List of (incident_id, similarity_score) tuples
        """
        query_text = incident.to_embedding_text()
        results = self.search(query_text, k=k + (1 if exclude_self else 0), threshold=threshold)

        if exclude_self:
            results = [(id_, score) for id_, score in results if id_ != incident.id]
            results = results[:k]

        return results

    def save(self, path: Path) -> None:
        """Save index and mappings to disk.

        Args:
            path: Directory to save to
        """
        if self.index is None:
            raise ValueError("No index to save")

        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save ID mappings
        mapping_file = path / "mappings.json"
        with open(mapping_file, "w") as f:
            json.dump(
                {
                    "id_map": {str(k): v for k, v in self.id_map.items()},
                    "incident_map": self.incident_map,
                },
                f,
            )

    def load(self, path: Path) -> None:
        """Load index and mappings from disk.

        Args:
            path: Directory to load from
        """
        index_file = path / "index.faiss"
        mapping_file = path / "mappings.json"

        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_file))

        # Load ID mappings
        if mapping_file.exists():
            with open(mapping_file) as f:
                mappings = json.load(f)
                self.id_map = {int(k): v for k, v in mappings["id_map"].items()}
                self.incident_map = mappings["incident_map"]

    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
