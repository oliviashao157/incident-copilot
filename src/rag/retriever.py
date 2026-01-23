"""Incident retrieval using vector similarity search."""

from pathlib import Path
from typing import Optional

from src.data.loader import IncidentLoader
from src.embedding.index import FAISSIndex
from src.schema import Incident, SimilarIncident


class IncidentRetriever:
    """Retrieve similar incidents using FAISS index."""

    def __init__(
        self,
        index: Optional[FAISSIndex] = None,
        loader: Optional[IncidentLoader] = None,
        index_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ):
        """Initialize retriever.

        Args:
            index: Pre-built FAISS index
            loader: Pre-loaded IncidentLoader
            index_path: Path to load index from
            data_path: Path to load incidents from
        """
        from src.config import get_settings

        settings = get_settings()

        # Initialize or load index
        if index is not None:
            self.index = index
        elif index_path:
            self.index = FAISSIndex(index_path=index_path)
        else:
            self.index = FAISSIndex(index_path=settings.index_dir)

        # Initialize or load loader
        if loader is not None:
            self.loader = loader
        else:
            self.loader = IncidentLoader(data_path or settings.processed_data_dir)
            # Load all incidents
            self.loader.load_all()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3,
    ) -> list[SimilarIncident]:
        """Retrieve similar incidents for a text query.

        Args:
            query: Query text (title + description)
            k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            List of SimilarIncident objects with scores
        """
        results = self.index.search(query, k=k, threshold=threshold)

        similar_incidents = []
        for incident_id, score in results:
            incident = self.loader.get_by_id(incident_id)
            if incident:
                similar_incidents.append(
                    SimilarIncident(incident=incident, similarity_score=score)
                )

        return similar_incidents

    def retrieve_for_incident(
        self,
        incident: Incident,
        k: int = 5,
        threshold: float = 0.3,
    ) -> list[SimilarIncident]:
        """Retrieve similar incidents for a given incident.

        Args:
            incident: Query incident
            k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            List of SimilarIncident objects with scores
        """
        query = incident.to_embedding_text()
        return self.retrieve(query, k=k, threshold=threshold)

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get an incident by ID."""
        return self.loader.get_by_id(incident_id)
