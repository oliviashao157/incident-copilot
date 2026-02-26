"""Incident retrieval using vector similarity search."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from src.data.incident_store import IncidentStore
from src.embedding.index import FAISSIndex
from src.schema import Category, Incident, Severity, SimilarIncident


class IncidentRetriever:
    """Retrieve similar incidents using FAISS index."""

    def __init__(
        self,
        index: Optional[FAISSIndex] = None,
        loader=None,
        store: Optional[IncidentStore] = None,
        index_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ):
        """Initialize retriever.

        Args:
            index: Pre-built FAISS index
            loader: Pre-loaded IncidentLoader (legacy, wraps in IncidentStore)
            store: Pre-loaded IncidentStore (preferred)
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

        # Initialize store (preferred) or wrap loader
        if store is not None:
            self.store = store
        elif loader is not None:
            # Wrap legacy loader in a store-like interface
            self.store = IncidentStore(data_path or settings.processed_data_dir)
            self.store.loader = loader
            # Build indices from already-loaded data
            all_incidents = loader.get_all()
            if all_incidents:
                self.store._build_indices(all_incidents)
        else:
            self.store = IncidentStore(data_path or settings.processed_data_dir)
            self.store.load_all()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3,
        *,
        categories: Optional[list[Category]] = None,
        severities: Optional[list[Severity]] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> list[SimilarIncident]:
        """Retrieve similar incidents for a text query.

        Args:
            query: Query text (title + description)
            k: Number of results to return
            threshold: Minimum similarity score
            categories: Filter to these categories only
            severities: Filter to these severities only
            after: Only incidents created after this date
            before: Only incidents created before this date

        Returns:
            List of SimilarIncident objects with scores
        """
        has_filters = any(v is not None for v in [categories, severities, after, before])

        if has_filters:
            # Over-fetch from FAISS, then post-filter by metadata
            allowed_ids = self.store.filter_ids(
                categories=categories,
                severities=severities,
                after=after,
                before=before,
            )
            results = self.index.search(query, k=k * 3, threshold=threshold)

            similar_incidents = []
            for incident_id, score in results:
                if incident_id not in allowed_ids:
                    continue
                incident = self.store.get_by_id(incident_id)
                if incident:
                    similar_incidents.append(
                        SimilarIncident(incident=incident, similarity_score=score)
                    )
                if len(similar_incidents) >= k:
                    break

            return similar_incidents
        else:
            # No filters — existing behavior
            results = self.index.search(query, k=k, threshold=threshold)

            similar_incidents = []
            for incident_id, score in results:
                incident = self.store.get_by_id(incident_id)
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
        return self.store.get_by_id(incident_id)
