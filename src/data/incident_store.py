"""Schema-validated incident storage with metadata indices."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from src.data.loader import IncidentLoader
from src.schema import Category, Incident, Severity


class IncidentStore:
    """Incident storage with in-memory metadata indices for filtered queries.

    Wraps IncidentLoader and adds indices by category, severity, and date
    for efficient metadata-based filtering.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize store with data directory.

        Args:
            data_dir: Path to processed data directory. Uses settings default if None.
        """
        self.loader = IncidentLoader(data_dir)
        self._by_category: dict[Category, set[str]] = {}
        self._by_severity: dict[Severity, set[str]] = {}
        self._by_date: list[tuple[datetime, str]] = []  # sorted by datetime
        self._indexed = False

    def load_all(self, pattern: str = "*.jsonl") -> list[Incident]:
        """Load all incidents and build metadata indices."""
        incidents = self.loader.load_all(pattern)
        self._build_indices(incidents)
        return incidents

    def _build_indices(self, incidents: list[Incident]) -> None:
        """Build in-memory metadata indices from loaded incidents."""
        self._by_category.clear()
        self._by_severity.clear()
        self._by_date.clear()

        for incident in incidents:
            self._index_incident(incident)

        self._by_date.sort(key=lambda x: x[0])
        self._indexed = True

    def _index_incident(self, incident: Incident) -> None:
        """Add a single incident to all indices."""
        # Category index
        self._by_category.setdefault(incident.category, set()).add(incident.id)

        # Severity index
        self._by_severity.setdefault(incident.severity, set()).add(incident.id)

        # Date index
        self._by_date.append((incident.created_at, incident.id))

    def filter_ids(
        self,
        categories: Optional[list[Category]] = None,
        severities: Optional[list[Severity]] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> set[str]:
        """Filter incident IDs by metadata criteria using AND logic.

        Args:
            categories: Include only these categories.
            severities: Include only these severities.
            after: Include only incidents created after this datetime.
            before: Include only incidents created before this datetime.

        Returns:
            Set of incident IDs matching all criteria.
        """
        if not self._indexed:
            return set()

        result_sets: list[set[str]] = []

        if categories is not None:
            cat_ids: set[str] = set()
            for cat in categories:
                cat_ids |= self._by_category.get(cat, set())
            result_sets.append(cat_ids)

        if severities is not None:
            sev_ids: set[str] = set()
            for sev in severities:
                sev_ids |= self._by_severity.get(sev, set())
            result_sets.append(sev_ids)

        if after is not None or before is not None:
            date_ids: set[str] = set()
            for dt, iid in self._by_date:
                if after is not None and dt < after:
                    continue
                if before is not None and dt > before:
                    continue
                date_ids.add(iid)
            result_sets.append(date_ids)

        if not result_sets:
            # No filters applied — return all IDs
            return {inc.id for inc in self.loader.get_all()}

        # AND logic: intersect all filter sets
        result = result_sets[0]
        for s in result_sets[1:]:
            result &= s
        return result

    def get_by_id(self, incident_id: str) -> Optional[Incident]:
        """Get an incident by ID."""
        return self.loader.get_by_id(incident_id)

    def get_all(self) -> list[Incident]:
        """Get all loaded incidents."""
        return self.loader.get_all()

    def get_stats(self) -> dict:
        """Get statistics about loaded incidents."""
        return self.loader.get_stats()
