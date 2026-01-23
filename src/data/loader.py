"""Unified data loading and processing."""

import json
from pathlib import Path
from typing import Optional

from src.schema import Incident


class IncidentLoader:
    """Load and manage incident data from various sources."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize loader with data directory."""
        from src.config import get_settings

        settings = get_settings()
        self.data_dir = data_dir or settings.processed_data_dir
        self._incidents: dict[str, Incident] = {}

    def load_jsonl(self, path: Path) -> list[Incident]:
        """Load incidents from a JSONL file."""
        incidents = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    incident = Incident(**data)
                    incidents.append(incident)
                    self._incidents[incident.id] = incident
        return incidents

    def load_all(self, pattern: str = "*.jsonl") -> list[Incident]:
        """Load all incidents from data directory matching pattern."""
        incidents = []
        for path in self.data_dir.glob(pattern):
            incidents.extend(self.load_jsonl(path))
        return incidents

    def get_by_id(self, incident_id: str) -> Optional[Incident]:
        """Get an incident by its ID."""
        return self._incidents.get(incident_id)

    def get_all(self) -> list[Incident]:
        """Get all loaded incidents."""
        return list(self._incidents.values())

    def save_jsonl(self, incidents: list[Incident], path: Path) -> None:
        """Save incidents to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for incident in incidents:
                f.write(incident.model_dump_json() + "\n")

    def combine_sources(
        self,
        synthetic_path: Optional[Path] = None,
        github_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        deduplicate: bool = True,
    ) -> list[Incident]:
        """Combine incidents from multiple sources.

        Args:
            synthetic_path: Path to synthetic incidents JSONL
            github_path: Path to GitHub incidents JSONL
            output_path: Optional path to save combined incidents
            deduplicate: Whether to remove potential duplicates

        Returns:
            Combined list of incidents
        """
        incidents = []

        if synthetic_path and synthetic_path.exists():
            incidents.extend(self.load_jsonl(synthetic_path))

        if github_path and github_path.exists():
            incidents.extend(self.load_jsonl(github_path))

        if deduplicate:
            incidents = self._deduplicate(incidents)

        if output_path:
            self.save_jsonl(incidents, output_path)

        return incidents

    def _deduplicate(self, incidents: list[Incident]) -> list[Incident]:
        """Remove duplicate incidents based on title similarity."""
        seen_titles: dict[str, Incident] = {}
        unique = []

        for incident in incidents:
            # Simple deduplication by normalized title
            normalized = incident.title.lower().strip()
            if normalized not in seen_titles:
                seen_titles[normalized] = incident
                unique.append(incident)

        return unique

    def get_stats(self) -> dict:
        """Get statistics about loaded incidents."""
        incidents = self.get_all()

        if not incidents:
            return {"total": 0}

        from collections import Counter

        categories = Counter(i.category.value for i in incidents)
        severities = Counter(i.severity.value for i in incidents)
        sources = Counter(i.source.value for i in incidents)
        has_resolution = sum(1 for i in incidents if i.resolution)

        return {
            "total": len(incidents),
            "by_category": dict(categories),
            "by_severity": dict(severities),
            "by_source": dict(sources),
            "with_resolution": has_resolution,
            "resolution_rate": has_resolution / len(incidents) if incidents else 0,
        }
