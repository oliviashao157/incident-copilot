"""Tests for schema models."""

from datetime import datetime

import pytest

from src.schema import (
    AnalysisResult,
    Category,
    Citation,
    Incident,
    IncidentInput,
    IncidentSource,
    Severity,
    SimilarIncident,
)


class TestIncident:
    """Tests for Incident model."""

    def test_create_minimal_incident(self):
        """Test creating incident with minimal fields."""
        incident = Incident(
            id="test-1",
            title="Test incident",
            description="This is a test",
            source=IncidentSource.MANUAL,
        )

        assert incident.id == "test-1"
        assert incident.title == "Test incident"
        assert incident.category == Category.UNKNOWN
        assert incident.severity == Severity.UNKNOWN
        assert incident.resolution is None

    def test_create_full_incident(self):
        """Test creating incident with all fields."""
        now = datetime.now()
        incident = Incident(
            id="test-2",
            title="Database outage",
            description="Primary database unreachable",
            category=Category.OUTAGE,
            severity=Severity.CRITICAL,
            source=IncidentSource.SYNTHETIC,
            created_at=now,
            resolved_at=now,
            resolution="Restarted database server",
            labels=["database", "critical"],
            metadata={"region": "us-east-1"},
        )

        assert incident.category == Category.OUTAGE
        assert incident.severity == Severity.CRITICAL
        assert incident.resolution == "Restarted database server"
        assert "database" in incident.labels

    def test_to_embedding_text(self):
        """Test embedding text generation."""
        incident = Incident(
            id="test-3",
            title="High latency",
            description="API response times elevated",
            source=IncidentSource.MANUAL,
            resolution="Added caching",
        )

        text = incident.to_embedding_text()
        assert "High latency" in text
        assert "API response times" in text
        assert "caching" in text

    def test_incident_serialization(self):
        """Test JSON serialization."""
        incident = Incident(
            id="test-4",
            title="Test",
            description="Description",
            source=IncidentSource.GITHUB,
        )

        json_str = incident.model_dump_json()
        assert "test-4" in json_str
        assert "github" in json_str


class TestIncidentInput:
    """Tests for IncidentInput model."""

    def test_valid_input(self):
        """Test valid input creation."""
        input_data = IncidentInput(
            title="API errors",
            description="Seeing 500 errors on checkout endpoint",
        )

        assert input_data.title == "API errors"

    def test_empty_title_rejected(self):
        """Test that empty title is rejected."""
        with pytest.raises(ValueError):
            IncidentInput(title="", description="Some description")

    def test_title_max_length(self):
        """Test title max length validation."""
        with pytest.raises(ValueError):
            IncidentInput(title="x" * 501, description="Description")


class TestSimilarIncident:
    """Tests for SimilarIncident model."""

    def test_similar_incident(self):
        """Test SimilarIncident creation."""
        incident = Incident(
            id="sim-1",
            title="Similar issue",
            description="Description",
            source=IncidentSource.MANUAL,
        )

        similar = SimilarIncident(incident=incident, similarity_score=0.85)

        assert similar.similarity_score == 0.85
        assert similar.incident.id == "sim-1"

    def test_similarity_score_bounds(self):
        """Test similarity score validation."""
        incident = Incident(
            id="sim-2",
            title="Test",
            description="Test",
            source=IncidentSource.MANUAL,
        )

        # Valid scores
        SimilarIncident(incident=incident, similarity_score=0.0)
        SimilarIncident(incident=incident, similarity_score=1.0)
        SimilarIncident(incident=incident, similarity_score=0.5)

        # Invalid scores
        with pytest.raises(ValueError):
            SimilarIncident(incident=incident, similarity_score=-0.1)

        with pytest.raises(ValueError):
            SimilarIncident(incident=incident, similarity_score=1.1)


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_analysis_result(self):
        """Test AnalysisResult creation."""
        result = AnalysisResult(
            query_title="Test incident",
            query_description="Test description",
            predicted_category=Category.LATENCY,
            predicted_severity=Severity.MEDIUM,
            category_confidence=0.85,
            similar_incidents=[],
            root_cause_hypothesis="Database connection pool exhausted",
            recommended_actions=["Scale database", "Add caching"],
            estimated_impact="Medium - 10% of users affected",
            citations=[],
            analysis_summary="Latency spike due to database issues",
        )

        assert result.predicted_category == Category.LATENCY
        assert len(result.recommended_actions) == 2
        assert result.category_confidence == 0.85


class TestEnums:
    """Tests for enum values."""

    def test_category_values(self):
        """Test all category values exist."""
        expected = {
            "latency", "outage", "deployment", "config",
            "capacity", "data", "security", "dependency",
            "network", "unknown",
        }
        actual = {c.value for c in Category}
        assert actual == expected

    def test_severity_values(self):
        """Test all severity values exist."""
        expected = {"critical", "high", "medium", "low", "unknown"}
        actual = {s.value for s in Severity}
        assert actual == expected

    def test_source_values(self):
        """Test all source values exist."""
        expected = {"synthetic", "github", "kaggle", "manual"}
        actual = {s.value for s in IncidentSource}
        assert actual == expected
