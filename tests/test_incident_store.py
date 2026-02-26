"""Tests for IncidentStore with metadata filtering."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.data.incident_store import IncidentStore
from src.schema import Category, Incident, IncidentSource, Severity


@pytest.fixture
def sample_incidents():
    """Create sample incidents with varied metadata."""
    base_time = datetime(2024, 6, 1, 12, 0, 0)
    return [
        Incident(
            id="inc-1",
            title="High latency on API",
            description="Response times increased",
            category=Category.LATENCY,
            severity=Severity.HIGH,
            source=IncidentSource.SYNTHETIC,
            created_at=base_time,
        ),
        Incident(
            id="inc-2",
            title="Service outage",
            description="Complete outage of payment service",
            category=Category.OUTAGE,
            severity=Severity.CRITICAL,
            source=IncidentSource.SYNTHETIC,
            created_at=base_time + timedelta(days=1),
        ),
        Incident(
            id="inc-3",
            title="Deployment failure",
            description="Canary deployment failed",
            category=Category.DEPLOYMENT,
            severity=Severity.MEDIUM,
            source=IncidentSource.SYNTHETIC,
            created_at=base_time + timedelta(days=2),
        ),
        Incident(
            id="inc-4",
            title="Database latency",
            description="Query latency spiked",
            category=Category.LATENCY,
            severity=Severity.MEDIUM,
            source=IncidentSource.SYNTHETIC,
            created_at=base_time + timedelta(days=3),
        ),
        Incident(
            id="inc-5",
            title="Security alert",
            description="Unauthorized access attempt",
            category=Category.SECURITY,
            severity=Severity.CRITICAL,
            source=IncidentSource.SYNTHETIC,
            created_at=base_time + timedelta(days=4),
        ),
    ]


@pytest.fixture
def store_with_data(sample_incidents, tmp_path):
    """Create a store with sample data loaded from JSONL."""
    # Write incidents to a JSONL file
    jsonl_path = tmp_path / "incidents.jsonl"
    with open(jsonl_path, "w") as f:
        for inc in sample_incidents:
            f.write(inc.model_dump_json() + "\n")

    store = IncidentStore(data_dir=tmp_path)
    store.load_all()
    return store


class TestIncidentStoreLoading:
    """Tests for loading and indexing."""

    def test_load_all_returns_incidents(self, store_with_data, sample_incidents):
        all_incidents = store_with_data.get_all()
        assert len(all_incidents) == len(sample_incidents)

    def test_get_by_id(self, store_with_data):
        inc = store_with_data.get_by_id("inc-1")
        assert inc is not None
        assert inc.title == "High latency on API"

    def test_get_by_id_missing(self, store_with_data):
        assert store_with_data.get_by_id("nonexistent") is None

    def test_indices_built(self, store_with_data):
        assert store_with_data._indexed is True
        assert len(store_with_data._by_category) > 0
        assert len(store_with_data._by_severity) > 0
        assert len(store_with_data._by_date) > 0


class TestFilterByCategory:
    """Tests for category-based filtering."""

    def test_single_category(self, store_with_data):
        ids = store_with_data.filter_ids(categories=[Category.LATENCY])
        assert ids == {"inc-1", "inc-4"}

    def test_multiple_categories(self, store_with_data):
        ids = store_with_data.filter_ids(categories=[Category.LATENCY, Category.OUTAGE])
        assert ids == {"inc-1", "inc-2", "inc-4"}

    def test_no_match_category(self, store_with_data):
        ids = store_with_data.filter_ids(categories=[Category.NETWORK])
        assert ids == set()


class TestFilterBySeverity:
    """Tests for severity-based filtering."""

    def test_single_severity(self, store_with_data):
        ids = store_with_data.filter_ids(severities=[Severity.CRITICAL])
        assert ids == {"inc-2", "inc-5"}

    def test_multiple_severities(self, store_with_data):
        ids = store_with_data.filter_ids(severities=[Severity.CRITICAL, Severity.HIGH])
        assert ids == {"inc-1", "inc-2", "inc-5"}


class TestFilterByDate:
    """Tests for date-based filtering."""

    def test_after_filter(self, store_with_data):
        after = datetime(2024, 6, 3, 0, 0, 0)
        ids = store_with_data.filter_ids(after=after)
        # inc-3 (June 3 12:00), inc-4 (June 4 12:00), inc-5 (June 5 12:00)
        assert ids == {"inc-3", "inc-4", "inc-5"}

    def test_before_filter(self, store_with_data):
        before = datetime(2024, 6, 2, 0, 0, 0)
        ids = store_with_data.filter_ids(before=before)
        # inc-1 (June 1 12:00)
        assert ids == {"inc-1"}

    def test_date_range(self, store_with_data):
        after = datetime(2024, 6, 2, 0, 0, 0)
        before = datetime(2024, 6, 4, 0, 0, 0)
        ids = store_with_data.filter_ids(after=after, before=before)
        # inc-2 (June 2 12:00), inc-3 (June 3 12:00)
        assert ids == {"inc-2", "inc-3"}


class TestCombinedFilters:
    """Tests for AND logic across multiple filter types."""

    def test_category_and_severity(self, store_with_data):
        ids = store_with_data.filter_ids(
            categories=[Category.LATENCY],
            severities=[Severity.MEDIUM],
        )
        # inc-4 is LATENCY + MEDIUM
        assert ids == {"inc-4"}

    def test_category_and_date(self, store_with_data):
        ids = store_with_data.filter_ids(
            categories=[Category.LATENCY],
            after=datetime(2024, 6, 3, 0, 0, 0),
        )
        # inc-4 is LATENCY + after June 3
        assert ids == {"inc-4"}

    def test_no_filters_returns_all(self, store_with_data, sample_incidents):
        ids = store_with_data.filter_ids()
        assert len(ids) == len(sample_incidents)

    def test_all_filters_no_match(self, store_with_data):
        ids = store_with_data.filter_ids(
            categories=[Category.SECURITY],
            severities=[Severity.LOW],
        )
        assert ids == set()


class TestGetStats:
    """Tests for stats reporting."""

    def test_stats(self, store_with_data):
        stats = store_with_data.get_stats()
        assert stats["total"] == 5
        assert "by_category" in stats
        assert "by_severity" in stats
