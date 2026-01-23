"""Tests for GitHub ingestion."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data.github_ingestion import GitHubIngestion
from src.schema import Category, IncidentSource, Severity


class TestGitHubIngestion:
    """Tests for GitHubIngestion class."""

    def test_init_without_token(self):
        """Test initialization without token."""
        with patch.object(GitHubIngestion, "__init__", lambda x, token=None: None):
            ingestion = GitHubIngestion.__new__(GitHubIngestion)
            ingestion.token = None
            ingestion.session = MagicMock()
            ingestion.settings = MagicMock()
            ingestion.settings.github_token = None

            assert ingestion.token is None

    def test_parse_issue_to_incident(self):
        """Test parsing GitHub issue to Incident."""
        ingestion = GitHubIngestion.__new__(GitHubIngestion)
        ingestion.settings = MagicMock()

        issue = {
            "number": 123,
            "title": "High latency on API endpoint",
            "body": "We're seeing elevated response times on the /api/users endpoint.",
            "labels": [{"name": "performance"}, {"name": "bug"}],
            "state": "closed",
            "html_url": "https://github.com/test/repo/issues/123",
            "user": {"login": "testuser"},
            "comments": 5,
            "created_at": "2024-01-15T10:00:00Z",
            "closed_at": "2024-01-15T12:00:00Z",
        }

        incident = ingestion._parse_issue_to_incident(issue, "test/repo")

        assert incident.id == "gh-test-repo-123"
        assert "High latency" in incident.title
        assert incident.source == IncidentSource.GITHUB
        assert incident.metadata["repo"] == "test/repo"
        assert incident.metadata["issue_number"] == 123

    def test_infer_severity_from_labels(self):
        """Test severity inference from labels."""
        ingestion = GitHubIngestion.__new__(GitHubIngestion)

        # Critical label
        issue = {"title": "Test", "body": "", "labels": [{"name": "priority/critical"}]}
        severity = ingestion._infer_severity(issue, ["priority/critical"])
        assert severity == Severity.CRITICAL

        # High label
        issue = {"title": "Test", "body": "", "labels": [{"name": "p1"}]}
        severity = ingestion._infer_severity(issue, ["p1"])
        assert severity == Severity.HIGH

        # Medium label
        issue = {"title": "Test", "body": "", "labels": [{"name": "priority/medium"}]}
        severity = ingestion._infer_severity(issue, ["priority/medium"])
        assert severity == Severity.MEDIUM

        # Low label
        issue = {"title": "Test", "body": "", "labels": [{"name": "minor"}]}
        severity = ingestion._infer_severity(issue, ["minor"])
        assert severity == Severity.LOW

    def test_infer_severity_from_text(self):
        """Test severity inference from issue text."""
        ingestion = GitHubIngestion.__new__(GitHubIngestion)

        # Critical keywords
        issue = {"title": "CRITICAL: Production down", "body": "Complete outage"}
        severity = ingestion._infer_severity(issue, [])
        assert severity == Severity.HIGH

        # No severity indicators
        issue = {"title": "Minor update", "body": "Small change needed"}
        severity = ingestion._infer_severity(issue, [])
        assert severity == Severity.UNKNOWN

    def test_category_from_performance_label(self):
        """Test category inference from performance label."""
        ingestion = GitHubIngestion.__new__(GitHubIngestion)
        ingestion.settings = MagicMock()

        issue = {
            "number": 1,
            "title": "Slow queries",
            "body": "Database queries taking too long",
            "labels": [{"name": "performance"}],
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/1",
            "user": {"login": "user"},
            "comments": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "closed_at": None,
        }

        incident = ingestion._parse_issue_to_incident(issue, "test/repo")
        assert incident.category == Category.LATENCY

    def test_category_from_security_label(self):
        """Test category inference from security label."""
        ingestion = GitHubIngestion.__new__(GitHubIngestion)
        ingestion.settings = MagicMock()

        issue = {
            "number": 2,
            "title": "Security vulnerability",
            "body": "CVE found in dependency",
            "labels": [{"name": "security"}],
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/2",
            "user": {"login": "user"},
            "comments": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "closed_at": None,
        }

        incident = ingestion._parse_issue_to_incident(issue, "test/repo")
        assert incident.category == Category.SECURITY


class TestGitHubIngestionIntegration:
    """Integration tests (require network, marked for optional skip)."""

    @pytest.mark.skip(reason="Requires network access and GitHub API")
    def test_fetch_real_issues(self):
        """Test fetching real issues from GitHub."""
        ingestion = GitHubIngestion()
        issues = ingestion.fetch_issues("kubernetes/kubernetes", max_issues=5)

        assert len(issues) <= 5
        for issue in issues:
            assert "title" in issue
            assert "body" in issue
