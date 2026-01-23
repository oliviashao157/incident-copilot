"""GitHub issues ingestion for incident data."""

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from src.classifier.categories import infer_category_from_labels, infer_category_from_text
from src.config import get_settings
from src.schema import Category, Incident, IncidentSource, Severity

# Repositories to fetch issues from (infrastructure/operations focused)
DEFAULT_REPOS = [
    "kubernetes/kubernetes",
    "prometheus/prometheus",
    "apache/airflow",
    "grafana/grafana",
    "argoproj/argo-cd",
]


class GitHubIngestion:
    """Fetch and process GitHub issues as incident-like data."""

    def __init__(self, token: Optional[str] = None):
        """Initialize with optional GitHub token."""
        self.settings = get_settings()
        self.token = token or self.settings.github_token
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "incident-copilot"

    def _make_request(self, url: str, params: Optional[dict] = None) -> dict | list:
        """Make a rate-limit-aware request to GitHub API."""
        response = self.session.get(url, params=params)

        # Handle rate limiting
        if response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            if reset_time:
                wait_time = max(reset_time - time.time(), 0) + 1
                print(f"Rate limited. Waiting {wait_time:.0f} seconds...")
                time.sleep(min(wait_time, 60))  # Cap at 60 seconds
                response = self.session.get(url, params=params)

        response.raise_for_status()
        return response.json()

    def _parse_issue_to_incident(self, issue: dict, repo: str) -> Incident:
        """Convert a GitHub issue to our Incident schema."""
        # Extract labels
        labels = [label["name"] for label in issue.get("labels", [])]

        # Infer category from labels first, then from text
        category = infer_category_from_labels(labels)
        if category == Category.UNKNOWN:
            text = f"{issue['title']} {issue.get('body', '')}"
            category, _ = infer_category_from_text(text)

        # Infer severity from labels and content
        severity = self._infer_severity(issue, labels)

        # Extract resolution from closing comment or linked PR
        resolution = None
        if issue.get("state") == "closed":
            resolution = self._extract_resolution(issue)

        # Parse timestamps
        created_at = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
        resolved_at = None
        if issue.get("closed_at"):
            resolved_at = datetime.fromisoformat(issue["closed_at"].replace("Z", "+00:00"))

        return Incident(
            id=f"gh-{repo.replace('/', '-')}-{issue['number']}",
            title=issue["title"],
            description=issue.get("body", "") or "",
            category=category,
            severity=severity,
            source=IncidentSource.GITHUB,
            created_at=created_at,
            resolved_at=resolved_at,
            resolution=resolution,
            labels=labels,
            metadata={
                "repo": repo,
                "issue_number": issue["number"],
                "url": issue["html_url"],
                "state": issue["state"],
                "author": issue["user"]["login"],
                "comments_count": issue.get("comments", 0),
            },
        )

    def _infer_severity(self, issue: dict, labels: list[str]) -> Severity:
        """Infer severity from issue labels and content."""
        labels_lower = [l.lower() for l in labels]

        # Check for explicit severity labels
        if any(s in labels_lower for s in ["critical", "priority/critical", "p0", "severity/critical"]):
            return Severity.CRITICAL
        if any(s in labels_lower for s in ["high", "priority/high", "p1", "severity/high", "important"]):
            return Severity.HIGH
        if any(s in labels_lower for s in ["medium", "priority/medium", "p2", "severity/medium"]):
            return Severity.MEDIUM
        if any(s in labels_lower for s in ["low", "priority/low", "p3", "severity/low", "minor"]):
            return Severity.LOW

        # Infer from keywords in title/body
        text = f"{issue['title']} {issue.get('body', '')}".lower()
        if any(word in text for word in ["critical", "urgent", "emergency", "outage", "down"]):
            return Severity.HIGH
        if any(word in text for word in ["important", "breaking", "regression"]):
            return Severity.MEDIUM

        return Severity.UNKNOWN

    def _extract_resolution(self, issue: dict) -> Optional[str]:
        """Extract resolution information from issue."""
        # This is a simplified version - in production you'd fetch comments
        # and look for linked PRs
        body = issue.get("body", "") or ""

        # Look for common resolution patterns
        patterns = [
            r"(?:fixed|resolved|closed) (?:by|in|via) #(\d+)",
            r"(?:root cause|solution|fix):\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                return f"Referenced: {match.group(0)}"

        return None

    def fetch_issues(
        self,
        repo: str,
        max_issues: int = 100,
        state: str = "all",
        labels: Optional[list[str]] = None,
    ) -> list[dict]:
        """Fetch issues from a GitHub repository.

        Args:
            repo: Repository in format 'owner/repo'
            max_issues: Maximum number of issues to fetch
            state: Issue state filter ('open', 'closed', 'all')
            labels: Optional labels to filter by

        Returns:
            List of raw issue dictionaries
        """
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            "state": state,
            "per_page": min(max_issues, 100),
            "sort": "updated",
            "direction": "desc",
        }
        if labels:
            params["labels"] = ",".join(labels)

        all_issues = []
        page = 1

        while len(all_issues) < max_issues:
            params["page"] = page
            issues = self._make_request(url, params)

            if not issues:
                break

            # Filter out pull requests (they have a 'pull_request' key)
            issues = [i for i in issues if "pull_request" not in i]
            all_issues.extend(issues)
            page += 1

            if len(issues) < params["per_page"]:
                break

        return all_issues[:max_issues]

    def fetch_and_convert(
        self,
        repos: Optional[list[str]] = None,
        max_per_repo: int = 50,
        incident_labels: Optional[list[str]] = None,
    ) -> list[Incident]:
        """Fetch issues from multiple repos and convert to incidents.

        Args:
            repos: List of repos to fetch from (defaults to DEFAULT_REPOS)
            max_per_repo: Max issues per repository
            incident_labels: Labels to filter for incident-like issues

        Returns:
            List of Incident objects
        """
        repos = repos or DEFAULT_REPOS

        # Labels that often indicate incident-like issues
        if incident_labels is None:
            incident_labels = ["bug", "performance", "crash", "outage", "incident"]

        incidents = []

        for repo in repos:
            print(f"Fetching issues from {repo}...")
            try:
                # Fetch issues with each label
                seen_ids = set()
                for label in incident_labels:
                    try:
                        issues = self.fetch_issues(
                            repo,
                            max_issues=max_per_repo // len(incident_labels),
                            labels=[label],
                        )
                        for issue in issues:
                            if issue["id"] not in seen_ids:
                                seen_ids.add(issue["id"])
                                incident = self._parse_issue_to_incident(issue, repo)
                                if incident.description:  # Skip empty issues
                                    incidents.append(incident)
                    except Exception as e:
                        print(f"  Warning: Failed to fetch {label} issues: {e}")

                # Also fetch some recent closed issues
                try:
                    closed_issues = self.fetch_issues(repo, max_issues=max_per_repo // 2, state="closed")
                    for issue in closed_issues:
                        if issue["id"] not in seen_ids:
                            seen_ids.add(issue["id"])
                            incident = self._parse_issue_to_incident(issue, repo)
                            if incident.description:
                                incidents.append(incident)
                except Exception as e:
                    print(f"  Warning: Failed to fetch closed issues: {e}")

                print(f"  Fetched {len([i for i in incidents if repo in i.id])} incidents from {repo}")

            except Exception as e:
                print(f"  Error fetching from {repo}: {e}")

        return incidents

    def save_raw(self, issues: list[dict], output_path: Path) -> None:
        """Save raw GitHub issues to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(issues, f, indent=2, default=str)

    def save_incidents(self, incidents: list[Incident], output_path: Path) -> None:
        """Save incidents to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for incident in incidents:
                f.write(incident.model_dump_json() + "\n")
