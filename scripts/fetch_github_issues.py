#!/usr/bin/env python3
"""Fetch GitHub issues as incident data."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.github_ingestion import DEFAULT_REPOS, GitHubIngestion


def main():
    parser = argparse.ArgumentParser(description="Fetch GitHub issues as incident data")
    parser.add_argument(
        "--repos",
        nargs="+",
        default=DEFAULT_REPOS,
        help="Repositories to fetch from (format: owner/repo)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=Path("data/processed/github.jsonl"),
        help="Output path for processed incidents JSONL",
    )
    parser.add_argument(
        "--max-per-repo",
        type=int,
        default=50,
        help="Maximum issues per repository",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )

    args = parser.parse_args()

    print(f"Fetching issues from {len(args.repos)} repositories...")
    print(f"Repos: {', '.join(args.repos)}")

    ingestion = GitHubIngestion(token=args.token)

    # Fetch and convert
    incidents = ingestion.fetch_and_convert(
        repos=args.repos,
        max_per_repo=args.max_per_repo,
    )

    print(f"\nFetched {len(incidents)} incidents total")

    # Print category distribution
    from collections import Counter

    categories = Counter(i.category.value for i in incidents)
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Save processed incidents
    ingestion.save_incidents(incidents, args.processed_output)
    print(f"\nSaved processed incidents to {args.processed_output}")


if __name__ == "__main__":
    main()
