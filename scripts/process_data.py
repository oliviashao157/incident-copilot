#!/usr/bin/env python3
"""Combine and process incident data from multiple sources."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import IncidentLoader


def main():
    parser = argparse.ArgumentParser(description="Process and combine incident data")
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=Path("data/processed/synthetic.jsonl"),
        help="Path to synthetic incidents JSONL",
    )
    parser.add_argument(
        "--github",
        type=Path,
        default=Path("data/processed/github.jsonl"),
        help="Path to GitHub incidents JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/incidents.jsonl"),
        help="Output path for combined incidents",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication",
    )

    args = parser.parse_args()

    loader = IncidentLoader()

    print("Loading data sources...")

    # Check which sources exist
    sources = []
    if args.synthetic.exists():
        sources.append(("synthetic", args.synthetic))
        print(f"  Found synthetic data: {args.synthetic}")
    else:
        print(f"  No synthetic data at {args.synthetic}")

    if args.github.exists():
        sources.append(("github", args.github))
        print(f"  Found GitHub data: {args.github}")
    else:
        print(f"  No GitHub data at {args.github}")

    if not sources:
        print("\nNo data sources found. Run:")
        print("  make generate-synthetic")
        print("  make fetch-github")
        return 1

    # Combine sources
    incidents = loader.combine_sources(
        synthetic_path=args.synthetic if args.synthetic.exists() else None,
        github_path=args.github if args.github.exists() else None,
        output_path=args.output,
        deduplicate=not args.no_dedupe,
    )

    print(f"\nCombined {len(incidents)} incidents")

    # Print statistics
    stats = loader.get_stats()
    print("\nData statistics:")
    print(f"  Total incidents: {stats['total']}")
    print(f"  With resolution: {stats['with_resolution']} ({stats['resolution_rate']:.0%})")

    print("\n  By category:")
    for cat, count in sorted(stats.get("by_category", {}).items()):
        print(f"    {cat}: {count}")

    print("\n  By severity:")
    for sev, count in sorted(stats.get("by_severity", {}).items()):
        print(f"    {sev}: {count}")

    print("\n  By source:")
    for src, count in sorted(stats.get("by_source", {}).items()):
        print(f"    {src}: {count}")

    print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
