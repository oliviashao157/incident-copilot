#!/usr/bin/env python3
"""Build FAISS index from incident data."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import IncidentLoader
from src.embedding.index import FAISSIndex


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from incident data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incidents.jsonl"),
        help="Input JSONL file with incidents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/index"),
        help="Output directory for index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run 'make process-data' first to create the incidents file.")
        return 1

    # Load incidents
    print(f"Loading incidents from {args.input}...")
    loader = IncidentLoader()
    incidents = loader.load_jsonl(args.input)
    print(f"Loaded {len(incidents)} incidents")

    if not incidents:
        print("Error: No incidents to index")
        return 1

    # Build index
    print(f"\nBuilding FAISS index with {args.model}...")
    index = FAISSIndex()
    index.build(incidents)
    print(f"Built index with {index.size} vectors")
    print(f"Embedding dimension: {index.embedder.dimension}")

    # Save index
    index.save(args.output)
    print(f"\nSaved index to {args.output}")

    # Test search
    print("\nTesting search with sample query...")
    test_query = "high latency on API causing timeouts"
    results = index.search(test_query, k=3)
    print(f"Query: '{test_query}'")
    print("Top 3 results:")
    for incident_id, score in results:
        incident = loader.get_by_id(incident_id)
        if incident:
            print(f"  [{score:.3f}] {incident.title[:60]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
