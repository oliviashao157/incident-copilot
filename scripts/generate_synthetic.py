#!/usr/bin/env python3
"""Generate synthetic incident data."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticGenerator
from src.schema import Category


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic incident data")
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Total number of incidents to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/synthetic.jsonl"),
        help="Output path for JSONL file",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for generation (requires API key)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[c.value for c in Category if c != Category.UNKNOWN],
        help="Categories to generate (default: all)",
    )

    args = parser.parse_args()

    # Parse categories
    categories = None
    if args.categories:
        categories = [Category(c) for c in args.categories]

    print(f"Generating {args.count} synthetic incidents...")
    print(f"Using LLM: {args.use_llm}")
    print(f"Categories: {[c.value for c in (categories or [])] or 'all'}")

    generator = SyntheticGenerator(use_llm=args.use_llm)
    incidents = generator.generate(
        count=args.count,
        categories=categories,
        seed=args.seed,
    )

    print(f"Generated {len(incidents)} incidents")

    # Save to file
    generator.save_to_jsonl(incidents, args.output)
    print(f"Saved to {args.output}")

    # Print sample
    print("\nSample incident:")
    sample = incidents[0]
    print(f"  Title: {sample.title}")
    print(f"  Category: {sample.category.value}")
    print(f"  Severity: {sample.severity.value}")
    print(f"  Description: {sample.description[:100]}...")


if __name__ == "__main__":
    main()
