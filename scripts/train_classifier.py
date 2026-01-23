#!/usr/bin/env python3
"""Train the incident classifier."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.model import IncidentClassifier
from src.data.loader import IncidentLoader
from src.schema import Category


def main():
    parser = argparse.ArgumentParser(description="Train the incident classifier")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incidents.jsonl"),
        help="Input JSONL file with incidents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/classifier"),
        help="Output directory for model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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

    # Filter for labeled incidents
    labeled = [i for i in incidents if i.category != Category.UNKNOWN]
    print(f"Labeled incidents: {len(labeled)}")

    if len(labeled) < 20:
        print("Error: Need at least 20 labeled incidents for training")
        return 1

    # Print class distribution
    from collections import Counter

    categories = Counter(i.category.value for i in labeled)
    print("\nClass distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Train classifier
    print("\nTraining classifier...")
    classifier = IncidentClassifier()
    metrics = classifier.train(
        incidents=labeled,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # Print metrics
    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print(f"Train size: {metrics['train_size']}")
    print(f"Test size: {metrics['test_size']}")
    print(f"\nAccuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")

    # Per-class metrics
    print("\nPer-class metrics:")
    report = metrics["classification_report"]
    for cat in sorted(categories.keys()):
        if cat in report:
            r = report[cat]
            print(f"  {cat:12} - P: {r['precision']:.2f}, R: {r['recall']:.2f}, F1: {r['f1-score']:.2f}")

    # Save model
    classifier.save(args.output)
    print(f"\nModel saved to {args.output}")

    # Save metrics
    metrics_path = args.output / "metrics.json"
    # Convert numpy types for JSON serialization
    metrics_json = {
        k: v if not hasattr(v, "tolist") else v.tolist() for k, v in metrics.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
