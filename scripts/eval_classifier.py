#!/usr/bin/env python3
"""Evaluate the trained classifier."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.model import IncidentClassifier
from src.data.loader import IncidentLoader
from src.schema import Category


def main():
    parser = argparse.ArgumentParser(description="Evaluate the incident classifier")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/classifier"),
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incidents.jsonl"),
        help="Input JSONL file with incidents",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of sample predictions to show",
    )

    args = parser.parse_args()

    # Check paths
    if not (args.model / "model.joblib").exists():
        print(f"Error: Model not found at {args.model}")
        print("Run 'make train-classifier' first.")
        return 1

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Load model
    print(f"Loading model from {args.model}...")
    classifier = IncidentClassifier(model_path=args.model)

    # Load incidents
    print(f"Loading incidents from {args.input}...")
    loader = IncidentLoader()
    incidents = loader.load_jsonl(args.input)

    # Filter labeled incidents
    labeled = [i for i in incidents if i.category != Category.UNKNOWN]
    print(f"Evaluating on {len(labeled)} labeled incidents")

    # Evaluate
    correct = 0
    predictions = []

    for incident in labeled:
        pred_cat, confidence = classifier.predict(incident)
        is_correct = pred_cat == incident.category
        correct += int(is_correct)
        predictions.append({
            "incident": incident,
            "predicted": pred_cat,
            "confidence": confidence,
            "correct": is_correct,
        })

    accuracy = correct / len(labeled) if labeled else 0
    print(f"\nAccuracy: {accuracy:.1%} ({correct}/{len(labeled)})")

    # Show sample predictions
    print(f"\n{'=' * 60}")
    print(f"SAMPLE PREDICTIONS (showing {args.sample})")
    print("=" * 60)

    # Show some correct and incorrect predictions
    incorrect = [p for p in predictions if not p["correct"]]
    correct_preds = [p for p in predictions if p["correct"]]

    print("\nIncorrect predictions:")
    for p in incorrect[: args.sample // 2]:
        i = p["incident"]
        print(f"  Title: {i.title[:50]}...")
        print(f"    Actual: {i.category.value}")
        print(f"    Predicted: {p['predicted'].value} (conf: {p['confidence']:.2f})")
        print()

    print("\nCorrect predictions:")
    for p in correct_preds[: args.sample // 2]:
        i = p["incident"]
        print(f"  Title: {i.title[:50]}...")
        print(f"    Category: {i.category.value} (conf: {p['confidence']:.2f})")
        print()

    # Confusion analysis
    from collections import Counter

    errors = Counter()
    for p in incorrect:
        key = (p["incident"].category.value, p["predicted"].value)
        errors[key] += 1

    if errors:
        print("\nMost common errors (actual -> predicted):")
        for (actual, predicted), count in errors.most_common(5):
            print(f"  {actual} -> {predicted}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
