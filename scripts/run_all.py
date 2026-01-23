#!/usr/bin/env python3
"""Run the complete pipeline: generate data, build index, train classifier."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def main():
    steps = [
        (
            ["python", "scripts/generate_synthetic.py", "--count", "100"],
            "Generate synthetic incident data",
        ),
        (
            ["python", "scripts/process_data.py"],
            "Process and combine data",
        ),
        (
            ["python", "scripts/build_index.py"],
            "Build FAISS index",
        ),
        (
            ["python", "scripts/train_classifier.py"],
            "Train classifier",
        ),
    ]

    print("=" * 60)
    print("INCIDENT COPILOT - FULL PIPELINE")
    print("=" * 60)

    failed_steps = []
    for cmd, description in steps:
        if not run_command(cmd, description):
            print(f"\n❌ FAILED: {description}")
            failed_steps.append(description)
        else:
            print(f"\n✅ COMPLETED: {description}")

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    if failed_steps:
        print(f"\n❌ {len(failed_steps)} step(s) failed:")
        for step in failed_steps:
            print(f"  - {step}")
        return 1
    else:
        print("\n✅ All steps completed successfully!")
        print("\nNext steps:")
        print("  1. Start the API: make run-api")
        print("  2. Start the UI: make run-ui")
        print("  3. Open http://localhost:8501 in your browser")
        return 0


if __name__ == "__main__":
    sys.exit(main())
