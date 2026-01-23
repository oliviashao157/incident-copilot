#!/usr/bin/env python3
"""Evaluate RAG pipeline against golden test cases."""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.schema import IncidentInput


def load_golden_cases(path: Path) -> list[dict]:
    """Load golden test cases from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["cases"]


def evaluate_classification(result, expected: dict) -> dict:
    """Evaluate classification accuracy."""
    category_match = result.predicted_category.value == expected["expected_category"]
    severity_match = result.predicted_severity.value == expected["expected_severity"]

    return {
        "category_correct": category_match,
        "severity_correct": severity_match,
        "predicted_category": result.predicted_category.value,
        "expected_category": expected["expected_category"],
        "predicted_severity": result.predicted_severity.value,
        "expected_severity": expected["expected_severity"],
        "category_confidence": result.category_confidence,
    }


def evaluate_retrieval(result, expected: dict) -> dict:
    """Evaluate retrieval quality."""
    has_similar = len(result.similar_incidents) > 0

    # Check if citations are present when expected
    has_citations = len(result.citations) > 0
    citation_match = has_citations if expected.get("should_cite_similar") else True

    return {
        "retrieved_count": len(result.similar_incidents),
        "has_similar_incidents": has_similar,
        "citation_count": len(result.citations),
        "has_required_citations": citation_match,
    }


def evaluate_analysis(result, expected: dict) -> dict:
    """Evaluate analysis quality."""
    # Check if expected keywords appear in analysis
    analysis_text = (
        f"{result.root_cause_hypothesis} "
        f"{' '.join(result.recommended_actions)} "
        f"{result.analysis_summary}"
    ).lower()

    expected_keywords = expected.get("expected_keywords", [])
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in analysis_text)
    keyword_coverage = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

    return {
        "has_root_cause": bool(result.root_cause_hypothesis),
        "has_actions": len(result.recommended_actions) > 0,
        "action_count": len(result.recommended_actions),
        "has_summary": bool(result.analysis_summary),
        "keyword_coverage": keyword_coverage,
        "keywords_found": keyword_hits,
        "keywords_expected": len(expected_keywords),
    }


def run_evaluation(pipeline: RAGPipeline, cases: list[dict]) -> dict:
    """Run evaluation on all test cases."""
    results = []
    total_latency = 0

    for i, case in enumerate(cases):
        print(f"Evaluating case {i+1}/{len(cases)}: {case['id']}...")

        input_data = IncidentInput(
            title=case["title"],
            description=case["description"],
        )

        start_time = time.time()
        try:
            result = pipeline.analyze(input_data)
            latency = time.time() - start_time
            total_latency += latency

            eval_result = {
                "case_id": case["id"],
                "title": case["title"],
                "latency_seconds": latency,
                "success": True,
                "classification": evaluate_classification(result, case),
                "retrieval": evaluate_retrieval(result, case),
                "analysis": evaluate_analysis(result, case),
            }

        except Exception as e:
            eval_result = {
                "case_id": case["id"],
                "title": case["title"],
                "success": False,
                "error": str(e),
            }

        results.append(eval_result)

    # Calculate aggregate metrics
    successful = [r for r in results if r["success"]]

    if successful:
        metrics = {
            "total_cases": len(cases),
            "successful_cases": len(successful),
            "success_rate": len(successful) / len(cases),
            "avg_latency_seconds": total_latency / len(successful),
            "classification": {
                "category_accuracy": sum(
                    1 for r in successful if r["classification"]["category_correct"]
                ) / len(successful),
                "severity_accuracy": sum(
                    1 for r in successful if r["classification"]["severity_correct"]
                ) / len(successful),
            },
            "retrieval": {
                "avg_retrieved": sum(r["retrieval"]["retrieved_count"] for r in successful)
                / len(successful),
                "citation_rate": sum(
                    1 for r in successful if r["retrieval"]["has_required_citations"]
                ) / len(successful),
            },
            "analysis": {
                "root_cause_rate": sum(
                    1 for r in successful if r["analysis"]["has_root_cause"]
                ) / len(successful),
                "avg_actions": sum(r["analysis"]["action_count"] for r in successful)
                / len(successful),
                "avg_keyword_coverage": sum(
                    r["analysis"]["keyword_coverage"] for r in successful
                ) / len(successful),
            },
        }
    else:
        metrics = {
            "total_cases": len(cases),
            "successful_cases": 0,
            "success_rate": 0,
            "error": "All evaluations failed",
        }

    return {
        "metrics": metrics,
        "results": results,
    }


def generate_report(evaluation: dict, output_path: Path) -> None:
    """Generate markdown evaluation report."""
    metrics = evaluation["metrics"]
    results = evaluation["results"]

    report = f"""# RAG Pipeline Evaluation Report

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Test Cases | {metrics['total_cases']} |
| Successful | {metrics['successful_cases']} |
| Success Rate | {metrics.get('success_rate', 0):.1%} |
| Avg Latency | {metrics.get('avg_latency_seconds', 0):.2f}s |

## Classification Performance

| Metric | Value |
|--------|-------|
| Category Accuracy | {metrics.get('classification', {}).get('category_accuracy', 0):.1%} |
| Severity Accuracy | {metrics.get('classification', {}).get('severity_accuracy', 0):.1%} |

## Retrieval Performance

| Metric | Value |
|--------|-------|
| Avg Retrieved | {metrics.get('retrieval', {}).get('avg_retrieved', 0):.1f} |
| Citation Rate | {metrics.get('retrieval', {}).get('citation_rate', 0):.1%} |

## Analysis Quality

| Metric | Value |
|--------|-------|
| Root Cause Rate | {metrics.get('analysis', {}).get('root_cause_rate', 0):.1%} |
| Avg Actions | {metrics.get('analysis', {}).get('avg_actions', 0):.1f} |
| Keyword Coverage | {metrics.get('analysis', {}).get('avg_keyword_coverage', 0):.1%} |

## Detailed Results

"""

    # Add detailed results for each case
    for r in results:
        status = "✅" if r["success"] else "❌"
        report += f"### {status} {r['case_id']}: {r['title'][:50]}...\n\n"

        if r["success"]:
            c = r["classification"]
            report += f"- **Category**: {c['predicted_category']} "
            report += f"({'✓' if c['category_correct'] else '✗ expected: ' + c['expected_category']})\n"
            report += f"- **Severity**: {c['predicted_severity']} "
            report += f"({'✓' if c['severity_correct'] else '✗ expected: ' + c['expected_severity']})\n"
            report += f"- **Confidence**: {c['category_confidence']:.2f}\n"
            report += f"- **Similar Retrieved**: {r['retrieval']['retrieved_count']}\n"
            report += f"- **Actions Suggested**: {r['analysis']['action_count']}\n"
            report += f"- **Latency**: {r['latency_seconds']:.2f}s\n"
        else:
            report += f"- **Error**: {r.get('error', 'Unknown error')}\n"

        report += "\n"

    # Save report
    output_path.write_text(report)
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("eval/golden_cases.json"),
        help="Path to golden test cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/report.md"),
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON output for detailed results",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Maximum number of cases to evaluate",
    )

    args = parser.parse_args()

    if not args.golden.exists():
        print(f"Error: Golden cases file not found: {args.golden}")
        return 1

    # Load test cases
    cases = load_golden_cases(args.golden)
    if args.max_cases:
        cases = cases[: args.max_cases]

    print(f"Loaded {len(cases)} test cases")

    # Initialize pipeline
    print("Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Make sure to run 'make build-index' and 'make train-classifier' first.")
        return 1

    # Run evaluation
    print("\nRunning evaluation...")
    evaluation = run_evaluation(pipeline, cases)

    # Generate report
    generate_report(evaluation, args.output)

    # Optionally save JSON results
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(evaluation, f, indent=2, default=str)
        print(f"JSON results saved to {args.json_output}")

    # Print summary
    metrics = evaluation["metrics"]
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Success Rate: {metrics.get('success_rate', 0):.1%}")
    print(f"Category Accuracy: {metrics.get('classification', {}).get('category_accuracy', 0):.1%}")
    print(f"Avg Latency: {metrics.get('avg_latency_seconds', 0):.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
