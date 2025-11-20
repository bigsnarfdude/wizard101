#!/usr/bin/env python3
"""
Evaluate Full Cascade Performance

Runs test samples through L0 → L1 → L2 → L3 cascade.
Tracks each sample's journey and records where it gets flagged.

Usage:
    python evaluate_cascade.py [--samples N] [--seed S] [--l2] [--l3]
"""

import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

from cascade import SafetyCascade, CascadeConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate cascade performance")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--l2", action="store_true", help="Enable L2 gauntlet")
    parser.add_argument("--l3", action="store_true", help="Enable L3 judge")
    args = parser.parse_args()

    # Load test data - prefer larger datasets
    test_path = Path("guardreasoner_test_10k.json")  # 10k balanced
    if not test_path.exists():
        test_path = Path("../wildguard_full_benchmark.json")  # 1.5k
    if not test_path.exists():
        test_path = Path("../combined_test.json")  # 200
    if not test_path.exists():
        print(f"Test data not found")
        return

    print(f"Using test data: {test_path}")

    with open(test_path) as f:
        test_data = json.load(f)

    # Sample
    random.seed(args.seed)
    if len(test_data) > args.samples:
        test_data = random.sample(test_data, args.samples)

    print(f"\nEvaluating {len(test_data)} samples through cascade")
    print(f"L2 enabled: {args.l2}")
    print(f"L3 enabled: {args.l3}")
    print("=" * 60)

    # Initialize cascade
    config = CascadeConfig(
        enable_l2=args.l2,
        enable_l3=args.l3,
    )
    cascade = SafetyCascade(config)

    # Run evaluation
    results = []
    layer_counts = Counter()
    correct_at_layer = defaultdict(int)
    total_at_layer = defaultdict(int)

    for i, item in enumerate(test_data):
        text = item.get("prompt", item.get("text", ""))
        expected = item.get("label", "").lower()
        if expected == "harmless":
            expected = "safe"

        # Run through cascade
        result = cascade.classify(text)

        # Record journey
        journey = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "expected": expected,
            "final_label": result.label,
            "stopped_at": result.stopped_at,
            "confidence": result.confidence,
            "latency_ms": result.total_latency_ms,
            "layers": result.layers,
        }
        results.append(journey)

        # Track stats
        layer_counts[result.stopped_at] += 1
        total_at_layer[result.stopped_at] += 1
        if result.label == expected:
            correct_at_layer[result.stopped_at] += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_data)}")

    # Calculate stats
    print("\n" + "=" * 60)
    print("CASCADE EVALUATION RESULTS")
    print("=" * 60)

    # Layer distribution
    print("\nLayer Distribution:")
    for layer in ["L0", "L1", "L2", "L3"]:
        count = layer_counts.get(layer, 0)
        pct = count / len(results) * 100 if results else 0
        print(f"  {layer}: {count} ({pct:.1f}%)")

    # Accuracy by layer
    print("\nAccuracy by Stopping Layer:")
    for layer in ["L0", "L1", "L2", "L3"]:
        total = total_at_layer.get(layer, 0)
        correct = correct_at_layer.get(layer, 0)
        if total > 0:
            acc = correct / total * 100
            print(f"  {layer}: {correct}/{total} = {acc:.1f}%")

    # Overall accuracy
    total_correct = sum(1 for r in results if r["final_label"] == r["expected"])
    overall_acc = total_correct / len(results) * 100 if results else 0
    print(f"\nOverall Cascade Accuracy: {total_correct}/{len(results)} = {overall_acc:.1f}%")

    # Confusion matrix
    tp = sum(1 for r in results if r["final_label"] == "harmful" and r["expected"] == "harmful")
    tn = sum(1 for r in results if r["final_label"] == "safe" and r["expected"] == "safe")
    fp = sum(1 for r in results if r["final_label"] == "harmful" and r["expected"] == "safe")
    fn = sum(1 for r in results if r["final_label"] == "safe" and r["expected"] == "harmful")

    print("\nConfusion Matrix:")
    print(f"  TP (harmful→harmful): {tp}")
    print(f"  TN (safe→safe): {tn}")
    print(f"  FP (safe→harmful): {fp}")
    print(f"  FN (harmful→safe): {fn}")

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMetrics:")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  F1: {f1*100:.1f}%")

    # Latency stats
    latencies = [r["latency_ms"] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0

    print(f"\nLatency:")
    print(f"  Average: {avg_latency:.1f}ms")
    print(f"  Min: {min_latency:.1f}ms")
    print(f"  Max: {max_latency:.1f}ms")

    # Save detailed results
    output_path = Path("cascade_eval_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "samples": len(results),
                "seed": args.seed,
                "l2_enabled": args.l2,
                "l3_enabled": args.l3,
            },
            "summary": {
                "accuracy": overall_acc,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1": f1 * 100,
                "avg_latency_ms": avg_latency,
                "layer_distribution": dict(layer_counts),
            },
            "confusion_matrix": {
                "tp": tp, "tn": tn, "fp": fp, "fn": fn
            },
            "journeys": results,
        }, f, indent=2)

    print(f"\nDetailed results saved to {output_path}")

    # Show example journeys
    print("\n" + "=" * 60)
    print("EXAMPLE JOURNEYS")
    print("=" * 60)

    # Show some correct and incorrect examples
    correct_examples = [r for r in results if r["final_label"] == r["expected"]][:2]
    incorrect_examples = [r for r in results if r["final_label"] != r["expected"]][:2]

    for r in correct_examples + incorrect_examples:
        status = "✓" if r["final_label"] == r["expected"] else "✗"
        print(f"\n{status} {r['text']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Got: {r['final_label']} (stopped at {r['stopped_at']})")
        print(f"  Journey: {' → '.join(l['level'] for l in r['layers'])}")
        print(f"  Latency: {r['latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
