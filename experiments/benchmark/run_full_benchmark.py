#!/usr/bin/env python3
"""
Run comprehensive benchmark evaluation on Safety Cascade.

Evaluates across multiple datasets and reports per-dataset metrics.

Usage:
    python run_full_benchmark.py [--quick]
"""

import gc
import json
import time
import argparse
import torch
from pathlib import Path
from collections import Counter, defaultdict

# Add cascade to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "cascade"))


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_all_benchmarks(quick=False):
    """Load all benchmark datasets."""
    benchmarks = {}
    benchmark_dir = Path(__file__).parent

    # Standard benchmarks
    files = [
        ("toxicchat", "toxicchat_test.json"),
        ("harmbench", "harmbench_test.json"),
        ("openai_mod", "openai_moderation.json"),
        ("beavertails", "beavertails_30k.json"),
        ("simplesafety", "simplesafetytests.json"),
    ]

    # Add existing datasets
    experiments_dir = benchmark_dir.parent
    cascade_dir = experiments_dir / "cascade"

    files.extend([
        ("wildguard", str(experiments_dir / "wildguard_full_benchmark.json")),
        ("heretic", str(experiments_dir / "combined_test.json")),
        ("guardreasoner", str(cascade_dir / "guardreasoner_test_10k.json")),
    ])

    for name, filename in files:
        if isinstance(filename, str) and filename.startswith("/"):
            filepath = Path(filename)
        else:
            filepath = benchmark_dir / filename

        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)

                # Normalize labels
                for item in data:
                    label = item.get("label", "").lower()
                    if label == "harmless":
                        item["label"] = "safe"

                    # Ensure text field exists
                    if "text" not in item:
                        item["text"] = item.get("prompt", "")

                if quick and len(data) > 500:
                    import random
                    random.seed(42)
                    data = random.sample(data, 500)

                benchmarks[name] = data
                print(f"  Loaded {name}: {len(data)} samples")
            except Exception as e:
                print(f"  Error loading {name}: {e}")
        else:
            print(f"  Not found: {name} ({filepath})")

    return benchmarks


def run_l0_batch(samples, threshold=0.7):
    """Run L0 on samples."""
    from l0_bouncer import L0Bouncer
    l0 = L0Bouncer()

    results = []
    for sample in samples:
        text = sample.get("text", "")
        result = l0.classify(text)

        results.append({
            "text": text,
            "expected": sample.get("label", ""),
            "predicted": result["label"],
            "confidence": result["confidence"],
            "stopped_at": "L0" if result["confidence"] >= threshold else "escalate",
            "source": sample.get("source", "unknown"),
        })

    del l0
    clear_gpu()
    return results


def calculate_metrics(results):
    """Calculate evaluation metrics."""
    tp = sum(1 for r in results if r["predicted"] == "harmful" and r["expected"] == "harmful")
    tn = sum(1 for r in results if r["predicted"] == "safe" and r["expected"] == "safe")
    fp = sum(1 for r in results if r["predicted"] == "harmful" and r["expected"] == "safe")
    fn = sum(1 for r in results if r["predicted"] == "safe" and r["expected"] == "harmful")

    accuracy = (tp + tn) / len(results) * 100 if results else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick mode (500 samples per dataset)")
    parser.add_argument("--l0-only", action="store_true", help="Only test L0 (no escalation)")
    args = parser.parse_args()

    print("="*60)
    print("COMPREHENSIVE SAFETY CASCADE BENCHMARK")
    print("="*60)

    # Load benchmarks
    print("\nLoading benchmarks...")
    benchmarks = load_all_benchmarks(quick=args.quick)

    if not benchmarks:
        print("No benchmarks loaded!")
        return

    total_samples = sum(len(data) for data in benchmarks.values())
    print(f"\nTotal samples: {total_samples}")

    # Evaluate each benchmark
    all_results = {}
    start_time = time.time()

    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)

    for name, data in benchmarks.items():
        print(f"\nEvaluating {name} ({len(data)} samples)...")
        results = run_l0_batch(data)
        all_results[name] = results

        metrics = calculate_metrics(results)
        print(f"  Accuracy: {metrics['accuracy']:.1f}%")
        print(f"  Recall: {metrics['recall']:.1f}%")
        print(f"  F1: {metrics['f1']:.1f}%")

    total_time = time.time() - start_time

    # Aggregate results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)

    summary = {}
    for name, results in all_results.items():
        metrics = calculate_metrics(results)
        summary[name] = metrics

        print(f"\n{name.upper()}:")
        print(f"  Samples: {metrics['total']}")
        print(f"  Accuracy: {metrics['accuracy']:.1f}%")
        print(f"  Precision: {metrics['precision']:.1f}%")
        print(f"  Recall: {metrics['recall']:.1f}%")
        print(f"  F1: {metrics['f1']:.1f}%")
        print(f"  FN: {metrics['fn']} | FP: {metrics['fp']}")

    # Overall metrics
    all_flat = []
    for results in all_results.values():
        all_flat.extend(results)

    overall = calculate_metrics(all_flat)

    print("\n" + "="*60)
    print("OVERALL PERFORMANCE")
    print("="*60)
    print(f"\nTotal Samples: {overall['total']}")
    print(f"Overall Accuracy: {overall['accuracy']:.1f}%")
    print(f"Overall Precision: {overall['precision']:.1f}%")
    print(f"Overall Recall: {overall['recall']:.1f}%")
    print(f"Overall F1: {overall['f1']:.1f}%")
    print(f"\nTotal Time: {total_time:.1f}s ({total_time/len(all_flat)*1000:.1f}ms/sample)")

    # Save results
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "quick" if args.quick else "full",
            "total_samples": overall["total"],
            "total_time_s": total_time,
            "overall": overall,
            "per_dataset": summary,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Highlight concerns
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    low_recall = [(n, m) for n, m in summary.items() if m["recall"] < 90]
    if low_recall:
        print("\nDatasets with recall < 90%:")
        for name, metrics in low_recall:
            print(f"  {name}: {metrics['recall']:.1f}% ({metrics['fn']} missed)")

    high_fp = [(n, m) for n, m in summary.items() if m["fp"] > m["total"] * 0.1]
    if high_fp:
        print("\nDatasets with >10% false positives:")
        for name, metrics in high_fp:
            print(f"  {name}: {metrics['fp']} FP ({metrics['fp']/metrics['total']*100:.1f}%)")


if __name__ == "__main__":
    main()
