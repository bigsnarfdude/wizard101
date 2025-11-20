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
    """Run L0 on samples, return caught and uncertain."""
    from l0_bouncer import L0Bouncer
    l0 = L0Bouncer()

    caught = []
    uncertain = []

    for idx, sample in enumerate(samples):
        text = sample.get("text", "")
        result = l0.classify(text)

        if result["confidence"] >= threshold:
            caught.append((idx, {
                "text": text,
                "expected": sample.get("label", ""),
                "predicted": result["label"],
                "confidence": result["confidence"],
                "stopped_at": "L0",
            }))
        else:
            uncertain.append((idx, sample, result))

    del l0
    clear_gpu()
    return caught, uncertain


def run_l1_batch(uncertain_samples, threshold=0.7):
    """Run L1 on uncertain samples."""
    if not uncertain_samples:
        return [], []

    from l1_analyst import L1Analyst
    l1 = L1Analyst()

    caught = []
    still_uncertain = []

    for idx, sample, l0_result in uncertain_samples:
        text = sample.get("text", "")
        result = l1.analyze(text)

        if result["confidence"] >= threshold:
            caught.append((idx, {
                "text": text,
                "expected": sample.get("label", ""),
                "predicted": result["label"],
                "confidence": result["confidence"],
                "stopped_at": "L1",
            }))
        else:
            still_uncertain.append((idx, sample, result))

    del l1
    clear_gpu()
    return caught, still_uncertain


def run_l2_batch(uncertain_samples):
    """Run L2 on remaining samples."""
    if not uncertain_samples:
        return []

    from l2_gauntlet import L2Gauntlet
    l2 = L2Gauntlet()

    results = []

    for idx, sample, l1_result in uncertain_samples:
        text = sample.get("text", "")
        result = l2.analyze(text)

        results.append((idx, {
            "text": text,
            "expected": sample.get("label", ""),
            "predicted": result["label"],
            "confidence": 0.9 if result["consensus"] else 0.7,
            "stopped_at": "L2",
        }))

    del l2
    return results


def run_cascade_batch(samples, l0_threshold=0.7, l1_threshold=0.7, enable_l2=True):
    """Run full cascade on samples."""
    # L0
    l0_caught, l0_uncertain = run_l0_batch(samples, l0_threshold)

    # L1
    l1_caught, l1_uncertain = run_l1_batch(l0_uncertain, l1_threshold)

    # L2
    if enable_l2 and l1_uncertain:
        l2_results = run_l2_batch(l1_uncertain)
    else:
        # If L2 disabled, use L1's prediction
        l2_results = []
        for idx, sample, l1_result in l1_uncertain:
            l2_results.append((idx, {
                "text": sample.get("text", ""),
                "expected": sample.get("label", ""),
                "predicted": l1_result["label"],
                "confidence": l1_result["confidence"],
                "stopped_at": "L1-final",
            }))

    # Aggregate results
    all_results = {}
    for idx, result in l0_caught:
        all_results[idx] = result
    for idx, result in l1_caught:
        all_results[idx] = result
    for idx, result in l2_results:
        all_results[idx] = result

    # Sort by index to maintain order
    results = [all_results[i] for i in sorted(all_results.keys())]

    return results, {
        "L0": len(l0_caught),
        "L1": len(l1_caught),
        "L2": len(l2_results),
    }


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
    parser.add_argument("--no-l2", action="store_true", help="Disable L2 (stop at L1)")
    parser.add_argument("--l0-threshold", type=float, default=0.7, help="L0 confidence threshold")
    parser.add_argument("--l1-threshold", type=float, default=0.7, help="L1 confidence threshold")
    args = parser.parse_args()

    print("="*60)
    print("COMPREHENSIVE SAFETY CASCADE BENCHMARK")
    print("="*60)

    mode = "L0 only" if args.l0_only else f"Full Cascade (L0→L1→L2)"
    print(f"Mode: {mode}")
    print(f"L0 threshold: {args.l0_threshold}")
    print(f"L1 threshold: {args.l1_threshold}")

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
    all_layer_dist = defaultdict(int)
    start_time = time.time()

    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)

    for name, data in benchmarks.items():
        print(f"\nEvaluating {name} ({len(data)} samples)...")

        if args.l0_only:
            # L0 only mode
            l0_caught, l0_uncertain = run_l0_batch(data, args.l0_threshold)
            # For L0-only, treat uncertain as L0's best guess
            results = []
            for idx, result in l0_caught:
                results.append(result)
            for idx, sample, l0_result in l0_uncertain:
                results.append({
                    "text": sample.get("text", ""),
                    "expected": sample.get("label", ""),
                    "predicted": l0_result["label"],
                    "confidence": l0_result["confidence"],
                    "stopped_at": "L0",
                })
            layer_dist = {"L0": len(data), "L1": 0, "L2": 0}
        else:
            # Full cascade
            results, layer_dist = run_cascade_batch(
                data,
                l0_threshold=args.l0_threshold,
                l1_threshold=args.l1_threshold,
                enable_l2=not args.no_l2
            )

        all_results[name] = results

        # Accumulate layer distribution
        for layer, count in layer_dist.items():
            all_layer_dist[layer] += count

        metrics = calculate_metrics(results)
        print(f"  Accuracy: {metrics['accuracy']:.1f}%")
        print(f"  Recall: {metrics['recall']:.1f}%")
        print(f"  F1: {metrics['f1']:.1f}%")
        print(f"  Layers: L0={layer_dist['L0']}, L1={layer_dist['L1']}, L2={layer_dist['L2']}")

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

    print(f"\nLayer Distribution:")
    total_layered = sum(all_layer_dist.values())
    for layer in ["L0", "L1", "L2"]:
        count = all_layer_dist[layer]
        pct = count / total_layered * 100 if total_layered > 0 else 0
        print(f"  {layer}: {count} ({pct:.1f}%)")

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
