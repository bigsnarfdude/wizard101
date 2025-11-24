#!/usr/bin/env python3
"""
Benchmark Cascade DLP v2 on ai4privacy dataset.
"""

import json
import sys
import time
from pathlib import Path
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cascade import DLPCascade


def main():
    """Run benchmark on ai4privacy dataset."""
    print("=" * 60)
    print("CASCADE DLP v2 - BENCHMARK")
    print("=" * 60)

    # Find dataset
    dataset_paths = [
        Path(__file__).parent.parent.parent.parent / "eval" / "datasets" / "ai4privacy_full.json",
        Path(__file__).parent.parent.parent.parent / "eval" / "datasets" / "ai4privacy_sample.json",
    ]

    dataset_path = None
    for path in dataset_paths:
        if path.exists():
            dataset_path = path
            break

    if not dataset_path:
        print("ERROR: No dataset found")
        print("Run download_all_datasets.py in eval/ first")
        return

    print(f"\nDataset: {dataset_path.name}")

    # Load dataset
    with open(dataset_path) as f:
        samples = json.load(f)

    # Limit for testing
    limit = 1000
    samples = samples[:limit]
    print(f"Samples: {len(samples)}")

    # Initialize DLP
    print("\nInitializing Cascade DLP v2...")
    dlp = DLPCascade()

    # Run benchmark
    print("\nRunning benchmark...")
    tp, fp, fn, tn = 0, 0, 0, 0
    latencies = []

    progress_interval = len(samples) // 20

    for i, sample in enumerate(samples):
        if i % progress_interval == 0:
            print(f"  Progress: {i}/{len(samples)} ({100*i//len(samples)}%)")

        text = sample.get("text", "")
        if not text:
            continue

        # Check if has PII
        has_pii = bool(sample.get("privacy_mask")) or bool(sample.get("span_labels"))

        # Run detection
        start = time.time()
        result = dlp.process(text)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        has_detection = len(result.detections) > 0

        if has_pii:
            if has_detection:
                tp += 1
            else:
                fn += 1
        else:
            if has_detection:
                fp += 1
            else:
                tn += 1

    print(f"  Progress: {len(samples)}/{len(samples)} (100%)")

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:6d}  |  FP: {fp:6d}")
    print(f"  FN: {fn:6d}  |  TN: {tn:6d}")

    print(f"\nMetrics:")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")

    print(f"\nLatency:")
    print(f"  p50: {statistics.median(latencies):.1f}ms")
    print(f"  p95: {sorted(latencies)[int(len(latencies) * 0.95)]:.1f}ms")
    print(f"  p99: {sorted(latencies)[int(len(latencies) * 0.99)]:.1f}ms")
    print(f"  avg: {statistics.mean(latencies):.1f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
