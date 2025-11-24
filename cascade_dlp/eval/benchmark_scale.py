"""
Scale benchmark evaluation for cascade_dlp.

Tests on full 200k ai4privacy dataset to check for overfitting.
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cascade import DLPCascade


@dataclass
class BenchmarkResult:
    """Result from benchmark evaluation."""
    dataset_name: str
    total_samples: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    avg_latency: float


def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> tuple:
    """Calculate precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def evaluate_ai4privacy_full(cascade: DLPCascade, test_path: str) -> BenchmarkResult:
    """
    Evaluate on full ai4privacy dataset (200k samples).
    """
    print(f"Loading {test_path}...")
    with open(test_path) as f:
        samples = json.load(f)

    total = len(samples)
    print(f"Loaded {total} samples")

    tp, fp, fn, tn = 0, 0, 0, 0
    latencies = []

    # Progress tracking
    progress_interval = total // 20  # Report every 5%

    for i, sample in enumerate(samples):
        if i % progress_interval == 0:
            pct = (i / total) * 100
            print(f"  Progress: {i}/{total} ({pct:.0f}%)")

        # Use source text (has PII) or masked text
        text = sample.get("text", "")
        if not text:
            continue

        # Check if this sample has PII (based on privacy_mask or span_labels)
        has_pii = bool(sample.get("privacy_mask")) or bool(sample.get("span_labels"))

        # Run cascade
        start = time.time()
        result = cascade.run(text)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        # Check if detected
        has_detection = len(result.total_detections) > 0

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

    print(f"  Progress: {total}/{total} (100%)")

    precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)

    return BenchmarkResult(
        dataset_name="ai4privacy_full",
        total_samples=len(samples),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_p50=statistics.median(latencies) if latencies else 0,
        latency_p95=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else (latencies[0] if latencies else 0),
        latency_p99=sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else (latencies[0] if latencies else 0),
        avg_latency=statistics.mean(latencies) if latencies else 0,
    )


def print_result(result: BenchmarkResult):
    """Print benchmark result in formatted table."""
    print(f"\n{'─' * 60}")
    print(f"Dataset: {result.dataset_name}")
    print(f"{'─' * 60}")
    print(f"Samples: {result.total_samples}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {result.true_positives:6d}  |  FP: {result.false_positives:6d}")
    print(f"  FN: {result.false_negatives:6d}  |  TN: {result.true_negatives:6d}")
    print(f"\nMetrics:")
    print(f"  Precision: {result.precision:.1%}")
    print(f"  Recall:    {result.recall:.1%}")
    print(f"  F1 Score:  {result.f1:.1%}")
    print(f"\nLatency:")
    print(f"  p50: {result.latency_p50:.1f}ms")
    print(f"  p95: {result.latency_p95:.1f}ms")
    print(f"  p99: {result.latency_p99:.1f}ms")
    print(f"  avg: {result.avg_latency:.1f}ms")


def main():
    """Run scale benchmark on full 200k dataset."""
    print("=" * 60)
    print("CASCADE DLP - SCALE BENCHMARK (200K SAMPLES)")
    print("=" * 60)

    # Initialize cascade
    cascade = DLPCascade(
        secret_threshold=0.7,
        pii_threshold=0.7,
        block_on_high_confidence=True
    )

    # Dataset path
    dataset_dir = Path(__file__).parent / "datasets"
    ai4privacy_full_path = dataset_dir / "ai4privacy_full.json"

    if not ai4privacy_full_path.exists():
        print(f"ERROR: {ai4privacy_full_path} not found")
        print("Run download_all_datasets.py first")
        return

    # Run evaluation
    print("\nEvaluating on ai4privacy FULL (200k samples)...")
    print("This will take several minutes...")

    start_time = time.time()
    result = evaluate_ai4privacy_full(cascade, str(ai4privacy_full_path))
    total_time = time.time() - start_time

    print_result(result)

    # Comparison with 1000 sample results
    print("\n" + "=" * 60)
    print("SCALE VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")
    print(f"Throughput: {result.total_samples / total_time:.0f} samples/sec")

    print("\nComparison (1000 vs 200k samples):")
    print("-" * 60)
    print(f"{'Metric':<15} {'1000 samples':>15} {'200k samples':>15}")
    print("-" * 60)
    # 1000 sample results from previous run
    print(f"{'Precision':<15} {'99.8%':>15} {result.precision:>14.1%}")
    print(f"{'Recall':<15} {'87.5%':>15} {result.recall:>14.1%}")
    print(f"{'F1':<15} {'93.3%':>15} {result.f1:>14.1%}")
    print(f"{'Latency p50':<15} {'5.1ms':>15} {result.latency_p50:>13.1f}ms")
    print("-" * 60)

    # Check for overfitting
    print("\nOverfitting Analysis:")
    if result.f1 > 0.90:
        print("  ✓ F1 > 90% - No significant overfitting detected")
    elif result.f1 > 0.85:
        print("  ⚠ F1 85-90% - Slight performance drop, may need investigation")
    else:
        print("  ✗ F1 < 85% - Possible overfitting on small sample")

    if abs(result.recall - 0.875) < 0.05:
        print("  ✓ Recall consistent with 1000-sample evaluation")
    else:
        print(f"  ⚠ Recall changed: 87.5% → {result.recall:.1%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
