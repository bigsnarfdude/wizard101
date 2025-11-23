"""
Benchmark evaluation for cascade_dlp.

Evaluates the DLP cascade against:
- ai4privacy PII dataset (1000 samples)
- Curated PII test set
- Curated secret test set

Metrics:
- Precision, Recall, F1
- Latency (p50, p95, p99)
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cascade import DLPCascade
from detectors.secret_detector import SecretDetector


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


def evaluate_secret_test_set(cascade: DLPCascade, test_path: str) -> BenchmarkResult:
    """Evaluate on curated secret test set."""
    with open(test_path) as f:
        test_cases = json.load(f)

    tp, fp, fn, tn = 0, 0, 0, 0
    latencies = []

    for case in test_cases:
        text = case["text"]
        expected = case["expected"]
        should_detect = case["label"]

        # Run cascade
        start = time.time()
        result = cascade.run(text)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        # Check detections
        detected_types = set()
        for detection in result.total_detections:
            detected_types.add(detection.entity_type)

        # Calculate TP/FP/FN/TN
        if should_detect:
            if len(detected_types) > 0:
                tp += 1
            else:
                fn += 1
        else:
            if len(detected_types) > 0:
                fp += 1
            else:
                tn += 1

    precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)

    return BenchmarkResult(
        dataset_name="secret_test_set",
        total_samples=len(test_cases),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_p50=statistics.median(latencies),
        latency_p95=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
        latency_p99=sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0],
        avg_latency=statistics.mean(latencies),
    )


def evaluate_pii_test_set(cascade: DLPCascade, test_path: str) -> BenchmarkResult:
    """Evaluate on curated PII test set."""
    with open(test_path) as f:
        test_cases = json.load(f)

    tp, fp, fn, tn = 0, 0, 0, 0
    latencies = []

    for case in test_cases:
        text = case["text"]
        expected = case["expected"]
        should_detect = case["label"]

        # Run cascade
        start = time.time()
        result = cascade.run(text)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        # Check if any PII detected
        has_detection = len(result.total_detections) > 0

        if should_detect:
            if has_detection:
                tp += 1
            else:
                fn += 1
        else:
            if has_detection:
                fp += 1
            else:
                tn += 1

    precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)

    return BenchmarkResult(
        dataset_name="pii_test_set",
        total_samples=len(test_cases),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_p50=statistics.median(latencies),
        latency_p95=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
        latency_p99=sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0],
        avg_latency=statistics.mean(latencies),
    )


def evaluate_ai4privacy(cascade: DLPCascade, test_path: str, limit: int = None) -> BenchmarkResult:
    """
    Evaluate on ai4privacy dataset.

    The dataset has masked text with PII - we check if cascade detects PII.
    """
    with open(test_path) as f:
        samples = json.load(f)

    if limit:
        samples = samples[:limit]

    tp, fp, fn, tn = 0, 0, 0, 0
    latencies = []

    for sample in samples:
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

    precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)

    return BenchmarkResult(
        dataset_name="ai4privacy",
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
    print(f"  TP: {result.true_positives:4d}  |  FP: {result.false_positives:4d}")
    print(f"  FN: {result.false_negatives:4d}  |  TN: {result.true_negatives:4d}")
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
    """Run benchmark evaluation."""
    print("=" * 60)
    print("CASCADE DLP - BENCHMARK EVALUATION")
    print("=" * 60)

    # Initialize cascade
    cascade = DLPCascade(
        secret_threshold=0.7,
        pii_threshold=0.7,
        block_on_high_confidence=True
    )

    # Dataset paths
    dataset_dir = Path(__file__).parent / "datasets"

    results = []

    # 1. Secret test set
    secret_path = dataset_dir / "secret_test_set.json"
    if secret_path.exists():
        print("\nEvaluating: secret_test_set...")
        result = evaluate_secret_test_set(cascade, str(secret_path))
        results.append(result)
        print_result(result)

    # 2. PII test set
    pii_path = dataset_dir / "pii_test_set.json"
    if pii_path.exists():
        print("\nEvaluating: pii_test_set...")
        result = evaluate_pii_test_set(cascade, str(pii_path))
        results.append(result)
        print_result(result)

    # 3. ai4privacy (full 1000 samples)
    ai4privacy_path = dataset_dir / "ai4privacy_sample.json"
    if ai4privacy_path.exists():
        print("\nEvaluating: ai4privacy (1000 samples)...")
        result = evaluate_ai4privacy(cascade, str(ai4privacy_path))
        results.append(result)
        print_result(result)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\n{'Dataset':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Latency':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.dataset_name:<20} {r.precision:>9.1%} {r.recall:>9.1%} {r.f1:>9.1%} {r.avg_latency:>8.1f}ms")

    # Overall metrics
    if results:
        total_tp = sum(r.true_positives for r in results)
        total_fp = sum(r.false_positives for r in results)
        total_fn = sum(r.false_negatives for r in results)
        total_tn = sum(r.true_negatives for r in results)
        overall_p, overall_r, overall_f1 = calculate_metrics(total_tp, total_fp, total_fn, total_tn)
        avg_latency = statistics.mean([r.avg_latency for r in results])

        print("-" * 60)
        print(f"{'OVERALL':<20} {overall_p:>9.1%} {overall_r:>9.1%} {overall_f1:>9.1%} {avg_latency:>8.1f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
