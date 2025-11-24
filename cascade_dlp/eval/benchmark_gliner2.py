"""
GLiNER2 benchmark on ai4privacy dataset.

Compares zero-shot PII detection vs current cascade approach.
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from gliner2 import GLiNER2
    GLINER2_AVAILABLE = True
except ImportError:
    GLINER2_AVAILABLE = False
    print("Install: pip install gliner2")


@dataclass
class BenchmarkResult:
    """Result from benchmark evaluation."""
    model_name: str
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


class GLiNER2Detector:
    """GLiNER2-based PII detector."""

    def __init__(self, model_name: str = "fastino/gliner2-base-v1"):
        print(f"Loading GLiNER2 model: {model_name}")
        self.model = GLiNER2.from_pretrained(model_name)

        # Comprehensive PII types for zero-shot detection
        self.pii_types = [
            # Personal identifiers
            "person name",
            "first name",
            "last name",
            "email address",
            "phone number",
            "social security number",
            "passport number",
            "driver license number",

            # Financial
            "credit card number",
            "bank account number",
            "iban",

            # Location
            "street address",
            "city",
            "zip code",
            "country",

            # Digital
            "ip address",
            "mac address",
            "username",
            "password",
            "api key",
            "access token",

            # Demographics
            "date of birth",
            "age",
            "gender",

            # Other
            "vehicle registration",
            "medical record number",
            "tax id",
        ]
        print(f"Configured {len(self.pii_types)} PII types")

    def detect(self, text: str) -> list:
        """Detect PII in text."""
        try:
            entities = self.model.extract_entities(text, self.pii_types)
            return entities
        except Exception as e:
            print(f"Detection error: {e}")
            return []


def evaluate_gliner2(dataset_path: str, limit: int = None) -> BenchmarkResult:
    """
    Evaluate GLiNER2 on ai4privacy dataset.
    """
    if not GLINER2_AVAILABLE:
        print("GLiNER2 not available")
        return None

    print(f"\nLoading dataset: {dataset_path}")
    with open(dataset_path) as f:
        samples = json.load(f)

    if limit:
        samples = samples[:limit]

    total = len(samples)
    print(f"Evaluating on {total} samples")

    # Initialize detector
    detector = GLiNER2Detector()

    tp, fp, fn, tn = 0, 0, 0, 0
    latencies = []

    # Progress tracking
    progress_interval = max(1, total // 20)

    for i, sample in enumerate(samples):
        if i % progress_interval == 0:
            pct = (i / total) * 100
            print(f"  Progress: {i}/{total} ({pct:.0f}%)")

        text = sample.get("text", "")
        if not text:
            continue

        # Check if sample has PII
        has_pii = bool(sample.get("privacy_mask")) or bool(sample.get("span_labels"))

        # Run detection
        start = time.time()
        entities = detector.detect(text)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        # Check if detected
        has_detection = len(entities) > 0

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
        model_name="GLiNER2-base",
        total_samples=len(samples),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_p50=statistics.median(latencies) if latencies else 0,
        latency_p95=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0,
        latency_p99=sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else 0,
        avg_latency=statistics.mean(latencies) if latencies else 0,
    )


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print(f"\n{'─' * 60}")
    print(f"Model: {result.model_name}")
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
    """Run GLiNER2 benchmark."""
    print("=" * 60)
    print("GLINER2 BENCHMARK - ai4privacy Dataset")
    print("=" * 60)

    # Dataset path
    dataset_dir = Path(__file__).parent / "datasets"

    # Check for datasets
    full_path = dataset_dir / "ai4privacy_full.json"
    sample_path = dataset_dir / "ai4privacy_sample.json"

    if full_path.exists():
        dataset_path = full_path
        # Start with smaller sample for initial test
        limit = 1000  # Increase to None for full benchmark
    elif sample_path.exists():
        dataset_path = sample_path
        limit = None
    else:
        print("ERROR: No dataset found")
        print("Run download_all_datasets.py first")
        return

    print(f"\nDataset: {dataset_path.name}")
    print(f"Limit: {limit if limit else 'all'}")

    # Run benchmark
    start_time = time.time()
    result = evaluate_gliner2(str(dataset_path), limit=limit)
    total_time = time.time() - start_time

    if result:
        print_result(result)

        # Comparison with cascade
        print("\n" + "=" * 60)
        print("COMPARISON: GLiNER2 vs Cascade (Presidio)")
        print("=" * 60)

        print(f"\nTotal time: {total_time/60:.1f} minutes")
        if result.total_samples > 0:
            print(f"Throughput: {result.total_samples / total_time:.0f} samples/sec")

        print("\n" + "-" * 60)
        print(f"{'Metric':<15} {'Cascade':>15} {'GLiNER2':>15}")
        print("-" * 60)
        # Cascade results from previous benchmark
        print(f"{'Precision':<15} {'100.0%':>15} {result.precision:>14.1%}")
        print(f"{'Recall':<15} {'88.2%':>15} {result.recall:>14.1%}")
        print(f"{'F1':<15} {'93.7%':>15} {result.f1:>14.1%}")
        print(f"{'Latency p50':<15} {'3.7ms':>15} {result.latency_p50:>13.1f}ms")
        print("-" * 60)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
