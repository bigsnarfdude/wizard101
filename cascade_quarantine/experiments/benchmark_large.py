#!/usr/bin/env python3
"""
Large-Scale Benchmark: Full Dataset Testing

Tests on xTRam1/safe-guard-prompt-injection dataset (8K+ samples).
Uses a held-out test set to measure real-world performance.

Usage:
    python experiments/benchmark_large.py [--samples N]

Default: 1000 samples (500 benign, 500 injection)
Use --samples 0 for full dataset (~8K samples, ~1 hour)
"""

import sys
import os
import time
import argparse
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quarantine import Quarantine


def load_dataset(n_samples=1000):
    """Load xTRam1 dataset."""
    import json

    # Try local JSON first (pre-exported)
    json_path = os.path.join(os.path.dirname(__file__), "dataset_xtram1.json")
    if os.path.exists(json_path):
        print(f"Loading from {json_path}...")
        with open(json_path) as f:
            ds = json.load(f)
    else:
        # Fall back to HuggingFace
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load('xTRam1/safe-guard-prompt-injection', split='train')
            ds = [{'text': x['text'], 'label': x['label']} for x in ds]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Either place dataset_xtram1.json in experiments/ or install: pip install datasets")
            sys.exit(1)

    # Split by label
    benign = [(x['text'], 0) for x in ds if x['label'] == 0]
    injection = [(x['text'], 1) for x in ds if x['label'] == 1]

    print(f"Dataset loaded: {len(benign)} benign, {len(injection)} injection")

    if n_samples == 0:
        # Use all samples
        samples = benign + injection
    else:
        # Balance sample sizes
        n_each = n_samples // 2
        random.seed(42)  # Reproducible
        samples = random.sample(benign, min(n_each, len(benign)))
        samples += random.sample(injection, min(n_each, len(injection)))

    random.shuffle(samples)
    return samples


def run_benchmark(n_samples=1000):
    """Run large-scale benchmark."""
    print("=" * 70)
    print("CASCADE QUARANTINE - LARGE SCALE BENCHMARK")
    print("=" * 70)
    print()

    # Load data
    print("Loading dataset...")
    samples = load_dataset(n_samples)
    print(f"Testing on {len(samples)} samples")
    print()

    # Count expected
    n_benign = sum(1 for _, label in samples if label == 0)
    n_injection = sum(1 for _, label in samples if label == 1)
    print(f"  Benign samples: {n_benign}")
    print(f"  Injection samples: {n_injection}")
    print()

    # Initialize quarantine
    print("Initializing Quarantine...")
    try:
        quarantine = Quarantine(model="qwen3:4b", use_classifier=True)
        print("✓ Quarantine ready")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    print()
    print("-" * 70)
    print("RUNNING TESTS...")
    print("-" * 70)
    print()

    # Track results
    tp, tn, fp, fn = 0, 0, 0, 0
    latencies = []
    fp_examples = []
    fn_examples = []

    start_time = time.time()

    for i, (text, label) in enumerate(samples):
        # Progress update every 100 samples
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{len(samples)} ({(i+1)/len(samples)*100:.1f}%) "
                  f"- {rate:.1f} samples/sec - ETA: {eta/60:.1f}min")

        # Run quarantine
        t0 = time.time()
        try:
            result = quarantine.extract_intent(text)
            latency = (time.time() - t0) * 1000
            latencies.append(latency)

            predicted_injection = not result.safe_to_proceed
            actual_injection = (label == 1)

            if actual_injection and predicted_injection:
                tp += 1
            elif not actual_injection and not predicted_injection:
                tn += 1
            elif not actual_injection and predicted_injection:
                fp += 1
                if len(fp_examples) < 10:
                    fp_examples.append({
                        "text": text[:100],
                        "suspicion": result.suspicion_level.value,
                    })
            else:  # actual_injection and not predicted_injection
                fn += 1
                if len(fn_examples) < 10:
                    fn_examples.append({
                        "text": text[:100],
                        "suspicion": result.suspicion_level.value,
                        "classifier": result.classifier_probability,
                    })

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Safe    Inject")
    print(f"  Actual Safe   {tn:5}   {fp:5}")
    print(f"  Actual Inject {fn:5}   {tp:5}")
    print()

    print("Metrics:")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print()

    fpr = fp / n_benign if n_benign > 0 else 0
    fnr = fn / n_injection if n_injection > 0 else 0
    print(f"  False Positive Rate: {fp}/{n_benign} ({fpr:.2%})")
    print(f"  False Negative Rate: {fn}/{n_injection} ({fnr:.2%})")
    print()

    # Latency stats
    if latencies:
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies) // 2]
        p95 = latencies_sorted[int(len(latencies) * 0.95)]
        p99 = latencies_sorted[int(len(latencies) * 0.99)]

        print("Latency:")
        print(f"  Min:    {min(latencies):.0f}ms")
        print(f"  Max:    {max(latencies):.0f}ms")
        print(f"  Avg:    {sum(latencies)/len(latencies):.0f}ms")
        print(f"  P50:    {p50:.0f}ms")
        print(f"  P95:    {p95:.0f}ms")
        print(f"  P99:    {p99:.0f}ms")
        print()

    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Throughput: {len(samples)/total_time:.1f} samples/sec")
    print()

    # Show error examples
    if fp_examples:
        print("FALSE POSITIVES (first 10):")
        for ex in fp_examples:
            print(f"  - {ex['text']}...")
            print(f"    Suspicion: {ex['suspicion']}")
        print()

    if fn_examples:
        print("FALSE NEGATIVES (first 10):")
        for ex in fn_examples:
            print(f"  - {ex['text']}...")
            print(f"    Suspicion: {ex['suspicion']}, Classifier: {ex['classifier']:.2%}")
        print()

    # Final verdict
    if accuracy >= 0.95 and fpr < 0.01:
        print("🎉 EXCELLENT - Production ready!")
    elif accuracy >= 0.90:
        print("✓ GOOD - Minor tuning recommended")
    elif accuracy >= 0.80:
        print("⚠️ FAIR - Needs improvement")
    else:
        print("❌ POOR - Significant work needed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples (0 for full dataset)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (auto-generates if not specified)")
    args = parser.parse_args()

    # Auto-generate output filename if not specified
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n = args.samples if args.samples > 0 else "full"
        args.output = os.path.join(os.path.dirname(__file__), f"benchmark_{n}_{timestamp}.txt")

    # Tee output to both console and file
    import sys

    class Tee:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Tee(args.output)
    print(f"Results will be saved to: {args.output}")
    print()

    run_benchmark(args.samples)

    print()
    print(f"Results saved to: {args.output}")
