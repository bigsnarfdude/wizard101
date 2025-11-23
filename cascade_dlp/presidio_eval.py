#!/usr/bin/env python3
"""
Presidio PII Detection - Project Emmentaler

Microsoft's battle-tested PII detection. Don't reinvent the wheel.
"""

import time
from collections import Counter
from datasets import load_dataset
from presidio_analyzer import AnalyzerEngine


def main():
    print("=" * 60)
    print("PRESIDIO PII DETECTION - Project Emmentaler")
    print("=" * 60)

    # Load Presidio
    print("\nLoading Presidio analyzer...")
    analyzer = AnalyzerEngine()
    print("Ready.")

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    print(f"Total samples: {len(ds)}")

    # Evaluate
    print("\n" + "-" * 60)
    print("EVALUATION")
    print("-" * 60)

    n_samples = 500
    tp, fp, fn = 0, 0, 0
    total_time = 0
    type_counts = Counter()

    for i in range(n_samples):
        item = ds[i]
        text = item['source_text']
        has_pii = bool(item.get('privacy_mask'))

        # Scan
        start = time.time()
        results = analyzer.analyze(text=text, language="en")
        total_time += time.time() - start

        found_pii = len(results) > 0

        # Count types
        for r in results:
            type_counts[r.entity_type] += 1

        # Metrics
        if found_pii and has_pii:
            tp += 1
        elif found_pii and not has_pii:
            fp += 1
        elif not found_pii and has_pii:
            fn += 1

    # Results
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_latency = (total_time / n_samples) * 1000

    print(f"\nResults ({n_samples} samples):")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    print(f"  Avg Latency: {avg_latency:.1f}ms")

    print(f"\nDetections by type:")
    for pii_type, count in type_counts.most_common(10):
        print(f"  {pii_type}: {count}")

    # Show examples
    print("\n" + "-" * 60)
    print("EXAMPLES")
    print("-" * 60)

    for i in range(3):
        text = ds[i]['source_text']
        results = analyzer.analyze(text=text, language="en")

        print(f"\nSample {i}:")
        print(f"  Text: {text[:100]}...")
        print(f"  Found: {[(r.entity_type, text[r.start:r.end]) for r in results[:5]]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
