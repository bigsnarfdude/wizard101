"""
Investigate false negatives in ai4privacy benchmark.

Find patterns in the 125 missed detections.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cascade import DLPCascade


def main():
    # Load dataset
    dataset_path = Path(__file__).parent / "datasets" / "ai4privacy_sample.json"
    with open(dataset_path) as f:
        samples = json.load(f)

    # Initialize cascade
    cascade = DLPCascade(
        secret_threshold=0.7,
        pii_threshold=0.7,
        block_on_high_confidence=True
    )

    # Find false negatives
    false_negatives = []
    detected_types = Counter()
    missed_types = Counter()

    for i, sample in enumerate(samples):
        text = sample.get("text", "")
        if not text:
            continue

        # Check ground truth
        privacy_mask = sample.get("privacy_mask", [])
        span_labels = sample.get("span_labels", [])
        has_pii = bool(privacy_mask) or bool(span_labels)

        if not has_pii:
            continue

        # Run cascade
        result = cascade.run(text)
        has_detection = len(result.total_detections) > 0

        # Track detected types
        for d in result.total_detections:
            detected_types[d.entity_type] += 1

        # If missed, analyze
        if not has_detection:
            false_negatives.append({
                "index": i,
                "text": text[:200] + "..." if len(text) > 200 else text,
                "privacy_mask": privacy_mask,
                "span_labels": span_labels,
            })

            # Track missed entity types
            if span_labels:
                for label in span_labels:
                    if isinstance(label, dict):
                        missed_types[label.get("label", "unknown")] += 1
                    elif isinstance(label, str):
                        missed_types[label] += 1

    # Report
    print("=" * 70)
    print("FALSE NEGATIVE INVESTIGATION")
    print("=" * 70)

    print(f"\nTotal false negatives: {len(false_negatives)}")

    # Show missed entity types
    print(f"\nMissed entity types:")
    for entity_type, count in missed_types.most_common(20):
        print(f"  {entity_type}: {count}")

    # Show detected types (what we ARE catching)
    print(f"\nDetected entity types (what's working):")
    for entity_type, count in detected_types.most_common(20):
        print(f"  {entity_type}: {count}")

    # Show sample false negatives
    print(f"\nSample false negatives (first 10):")
    print("-" * 70)

    for fn in false_negatives[:10]:
        print(f"\n[{fn['index']}] Text: {fn['text']}")
        if fn['span_labels']:
            print(f"    Labels: {fn['span_labels'][:5]}")
        if fn['privacy_mask']:
            print(f"    Mask: {fn['privacy_mask'][:100]}...")

    # Check dataset format
    print("\n" + "=" * 70)
    print("DATASET FORMAT CHECK")
    print("=" * 70)

    # Sample the first item to understand structure
    sample = samples[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Text length: {len(sample.get('text', ''))}")
    print(f"Privacy mask type: {type(sample.get('privacy_mask'))}")
    print(f"Span labels type: {type(sample.get('span_labels'))}")

    if sample.get('span_labels'):
        print(f"First span label: {sample['span_labels'][0] if sample['span_labels'] else 'None'}")

    # Check if text has the actual PII or is already masked
    print(f"\nSample text (first 300 chars):")
    print(sample.get('text', '')[:300])


if __name__ == "__main__":
    main()
