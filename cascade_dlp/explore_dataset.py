#!/usr/bin/env python3
"""
Explore ai4privacy PII dataset and run baseline pattern scanner.

Quick understanding of what we're working with.
"""

import re
from collections import Counter
from datasets import load_dataset

# Pattern scanner baseline
PATTERNS = {
    "EMAIL": r"\b[\w.-]+@[\w.-]+\.\w{2,}\b",
    "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
}


def scan_text(text: str) -> dict:
    """Scan text for PII patterns."""
    findings = {}
    for name, pattern in PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            findings[name] = matches
    return findings


def main():
    print("=" * 70)
    print("AI4PRIVACY DATASET EXPLORATION")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    print(f"Total samples: {len(ds)}")

    # Explore structure
    print("\n" + "-" * 70)
    print("DATASET STRUCTURE")
    print("-" * 70)

    sample = ds[0]
    print(f"Fields: {list(sample.keys())}")
    print(f"\nExample:")
    print(f"  source_text: {sample['source_text'][:100]}...")
    print(f"  target_text: {sample['target_text'][:100]}...")
    print(f"  language: {sample.get('language', 'N/A')}")

    if 'privacy_mask' in sample:
        print(f"  privacy_mask: {sample['privacy_mask'][:3]}...")
    if 'span_labels' in sample:
        print(f"  span_labels: {sample['span_labels'][:3]}...")

    # Count PII types in dataset
    print("\n" + "-" * 70)
    print("PII TYPE DISTRIBUTION")
    print("-" * 70)

    pii_types = Counter()

    # Sample first 5000 for speed
    for i, item in enumerate(ds):
        if i >= 5000:
            break
        if 'privacy_mask' in item and item['privacy_mask']:
            for mask in item['privacy_mask']:
                if isinstance(mask, dict) and 'label' in mask:
                    pii_types[mask['label']] += 1
                elif isinstance(mask, str):
                    pii_types[mask] += 1

    print(f"\nTop 15 PII types (first 5K samples):")
    for pii_type, count in pii_types.most_common(15):
        print(f"  {pii_type}: {count}")

    # Run pattern scanner baseline
    print("\n" + "-" * 70)
    print("PATTERN SCANNER BASELINE")
    print("-" * 70)

    # Test on 1000 samples
    test_size = 1000
    pattern_findings = Counter()
    texts_with_pii = 0

    for i in range(test_size):
        findings = scan_text(ds[i]['source_text'])
        if findings:
            texts_with_pii += 1
            for pii_type in findings:
                pattern_findings[pii_type] += len(findings[pii_type])

    print(f"\nPattern scanner results (first {test_size} samples):")
    print(f"Texts with detected PII: {texts_with_pii}/{test_size} ({texts_with_pii/test_size*100:.1f}%)")
    print(f"\nDetections by type:")
    for pii_type, count in pattern_findings.most_common():
        print(f"  {pii_type}: {count}")

    # Show some examples
    print("\n" + "-" * 70)
    print("EXAMPLE DETECTIONS")
    print("-" * 70)

    examples_shown = 0
    for i in range(min(100, len(ds))):
        findings = scan_text(ds[i]['source_text'])
        if findings and examples_shown < 5:
            print(f"\nSample {i}:")
            print(f"  Text: {ds[i]['source_text'][:150]}...")
            print(f"  Found: {findings}")
            examples_shown += 1

    # Calculate baseline metrics
    print("\n" + "-" * 70)
    print("BASELINE METRICS ESTIMATE")
    print("-" * 70)

    # Compare pattern detections to ground truth
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    for i in range(min(500, len(ds))):
        item = ds[i]
        pattern_found = bool(scan_text(item['source_text']))

        # Check if ground truth has PII
        has_pii = False
        if 'privacy_mask' in item and item['privacy_mask']:
            has_pii = len(item['privacy_mask']) > 0

        if pattern_found and has_pii:
            tp += 1
        elif pattern_found and not has_pii:
            fp += 1
        elif not pattern_found and has_pii:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nBaseline (first 500 samples):")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Gap Analysis: Which PII types are we missing? (high FN)
2. Pattern Expansion: Add patterns for top missed types
3. False Positive Analysis: What's causing FPs?
4. NER Baseline: Compare to transformer-based detection
""")


if __name__ == "__main__":
    main()
