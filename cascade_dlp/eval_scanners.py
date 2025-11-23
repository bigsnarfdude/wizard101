#!/usr/bin/env python3
"""
Compare PII detection approaches:
1. Pattern scanner (baseline)
2. spaCy NER
3. Presidio (Microsoft's PII detection)

Find the best speed/accuracy tradeoff.
"""

import re
import time
from collections import Counter
from datasets import load_dataset

# Try imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not installed: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("Presidio not installed: pip install presidio-analyzer")


# Pattern scanner
PATTERNS = {
    "EMAIL": r"\b[\w.-]+@[\w.-]+\.\w{2,}\b",
    "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
}


def pattern_scan(text: str) -> list:
    """Pattern-based scanner."""
    findings = []
    for pii_type, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            findings.append({
                "type": pii_type,
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })
    return findings


def spacy_scan(nlp, text: str) -> list:
    """spaCy NER scanner."""
    doc = nlp(text)
    findings = []

    # Map spaCy entities to PII types
    pii_mapping = {
        "PERSON": "NAME",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "DATE": "DATE",
        "TIME": "TIME",
        "MONEY": "FINANCIAL",
        "CARDINAL": "NUMBER",
    }

    for ent in doc.ents:
        if ent.label_ in pii_mapping:
            findings.append({
                "type": pii_mapping[ent.label_],
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char
            })

    return findings


def presidio_scan(analyzer, text: str) -> list:
    """Presidio PII scanner."""
    results = analyzer.analyze(text=text, language="en")
    findings = []

    for result in results:
        findings.append({
            "type": result.entity_type,
            "text": text[result.start:result.end],
            "start": result.start,
            "end": result.end,
            "score": result.score
        })

    return findings


def combined_scan(nlp, text: str) -> list:
    """Pattern + spaCy combined."""
    findings = pattern_scan(text)
    findings.extend(spacy_scan(nlp, text))
    return findings


def evaluate_scanner(scanner_fn, dataset, name: str, n_samples: int = 500):
    """Evaluate a scanner on the dataset."""
    print(f"\n{'─' * 50}")
    print(f"Evaluating: {name}")
    print(f"{'─' * 50}")

    tp = 0  # True positives (found PII when PII exists)
    fp = 0  # False positives (found PII when no PII)
    fn = 0  # False negatives (missed PII when PII exists)

    total_time = 0
    type_counts = Counter()

    for i in range(min(n_samples, len(dataset))):
        item = dataset[i]
        text = item['source_text']

        # Ground truth
        has_pii = bool(item.get('privacy_mask'))

        # Scan
        start = time.time()
        findings = scanner_fn(text)
        total_time += time.time() - start

        found_pii = len(findings) > 0

        # Count types
        for f in findings:
            type_counts[f['type']] += 1

        # Metrics
        if found_pii and has_pii:
            tp += 1
        elif found_pii and not has_pii:
            fp += 1
        elif not found_pii and has_pii:
            fn += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_latency = (total_time / n_samples) * 1000

    print(f"\nResults ({n_samples} samples):")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    print(f"  Avg Latency: {avg_latency:.1f}ms")

    print(f"\nTop detections:")
    for pii_type, count in type_counts.most_common(5):
        print(f"  {pii_type}: {count}")

    return {
        "name": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": avg_latency
    }


def main():
    print("=" * 60)
    print("PII SCANNER COMPARISON - Project Emmentaler")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    print(f"Total samples: {len(ds)}")

    results = []
    n_samples = 500

    # 1. Pattern scanner baseline
    results.append(evaluate_scanner(
        pattern_scan, ds, "Pattern Scanner (regex)", n_samples
    ))

    # 2. spaCy NER
    if SPACY_AVAILABLE:
        print("\nLoading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        results.append(evaluate_scanner(
            lambda text: spacy_scan(nlp, text),
            ds, "spaCy NER", n_samples
        ))

        # 3. Combined Pattern + spaCy
        results.append(evaluate_scanner(
            lambda text: combined_scan(nlp, text),
            ds, "Pattern + spaCy", n_samples
        ))

    # 4. Presidio
    if PRESIDIO_AVAILABLE:
        print("\nLoading Presidio analyzer...")
        analyzer = AnalyzerEngine()
        results.append(evaluate_scanner(
            lambda text: presidio_scan(analyzer, text),
            ds, "Presidio", n_samples
        ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n{:<25} {:>10} {:>10} {:>10} {:>12}".format(
        "Scanner", "Precision", "Recall", "F1", "Latency"
    ))
    print("-" * 60)

    for r in results:
        print("{:<25} {:>10.1%} {:>10.1%} {:>10.1%} {:>10.1f}ms".format(
            r["name"], r["precision"], r["recall"], r["f1"], r["latency_ms"]
        ))

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if results:
        best = max(results, key=lambda x: x["f1"])
        print(f"\nBest F1: {best['name']} ({best['f1']:.1%})")

        fastest = min(results, key=lambda x: x["latency_ms"])
        print(f"Fastest: {fastest['name']} ({fastest['latency_ms']:.1f}ms)")

    print("\nNext steps:")
    print("1. If Presidio wins: use as primary scanner")
    print("2. If spaCy wins: faster, lighter weight")
    print("3. Consider cascade: pattern first → NER for misses")


if __name__ == "__main__":
    main()
