"""
Comprehensive failure analysis for cascade_dlp.

Creates a database of all failures with categorization:
- Entity types missed
- Language detection
- Pattern analysis
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cascade import DLPCascade


def detect_language(text: str) -> str:
    """Simple language detection based on common patterns."""
    # Check for common non-English patterns
    french_patterns = [
        r'\b(le|la|les|de|du|des|un|une|et|est|sont|pour|dans|avec|sur|par|ce|cette)\b',
        r'\b(je|tu|il|elle|nous|vous|ils|elles|mon|ma|mes|ton|ta|tes)\b',
    ]
    italian_patterns = [
        r'\b(il|la|lo|le|gli|un|una|di|da|in|con|su|per|che|non|sono)\b',
        r'\b(io|tu|lui|lei|noi|voi|loro|mio|mia|tuo|tua)\b',
    ]
    spanish_patterns = [
        r'\b(el|la|los|las|un|una|de|en|con|por|para|que|no|es|son)\b',
        r'\b(yo|tu|el|ella|nosotros|vosotros|ellos|mi|tu|su)\b',
    ]
    german_patterns = [
        r'\b(der|die|das|den|dem|ein|eine|und|ist|sind|von|zu|mit|auf|für)\b',
        r'\b(ich|du|er|sie|es|wir|ihr|mein|dein|sein)\b',
    ]

    text_lower = text.lower()

    french_score = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in french_patterns)
    italian_score = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in italian_patterns)
    spanish_score = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in spanish_patterns)
    german_score = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in german_patterns)

    scores = {
        'french': french_score,
        'italian': italian_score,
        'spanish': spanish_score,
        'german': german_score,
    }

    max_score = max(scores.values())
    if max_score >= 3:  # Threshold for non-English
        return max(scores, key=scores.get)

    return 'english'


def extract_expected_entities(sample: Dict) -> List[str]:
    """Extract expected entity types from sample."""
    entities = []

    # From privacy_mask
    if 'privacy_mask' in sample and sample['privacy_mask']:
        for mask in sample['privacy_mask']:
            if isinstance(mask, dict) and 'label' in mask:
                entities.append(mask['label'])
            elif isinstance(mask, str):
                entities.append(mask)

    # From span_labels
    if 'span_labels' in sample and sample['span_labels']:
        for label in sample['span_labels']:
            if isinstance(label, str):
                entities.append(label)
            elif isinstance(label, dict) and 'label' in label:
                entities.append(label['label'])

    return list(set(entities))


def analyze_failures(cascade: DLPCascade, dataset_path: str, output_path: str, limit: int = None):
    """
    Analyze all failures and create detailed database.
    """
    print(f"Loading {dataset_path}...")
    with open(dataset_path) as f:
        samples = json.load(f)

    if limit:
        samples = samples[:limit]

    total = len(samples)
    print(f"Analyzing {total} samples...")

    # Track failures
    failures = []
    failure_stats = {
        'by_language': defaultdict(int),
        'by_entity_type': defaultdict(int),
        'by_entity_count': defaultdict(int),
    }

    # Track successes for comparison
    successes = {
        'by_language': defaultdict(int),
        'by_entity_type': defaultdict(int),
    }

    progress_interval = total // 20

    for i, sample in enumerate(samples):
        if i % progress_interval == 0:
            pct = (i / total) * 100
            print(f"  Progress: {i}/{total} ({pct:.0f}%)")

        text = sample.get('text', '')
        if not text:
            continue

        # Check if has PII
        has_pii = bool(sample.get('privacy_mask')) or bool(sample.get('span_labels'))
        if not has_pii:
            continue

        # Run detection
        result = cascade.run(text)
        detected = len(result.total_detections) > 0

        # Get language and expected entities
        language = detect_language(text)
        expected_entities = extract_expected_entities(sample)

        if detected:
            # Success
            successes['by_language'][language] += 1
            for entity in expected_entities:
                successes['by_entity_type'][entity] += 1
        else:
            # Failure - record details
            failure = {
                'sample_index': i,
                'text': text[:500] + '...' if len(text) > 500 else text,
                'language': language,
                'expected_entities': expected_entities,
                'entity_count': len(expected_entities),
            }
            failures.append(failure)

            # Update stats
            failure_stats['by_language'][language] += 1
            failure_stats['by_entity_count'][len(expected_entities)] += 1
            for entity in expected_entities:
                failure_stats['by_entity_type'][entity] += 1

    print(f"  Progress: {total}/{total} (100%)")

    # Create output database
    output = {
        'summary': {
            'total_samples': total,
            'total_failures': len(failures),
            'failure_rate': len(failures) / total if total > 0 else 0,
        },
        'failure_stats': {
            'by_language': dict(failure_stats['by_language']),
            'by_entity_type': dict(sorted(failure_stats['by_entity_type'].items(), key=lambda x: -x[1])),
            'by_entity_count': dict(sorted(failure_stats['by_entity_count'].items())),
        },
        'success_stats': {
            'by_language': dict(successes['by_language']),
            'by_entity_type': dict(sorted(successes['by_entity_type'].items(), key=lambda x: -x[1])),
        },
        'failures': failures,
    }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output


def print_analysis(analysis: Dict):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS REPORT")
    print("=" * 70)

    summary = analysis['summary']
    print(f"\nTotal samples: {summary['total_samples']}")
    print(f"Total failures: {summary['total_failures']}")
    print(f"Failure rate: {summary['failure_rate']:.1%}")

    # Language breakdown
    print("\n" + "-" * 70)
    print("FAILURES BY LANGUAGE")
    print("-" * 70)

    by_lang = analysis['failure_stats']['by_language']
    success_lang = analysis['success_stats']['by_language']

    total_by_lang = {}
    for lang in set(list(by_lang.keys()) + list(success_lang.keys())):
        total_by_lang[lang] = by_lang.get(lang, 0) + success_lang.get(lang, 0)

    for lang, fails in sorted(by_lang.items(), key=lambda x: -x[1]):
        total = total_by_lang[lang]
        rate = fails / total if total > 0 else 0
        print(f"  {lang:15} {fails:6d} failures / {total:6d} total ({rate:.1%} fail rate)")

    # Entity type breakdown
    print("\n" + "-" * 70)
    print("FAILURES BY ENTITY TYPE (top 20)")
    print("-" * 70)

    by_entity = analysis['failure_stats']['by_entity_type']
    success_entity = analysis['success_stats']['by_entity_type']

    items = list(by_entity.items())[:20]
    for entity, fails in items:
        success = success_entity.get(entity, 0)
        total = fails + success
        rate = fails / total if total > 0 else 0
        print(f"  {entity:25} {fails:5d} / {total:5d} ({rate:.1%})")

    # Sample failures
    print("\n" + "-" * 70)
    print("SAMPLE FAILURES (first 5)")
    print("-" * 70)

    for i, failure in enumerate(analysis['failures'][:5]):
        print(f"\n{i+1}. Language: {failure['language']}")
        print(f"   Expected: {', '.join(failure['expected_entities'])}")
        print(f"   Text: {failure['text'][:100]}...")

    print("\n" + "=" * 70)


def main():
    """Run comprehensive failure analysis."""
    print("=" * 70)
    print("CASCADE DLP - COMPREHENSIVE FAILURE ANALYSIS")
    print("=" * 70)

    # Initialize cascade
    cascade = DLPCascade(
        secret_threshold=0.7,
        pii_threshold=0.7,
        block_on_high_confidence=True
    )

    # Paths
    dataset_dir = Path(__file__).parent / "datasets"

    # Check for full dataset
    full_path = dataset_dir / "ai4privacy_full.json"
    sample_path = dataset_dir / "ai4privacy_sample.json"

    if full_path.exists():
        input_path = full_path
        output_name = "failure_analysis_full.json"
    elif sample_path.exists():
        input_path = sample_path
        output_name = "failure_analysis_sample.json"
    else:
        print("ERROR: No dataset found")
        return

    output_path = dataset_dir / output_name

    print(f"\nAnalyzing: {input_path.name}")
    print(f"Output: {output_path.name}")

    start_time = time.time()
    analysis = analyze_failures(cascade, str(input_path), str(output_path))
    elapsed = time.time() - start_time

    print(f"\nAnalysis completed in {elapsed/60:.1f} minutes")

    # Print report
    print_analysis(analysis)

    print(f"\nFull analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
