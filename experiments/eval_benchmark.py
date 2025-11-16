#!/usr/bin/env python3
"""
Evaluate Serial Gauntlet on Academic Benchmark

Runs academic_benchmark.json through the serial gauntlet and calculates:
- Overall accuracy
- Per-policy F1 scores
- False positive rate (XSTest safe prompts)
- False negative rate (SimpleSafetyTests unsafe prompts)
- Multi-policy accuracy (compare to OpenAI's 43.6%)

Usage:
    python eval_benchmark.py [benchmark_file]
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import defaultdict

# Import from serial_gauntlet_simple
from serial_gauntlet_simple import load_policies, check_one_policy


@dataclass
class EvalResult:
    """Evaluation result for one sample"""
    content: str
    true_labels: Set[str]
    predicted_labels: Set[str]
    source: str
    true_label: str  # 'safe' or 'unsafe' or 'toxic'
    policy_results: Dict[str, bool]  # {policy_name: violation}
    latency_ms: float


def load_benchmark(filepath: str) -> List[Dict]:
    """Load academic benchmark JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {filepath}")
    return data


def evaluate_sample(
    content: str,
    policies: Dict[str, str],
    model: str = "gpt-oss:20b"
) -> Dict[str, any]:
    """
    Run one sample through gauntlet

    Returns:
        {
            'predicted_labels': Set[str],
            'policy_results': {policy_name: violation_bool},
            'latency_ms': float
        }
    """
    start = time.time()

    policy_results = {}
    predicted_labels = set()

    for policy_name, policy_text in policies.items():
        result = check_one_policy(content, policy_name, policy_text, model)

        policy_results[policy_name] = result.violation

        if result.violation:
            predicted_labels.add(policy_name)

    latency_ms = (time.time() - start) * 1000

    return {
        'predicted_labels': predicted_labels,
        'policy_results': policy_results,
        'latency_ms': latency_ms
    }


def calculate_metrics(results: List[EvalResult]) -> Dict[str, any]:
    """
    Calculate evaluation metrics

    Metrics:
    - Overall accuracy (any policy match)
    - Per-policy F1 scores
    - False positive rate (safe → flagged)
    - False negative rate (unsafe → missed)
    - Multi-policy accuracy (exact match all policies)
    """

    metrics = {
        'total_samples': len(results),
        'overall': {},
        'per_policy': {},
        'by_source': {},
        'multi_policy': {},
    }

    # Overall metrics (any policy)
    correct = 0
    false_positives = 0
    false_negatives = 0

    # Multi-policy metrics (exact match)
    exact_match = 0

    # Per-policy metrics
    policy_names = ['hate_speech', 'violence', 'self_harm', 'sexual_minors', 'harassment', 'illegal']
    policy_stats = {name: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for name in policy_names}

    # By source metrics
    source_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'fp': 0, 'fn': 0})

    for result in results:
        # Overall: is sample correctly classified as safe/unsafe?
        sample_is_unsafe = len(result.true_labels) > 0
        sample_flagged = len(result.predicted_labels) > 0

        if sample_is_unsafe == sample_flagged:
            correct += 1

        if not sample_is_unsafe and sample_flagged:
            false_positives += 1

        if sample_is_unsafe and not sample_flagged:
            false_negatives += 1

        # Multi-policy: exact match of all labels
        if result.true_labels == result.predicted_labels:
            exact_match += 1

        # Per-policy: track TP, FP, TN, FN
        for policy_name in policy_names:
            true_violation = policy_name in result.true_labels
            pred_violation = policy_name in result.predicted_labels

            if true_violation and pred_violation:
                policy_stats[policy_name]['tp'] += 1
            elif not true_violation and pred_violation:
                policy_stats[policy_name]['fp'] += 1
            elif not true_violation and not pred_violation:
                policy_stats[policy_name]['tn'] += 1
            elif true_violation and not pred_violation:
                policy_stats[policy_name]['fn'] += 1

        # By source
        source = result.source
        source_stats[source]['total'] += 1
        if sample_is_unsafe == sample_flagged:
            source_stats[source]['correct'] += 1
        if not sample_is_unsafe and sample_flagged:
            source_stats[source]['fp'] += 1
        if sample_is_unsafe and not sample_flagged:
            source_stats[source]['fn'] += 1

    # Calculate overall metrics
    total = len(results)
    metrics['overall'] = {
        'accuracy': correct / total if total > 0 else 0,
        'false_positive_rate': false_positives / total if total > 0 else 0,
        'false_negative_rate': false_negatives / total if total > 0 else 0,
    }

    # Calculate per-policy F1 scores
    for policy_name, stats in policy_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics['per_policy'][policy_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }

    # Calculate by-source accuracy
    for source, stats in source_stats.items():
        total_src = stats['total']
        metrics['by_source'][source] = {
            'accuracy': stats['correct'] / total_src if total_src > 0 else 0,
            'samples': total_src,
            'fp_rate': stats['fp'] / total_src if total_src > 0 else 0,
            'fn_rate': stats['fn'] / total_src if total_src > 0 else 0,
        }

    # Multi-policy accuracy
    metrics['multi_policy']['exact_match_accuracy'] = exact_match / total if total > 0 else 0

    return metrics


def print_metrics(metrics: Dict[str, any]):
    """Pretty print metrics"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print(f"\nTotal Samples: {metrics['total_samples']}")

    print("\n" + "-"*80)
    print("OVERALL METRICS")
    print("-"*80)
    print(f"Accuracy (safe/unsafe):        {metrics['overall']['accuracy']:.1%}")
    print(f"False Positive Rate:           {metrics['overall']['false_positive_rate']:.1%}")
    print(f"False Negative Rate:           {metrics['overall']['false_negative_rate']:.1%}")

    print("\n" + "-"*80)
    print("MULTI-POLICY ACCURACY (Exact Match)")
    print("-"*80)
    print(f"Exact Match Accuracy:          {metrics['multi_policy']['exact_match_accuracy']:.1%}")
    print(f"  (Compare to OpenAI gpt-oss-safeguard-20b: 43.6%)")

    print("\n" + "-"*80)
    print("PER-POLICY F1 SCORES")
    print("-"*80)
    print(f"{'Policy':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)

    for policy_name, stats in metrics['per_policy'].items():
        print(f"{policy_name:<20} {stats['precision']:<12.1%} {stats['recall']:<12.1%} {stats['f1']:<12.1%}")

    print("\n" + "-"*80)
    print("BY-SOURCE BREAKDOWN")
    print("-"*80)
    print(f"{'Source':<25} {'Samples':<10} {'Accuracy':<12} {'FP Rate':<12} {'FN Rate':<12}")
    print("-"*80)

    for source, stats in metrics['by_source'].items():
        print(f"{source:<25} {stats['samples']:<10} {stats['accuracy']:<12.1%} {stats['fp_rate']:<12.1%} {stats['fn_rate']:<12.1%}")

    print("\n" + "="*80)


def main():
    # Get benchmark file
    if len(sys.argv) > 1:
        benchmark_file = sys.argv[1]
    else:
        benchmark_file = "academic_benchmark.json"

    benchmark_path = Path(__file__).parent / benchmark_file

    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    # Load benchmark
    benchmark = load_benchmark(str(benchmark_path))

    # Load policies
    print("\nLoading policies...")
    policies = load_policies()
    print(f"Loaded {len(policies)} policies")

    # Run evaluation
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    print(f"Model: gpt-oss:20b")
    print(f"Samples: {len(benchmark)}")
    print(f"Estimated time: ~{len(benchmark) * 12} seconds ({len(benchmark) * 12 / 60:.1f} minutes)")
    print("="*80)

    results = []

    for i, sample in enumerate(benchmark, 1):
        content = sample['content']
        true_labels = set(sample['labels'])
        source = sample['source']
        true_label = sample.get('true_label', 'unknown')

        print(f"\n[{i}/{len(benchmark)}] {source}: {content[:50]}...")

        # Run through gauntlet
        eval_result = evaluate_sample(content, policies)

        # Store result
        result = EvalResult(
            content=content,
            true_labels=true_labels,
            predicted_labels=eval_result['predicted_labels'],
            source=source,
            true_label=true_label,
            policy_results=eval_result['policy_results'],
            latency_ms=eval_result['latency_ms']
        )
        results.append(result)

        # Show prediction
        pred = eval_result['predicted_labels']
        true = true_labels
        match = "✓" if pred == true else "✗"
        print(f"  Predicted: {pred if pred else 'SAFE'}")
        print(f"  True:      {true if true else 'SAFE'} {match}")
        print(f"  Time:      {eval_result['latency_ms']:.0f}ms")

    # Calculate metrics
    print("\n\nCalculating metrics...")
    metrics = calculate_metrics(results)

    # Print results
    print_metrics(metrics)

    # Save results
    output_file = benchmark_path.parent / "eval_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'results': [
                {
                    'content': r.content,
                    'true_labels': list(r.true_labels),
                    'predicted_labels': list(r.predicted_labels),
                    'source': r.source,
                    'match': r.true_labels == r.predicted_labels,
                }
                for r in results
            ]
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
