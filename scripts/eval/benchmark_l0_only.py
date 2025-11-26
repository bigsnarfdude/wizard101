#!/usr/bin/env python3
"""
L0-only benchmark for fast evaluation.

Evaluates just the L0 Bouncer without loading L1/L2/L3.
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cascade_inbound.l0_bouncer import L0Bouncer
from tqdm import tqdm


BENCHMARKS = [
    ("simplesafetytests.json", "SimpleSafetyTests", 100),
    ("jailbreakbench.json", "JailbreakBench", 200),
    ("strongreject.json", "StrongREJECT", 313),
    ("harmbench_test.json", "HarmBench", 500),
    ("sgbench.json", "SGBench", 1442),
    ("openai_moderation.json", "OpenAI Moderation", 1680),
    ("beavertails_30k.json", "BeaverTails", 3021),
    ("toxicchat_test.json", "ToxicChat", 5083),
    ("salad_bench_attack.json", "SALAD-Bench Attack", 5000),
    ("salad_bench_base.json", "SALAD-Bench Base", 21318),
    ("or_bench.json", "OR-Bench", 82333),
    ("combined_benchmark.json", "Combined", 10384),
]


def normalize_label(label):
    """Normalize label to safe/harmful."""
    if label is None:
        return "harmful"
    label = str(label).lower().strip()
    if label in ['safe', 'benign', 'unharmful', 'harmless', '0', 'false']:
        return 'safe'
    return 'harmful'


def get_prompt(sample):
    """Extract prompt from sample."""
    for key in ['prompt', 'text', 'question', 'input', 'content']:
        if key in sample and sample[key]:
            return sample[key]
    return None


def evaluate_benchmark(bouncer, benchmark_path, name):
    """Evaluate L0 bouncer on a single benchmark."""

    with open(benchmark_path) as f:
        data = json.load(f)

    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    latencies = []
    errors = []

    for i, sample in enumerate(tqdm(data, desc=name, leave=False)):
        prompt = get_prompt(sample)
        if not prompt:
            errors.append(f"Sample {i}: No prompt found")
            continue

        true_label = normalize_label(sample.get("label"))

        try:
            start = time.time()
            result = bouncer.classify(prompt)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            pred_label = result["label"]

            if true_label == "harmful" and pred_label == "harmful":
                results["tp"] += 1
            elif true_label == "safe" and pred_label == "harmful":
                results["fp"] += 1
            elif true_label == "safe" and pred_label == "safe":
                results["tn"] += 1
            else:
                results["fn"] += 1

        except Exception as e:
            errors.append(f"Sample {i}: {str(e)[:50]}")

    tp, fp, tn, fn = results["tp"], results["fp"], results["tn"], results["fn"]
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "name": name,
        "file": str(benchmark_path.name),
        "samples": len(data),
        "evaluated": total,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "fpr": round(fpr * 100, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "confusion": results,
        "errors": len(errors)
    }


def main():
    parser = argparse.ArgumentParser(description="L0-only benchmark")
    parser.add_argument("--dataset", type=str, help="Run single dataset")
    args = parser.parse_args()

    print("Loading L0 Bouncer...")
    bouncer = L0Bouncer()

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    benchmark_dir = repo_root / "data" / "benchmark"

    benchmarks = BENCHMARKS
    if args.dataset:
        benchmarks = [(f, n, s) for f, n, s in BENCHMARKS if args.dataset in f or args.dataset.lower() in n.lower()]
        if not benchmarks:
            print(f"Dataset '{args.dataset}' not found.")
            sys.exit(1)

    results = []
    start_time = datetime.now()

    print(f"\n{'='*70}")
    print(f"L0-ONLY BENCHMARK - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for filename, name, expected_samples in benchmarks:
        path = benchmark_dir / filename
        if not path.exists():
            print(f"Skipping {name}: {filename} not found")
            continue

        print(f"\nEvaluating {name} ({expected_samples:,} samples)...")
        result = evaluate_benchmark(bouncer, path, name)
        results.append(result)

        print(f"  Accuracy: {result['accuracy']:.1f}%  |  "
              f"Precision: {result['precision']:.1f}%  |  "
              f"Recall: {result['recall']:.1f}%  |  "
              f"F1: {result['f1']:.1f}%  |  "
              f"FPR: {result['fpr']:.1f}%  |  "
              f"Latency: {result['avg_latency_ms']:.1f}ms")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Benchmark':<25} {'Samples':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7}")
    print("-"*70)

    total_samples = 0
    for r in results:
        print(f"{r['name']:<25} {r['evaluated']:>8,} {r['accuracy']:>6.1f}% "
              f"{r['precision']:>6.1f}% {r['recall']:>6.1f}% {r['f1']:>6.1f}% {r['fpr']:>6.1f}%")
        total_samples += r['evaluated']

    print("-"*70)
    print(f"{'TOTAL':<25} {total_samples:>8,}")
    print(f"\nDuration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Throughput: {total_samples/duration:.1f} samples/sec")

    # Save results
    output_dir = repo_root / "experiments"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"l0_benchmark_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "model": "L0 Bouncer only",
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
