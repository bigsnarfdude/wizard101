#!/usr/bin/env python3
"""
Evaluate cascade on new benchmark datasets.

Datasets:
- JailbreakBench: 200 samples
- SG-Bench: 1,442 samples
- StrongREJECT: 313 samples
- WildGuardMix: 88,444 samples (subset)

Usage:
    python eval_new_benchmarks.py --benchmark jailbreakbench
    python eval_new_benchmarks.py --benchmark sgbench
    python eval_new_benchmarks.py --benchmark strongreject
    python eval_new_benchmarks.py --benchmark wildguardmix --samples 1000
    python eval_new_benchmarks.py --benchmark all --samples 500
"""

import argparse
import json
import time
import random
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
BENCHMARKS = {
    "jailbreakbench": REPO_ROOT / "data/benchmark/jailbreakbench.json",
    "sgbench": REPO_ROOT / "data/benchmark/sgbench.json",
    "strongreject": REPO_ROOT / "data/benchmark/strongreject.json",
    "wildguardmix": REPO_ROOT / "data/benchmark/wildjailbreak.json",
}
RESULTS_DIR = Path(__file__).parent / "outputs"
L0_MODEL_PATH = REPO_ROOT / "experiments/cascade/models/l0_bouncer_full"


def load_benchmark(name: str, max_samples: int = None):
    """Load benchmark dataset."""
    path = BENCHMARKS.get(name)
    if not path or not path.exists():
        raise ValueError(f"Benchmark not found: {name}")

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if max_samples and len(data) > max_samples:
        random.seed(42)
        data = random.sample(data, max_samples)

    return data


def evaluate_l0(samples):
    """Run L0 evaluation on samples."""
    print(f"\nLoading L0 model from {L0_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(L0_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(L0_MODEL_PATH)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model on: {device}")

    results = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            # Assuming index 0 = safe, index 1 = harmful
            safe_prob = probs[0][0].item()
            harmful_prob = probs[0][1].item()

            if harmful_prob > safe_prob:
                predicted = "harmful"
                confidence = harmful_prob
            else:
                predicted = "safe"
                confidence = safe_prob

        results.append({
            "text": sample["text"][:100],
            "ground_truth": sample["label"],
            "predicted": predicted,
            "confidence": confidence,
            "correct": predicted == sample["label"],
            "needs_l1": confidence < 0.9
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(samples)}")

    elapsed = time.time() - start_time
    return results, elapsed


def calculate_metrics(results):
    """Calculate evaluation metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    tp = sum(1 for r in results if r["ground_truth"] == "harmful" and r["predicted"] == "harmful")
    tn = sum(1 for r in results if r["ground_truth"] == "safe" and r["predicted"] == "safe")
    fp = sum(1 for r in results if r["ground_truth"] == "safe" and r["predicted"] == "harmful")
    fn = sum(1 for r in results if r["ground_truth"] == "harmful" and r["predicted"] == "safe")

    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    needs_l1 = sum(1 for r in results if r["needs_l1"])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "needs_l1": needs_l1,
        "needs_l1_pct": needs_l1 / total if total > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on new benchmarks")
    parser.add_argument("--benchmark", choices=list(BENCHMARKS.keys()) + ["all"],
                        default="all", help="Benchmark to evaluate")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max samples per benchmark")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.benchmark == "all":
        benchmarks = list(BENCHMARKS.keys())
    else:
        benchmarks = [args.benchmark]

    all_results = {}

    for bench_name in benchmarks:
        print("\n" + "=" * 60)
        print(f"Evaluating: {bench_name}")
        print("=" * 60)

        # Load data
        samples = load_benchmark(bench_name, args.samples)
        print(f"Loaded {len(samples)} samples")

        # Count distribution
        harmful = sum(1 for s in samples if s["label"] == "harmful")
        safe = len(samples) - harmful
        print(f"Distribution: {harmful} harmful, {safe} safe")

        # Evaluate
        results, elapsed = evaluate_l0(samples)
        metrics = calculate_metrics(results)

        # Print results
        print(f"\n{bench_name.upper()} Results:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.1f}%")
        print(f"  Precision: {metrics['precision']*100:.1f}%")
        print(f"  Recall:    {metrics['recall']*100:.1f}%")
        print(f"  F1:        {metrics['f1']*100:.1f}%")
        print(f"  Time:      {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/sec)")
        print(f"  Needs L1:  {metrics['needs_l1']} ({metrics['needs_l1_pct']*100:.1f}%)")
        print(f"  FN (dangerous): {metrics['fn']}")

        all_results[bench_name] = {
            "metrics": metrics,
            "elapsed": elapsed,
            "samples": len(samples)
        }

        # Save individual results
        output_path = RESULTS_DIR / f"{bench_name}_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "benchmark": bench_name,
                "metrics": metrics,
                "elapsed": elapsed,
                "results": results
            }, f, indent=2)
        print(f"  Saved to: {output_path}")

    # Summary
    if len(benchmarks) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n{'Benchmark':<15} {'Samples':>8} {'Acc':>8} {'F1':>8} {'FN':>6}")
        print("-" * 50)

        total_samples = 0
        total_correct = 0
        total_fn = 0

        for name, data in all_results.items():
            m = data["metrics"]
            print(f"{name:<15} {data['samples']:>8} {m['accuracy']*100:>7.1f}% {m['f1']*100:>7.1f}% {m['fn']:>6}")
            total_samples += data["samples"]
            total_correct += m["correct"]
            total_fn += m["fn"]

        print("-" * 50)
        overall_acc = total_correct / total_samples if total_samples > 0 else 0
        print(f"{'TOTAL':<15} {total_samples:>8} {overall_acc*100:>7.1f}% {'':>8} {total_fn:>6}")

    # Save summary
    summary_path = RESULTS_DIR / "new_benchmarks_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
