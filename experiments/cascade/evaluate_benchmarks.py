#!/usr/bin/env python3
"""
Evaluate Cascade on Standard Safety Benchmarks

Tests our cascade system on the same benchmarks used by GuardReasoner:
- SimpleSafetyTests
- AegisSafetyTest
- OpenAIModeration
- HarmBenchPrompt
- WildGuardTest
- HarmBenchResponse
- XSTestResponseHarmful
- XSTestResponseRefusal
- ToxicChat
- SafeRLHF
- BeaverTails

Usage:
    python evaluate_benchmarks.py --dataset all
    python evaluate_benchmarks.py --dataset wildguard
    python evaluate_benchmarks.py --dataset toxicchat --samples 100
"""

import gc
import json
import time
import argparse
import torch
from pathlib import Path
from collections import Counter
from datetime import datetime


# Dataset configurations
DATASETS = {
    "simplesafetytests": {
        "path": "../benchmark/simplesafetytests.json",
        "name": "SimpleSafetyTests",
        "text_key": "text",
        "label_key": "label",
    },
    "openai_moderation": {
        "path": "../benchmark/openai_moderation.json",
        "name": "OpenAIModeration",
        "text_key": "prompt",
        "label_key": "label",
    },
    "harmbench": {
        "path": "../benchmark/harmbench_test.json",
        "name": "HarmBench",
        "text_key": "text",
        "label_key": "label",
    },
    "toxicchat": {
        "path": "../benchmark/toxicchat_test.json",
        "name": "ToxicChat",
        "text_key": "text",
        "label_key": "label",
    },
    "beavertails": {
        "path": "../benchmark/beavertails_30k.json",
        "name": "BeaverTails",
        "text_key": "text",
        "label_key": "label",
    },
    "wildguard": {
        "path": "../wildguard_full_benchmark.json",
        "name": "WildGuard",
        "text_key": "prompt",
        "label_key": "label",
    },
    "heretic": {
        "path": "./combined_test.json",
        "name": "Heretic",
        "text_key": "text",
        "label_key": "label",
    },
}


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def normalize_label(label):
    """Normalize label to safe/harmful."""
    label = str(label).lower().strip()
    if label in ["safe", "harmless", "benign", "0", "false"]:
        return "safe"
    elif label in ["harmful", "unsafe", "malicious", "1", "true"]:
        return "harmful"
    return label


def load_dataset(dataset_name, samples=None):
    """Load and prepare a benchmark dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]
    path = Path(config["path"])

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Normalize format
    samples_list = []
    for item in data:
        text = item.get(config["text_key"], "")
        label = normalize_label(item.get(config["label_key"], ""))

        if text and label in ["safe", "harmful"]:
            samples_list.append({
                "text": text,
                "label": label,
            })

    # Limit samples if requested
    if samples and samples < len(samples_list):
        import random
        random.seed(42)
        samples_list = random.sample(samples_list, samples)

    return samples_list, config["name"]


def run_cascade_evaluation(samples, l0_threshold=0.7, l1_threshold=0.7, skip_l2=False, skip_l3=True):
    """Run cascade evaluation on samples."""
    results = []

    # Stage 1: L0
    print("\n" + "="*50)
    print("L0 BOUNCER")
    print("="*50)

    from l0_bouncer import L0Bouncer
    l0 = L0Bouncer()

    l0_results = []
    l0_uncertain_idx = []

    start = time.time()
    for idx, sample in enumerate(samples):
        result = l0.classify(sample["text"])
        l0_results.append({
            "idx": idx,
            "text": sample["text"],
            "expected": sample["label"],
            "l0_label": result["label"],
            "l0_confidence": result["confidence"],
        })
        if result["confidence"] < l0_threshold:
            l0_uncertain_idx.append(idx)

        if (idx + 1) % 100 == 0:
            print(f"  L0: {idx + 1}/{len(samples)}")

    l0_time = time.time() - start
    l0_correct = sum(1 for r in l0_results if r["l0_label"] == r["expected"])
    print(f"L0: {l0_correct}/{len(samples)} = {l0_correct/len(samples)*100:.1f}% ({l0_time:.1f}s)")
    print(f"  Escalating: {len(l0_uncertain_idx)}")

    del l0
    clear_gpu()

    # Stage 2: L1
    if l0_uncertain_idx:
        print("\n" + "="*50)
        print("L1 ANALYST")
        print("="*50)

        from l1_analyst import L1Analyst
        l1 = L1Analyst()

        l1_uncertain_idx = []

        start = time.time()
        for i, idx in enumerate(l0_uncertain_idx):
            result = l1.analyze(l0_results[idx]["text"])
            l0_results[idx]["l1_label"] = result["label"]
            l0_results[idx]["l1_confidence"] = result["confidence"]

            if result["confidence"] < l1_threshold:
                l1_uncertain_idx.append(idx)

            if (i + 1) % 20 == 0:
                print(f"  L1: {i + 1}/{len(l0_uncertain_idx)}")

        l1_time = time.time() - start
        l1_samples = [l0_results[idx] for idx in l0_uncertain_idx]
        l1_correct = sum(1 for r in l1_samples if r.get("l1_label") == r["expected"])
        print(f"L1: {l1_correct}/{len(l0_uncertain_idx)} = {l1_correct/len(l0_uncertain_idx)*100:.1f}% ({l1_time:.1f}s)")
        print(f"  Escalating: {len(l1_uncertain_idx)}")

        del l1
        clear_gpu()
    else:
        l1_uncertain_idx = []

    # Stage 3: L2 (optional)
    if l1_uncertain_idx and not skip_l2:
        print("\n" + "="*50)
        print("L2 GAUNTLET")
        print("="*50)

        from l2_gauntlet import L2Gauntlet
        l2 = L2Gauntlet()

        start = time.time()
        for i, idx in enumerate(l1_uncertain_idx):
            result = l2.analyze(l0_results[idx]["text"])
            l0_results[idx]["l2_label"] = result["label"]
            l0_results[idx]["l2_votes"] = result["votes"]

            if (i + 1) % 10 == 0:
                print(f"  L2: {i + 1}/{len(l1_uncertain_idx)}")

        l2_time = time.time() - start
        l2_samples = [l0_results[idx] for idx in l1_uncertain_idx]
        l2_correct = sum(1 for r in l2_samples if r.get("l2_label") == r["expected"])
        print(f"L2: {l2_correct}/{len(l1_uncertain_idx)} = {l2_correct/len(l1_uncertain_idx)*100:.1f}% ({l2_time:.1f}s)")

        del l2

    # Compute final labels
    for r in l0_results:
        if "l2_label" in r:
            r["final_label"] = r["l2_label"]
            r["stopped_at"] = "L2"
        elif "l1_label" in r:
            r["final_label"] = r["l1_label"]
            r["stopped_at"] = "L1"
        else:
            r["final_label"] = r["l0_label"]
            r["stopped_at"] = "L0"

    return l0_results


def compute_metrics(results):
    """Compute evaluation metrics."""
    correct = sum(1 for r in results if r["final_label"] == r["expected"])
    accuracy = correct / len(results) * 100 if results else 0

    tp = sum(1 for r in results if r["final_label"] == "harmful" and r["expected"] == "harmful")
    tn = sum(1 for r in results if r["final_label"] == "safe" and r["expected"] == "safe")
    fp = sum(1 for r in results if r["final_label"] == "harmful" and r["expected"] == "safe")
    fn = sum(1 for r in results if r["final_label"] == "safe" and r["expected"] == "harmful")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    layer_dist = Counter(r["stopped_at"] for r in results)

    return {
        "accuracy": accuracy,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "correct": correct,
        "total": len(results),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "layer_distribution": dict(layer_dist),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate cascade on standard benchmarks")
    parser.add_argument("--dataset", type=str, default="all",
                        help=f"Dataset to evaluate: {list(DATASETS.keys())} or 'all'")
    parser.add_argument("--samples", type=int, default=None, help="Limit samples per dataset")
    parser.add_argument("--l0-threshold", type=float, default=0.7)
    parser.add_argument("--l1-threshold", type=float, default=0.7)
    parser.add_argument("--skip-l2", action="store_true", help="Skip L2 gauntlet")
    args = parser.parse_args()

    # Determine which datasets to evaluate
    if args.dataset == "all":
        dataset_names = list(DATASETS.keys())
    else:
        dataset_names = [args.dataset]

    all_results = {}

    for dataset_name in dataset_names:
        print("\n" + "="*60)
        print(f"EVALUATING: {dataset_name.upper()}")
        print("="*60)

        try:
            samples, display_name = load_dataset(dataset_name, args.samples)
            print(f"Loaded {len(samples)} samples")

            # Count distribution
            dist = Counter(s["label"] for s in samples)
            print(f"  Safe: {dist['safe']}, Harmful: {dist['harmful']}")

            # Run evaluation
            results = run_cascade_evaluation(
                samples,
                args.l0_threshold,
                args.l1_threshold,
                args.skip_l2
            )

            # Compute metrics
            metrics = compute_metrics(results)

            print(f"\n{'='*50}")
            print(f"{display_name} RESULTS")
            print(f"{'='*50}")
            print(f"Accuracy: {metrics['accuracy']:.1f}%")
            print(f"Precision: {metrics['precision']:.1f}%")
            print(f"Recall: {metrics['recall']:.1f}%")
            print(f"F1: {metrics['f1']:.1f}%")
            print(f"Layer distribution: {metrics['layer_distribution']}")

            all_results[dataset_name] = {
                "name": display_name,
                "samples": len(samples),
                "metrics": metrics,
            }

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Summary
    if len(dataset_names) > 1:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        for name, result in all_results.items():
            if "error" in result:
                print(f"{name}: ERROR - {result['error']}")
            else:
                m = result["metrics"]
                print(f"{result['name']}: Acc={m['accuracy']:.1f}% F1={m['f1']:.1f}% ({result['samples']} samples)")

    # Save results
    output_path = Path(f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "l0_threshold": args.l0_threshold,
                "l1_threshold": args.l1_threshold,
                "skip_l2": args.skip_l2,
                "samples_limit": args.samples,
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
