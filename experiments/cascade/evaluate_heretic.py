#!/usr/bin/env python3
"""
Evaluate Cascade on Heretic Dataset

Heretic is a jailbreak-focused adversarial dataset with 1000 samples.
Tests cascade robustness on adversarial prompts.

Usage:
    python evaluate_heretic.py [--batch] [--full]

    --full: Use all 1000 samples (train + test)
    --batch: Memory-efficient batch mode
"""

import gc
import json
import time
import argparse
import torch
from pathlib import Path
from collections import Counter


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_l0_batch(samples, threshold=0.7):
    """Run L0 on all samples."""
    print("\n" + "="*60)
    print("STAGE 1: L0 BOUNCER")
    print("="*60)

    from l0_bouncer import L0Bouncer
    l0 = L0Bouncer()

    caught = []
    uncertain = []

    for idx, sample in enumerate(samples):
        text = sample.get("text", "")
        result = l0.classify(text)

        if result["confidence"] >= threshold:
            caught.append((idx, {
                "text": text,
                "expected": sample["label"],
                "final_label": result["label"],
                "stopped_at": "L0",
                "confidence": result["confidence"],
            }))
        else:
            uncertain.append((idx, sample))

        if (idx + 1) % 50 == 0:
            print(f"  L0 processed {idx + 1}/{len(samples)}")

    print(f"\nL0 Results:")
    print(f"  Caught: {len(caught)} ({len(caught)/len(samples)*100:.1f}%)")
    print(f"  Uncertain: {len(uncertain)} ({len(uncertain)/len(samples)*100:.1f}%)")

    del l0
    clear_gpu()
    return caught, uncertain


def run_l1_batch(uncertain_samples, threshold=0.7):
    """Run L1 on uncertain samples."""
    if not uncertain_samples:
        return [], []

    print("\n" + "="*60)
    print("STAGE 2: L1 ANALYST")
    print("="*60)

    from l1_analyst import L1Analyst
    l1 = L1Analyst()

    caught = []
    uncertain = []

    for idx, sample in uncertain_samples:
        text = sample.get("text", "")
        result = l1.analyze(text)

        if result["confidence"] >= threshold:
            caught.append((idx, {
                "text": text,
                "expected": sample["label"],
                "final_label": result["label"],
                "stopped_at": "L1",
                "confidence": result["confidence"],
            }))
        else:
            uncertain.append((idx, sample, result))

    print(f"\nL1 Results:")
    print(f"  Caught: {len(caught)} ({len(caught)/len(uncertain_samples)*100:.1f}%)")
    print(f"  Uncertain: {len(uncertain)} ({len(uncertain)/len(uncertain_samples)*100:.1f}%)")

    del l1
    clear_gpu()
    return caught, uncertain


def run_l2_batch(uncertain_samples):
    """Run L2 on remaining samples."""
    if not uncertain_samples:
        return []

    print("\n" + "="*60)
    print("STAGE 3: L2 GAUNTLET")
    print("="*60)

    from l2_gauntlet import L2Gauntlet
    l2 = L2Gauntlet()

    results = []

    for idx, sample, l1_result in uncertain_samples:
        text = sample.get("text", "")
        result = l2.analyze(text)

        results.append((idx, {
            "text": text,
            "expected": sample["label"],
            "final_label": result["label"],
            "stopped_at": "L2",
            "confidence": 0.9 if result["consensus"] else 0.7,
        }))

    print(f"\nL2 Results: {len(results)} samples processed")
    del l2
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate cascade on Heretic dataset")
    parser.add_argument("--batch", action="store_true", help="Use batch mode (memory efficient)")
    parser.add_argument("--full", action="store_true", help="Use full dataset (train + test = 1000 samples)")
    parser.add_argument("--l0-threshold", type=float, default=0.7, help="L0 confidence threshold")
    parser.add_argument("--l1-threshold", type=float, default=0.7, help="L1 confidence threshold")
    args = parser.parse_args()

    # Load Heretic data
    test_path = Path("../combined_test.json")
    train_path = Path("../combined_train.json")

    if not test_path.exists():
        print(f"Heretic test data not found at {test_path}")
        return

    if args.full:
        if not train_path.exists():
            print(f"Heretic train data not found at {train_path}")
            return
        print(f"Loading FULL Heretic dataset (train + test)")
        with open(train_path) as f:
            train_data = json.load(f)
        with open(test_path) as f:
            test_data = json.load(f)
        test_data = train_data + test_data
        print(f"  Train: {len(train_data)} + Test: {len(test_data) - len(train_data)} = {len(test_data)} total")
    else:
        print(f"Loading Heretic test set: {test_path}")
        with open(test_path) as f:
            test_data = json.load(f)

    # Normalize labels: harmless -> safe, harmful -> harmful
    for item in test_data:
        if item["label"] == "harmless":
            item["label"] = "safe"

    # Count distribution
    label_dist = Counter(item["label"] for item in test_data)
    print(f"\nHeretic Test Set: {len(test_data)} samples")
    print(f"  Safe: {label_dist['safe']}")
    print(f"  Harmful: {label_dist['harmful']}")

    print(f"\nL0 threshold: {args.l0_threshold}")
    print(f"L1 threshold: {args.l1_threshold}")

    start_time = time.time()

    if args.batch:
        # Batch mode - memory efficient
        l0_caught, l0_uncertain = run_l0_batch(test_data, args.l0_threshold)
        l1_caught, l1_uncertain = run_l1_batch(l0_uncertain, args.l1_threshold)
        l2_results = run_l2_batch(l1_uncertain)

        # Aggregate
        all_results = {}
        for idx, result in l0_caught:
            all_results[idx] = result
        for idx, result in l1_caught:
            all_results[idx] = result
        for idx, result in l2_results:
            all_results[idx] = result

        results = [all_results[i] for i in sorted(all_results.keys())]
    else:
        # Standard mode - all models loaded
        from cascade import SafetyCascade, CascadeConfig

        config = CascadeConfig(
            l0_confidence_threshold=args.l0_threshold,
            l1_confidence_threshold=args.l1_threshold,
            enable_l2=True,
            enable_l3=False,
        )
        cascade = SafetyCascade(config)

        results = []
        for i, item in enumerate(test_data):
            text = item.get("text", "")
            expected = item["label"]

            result = cascade.classify(text)

            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "expected": expected,
                "final_label": result.label,
                "stopped_at": result.stopped_at,
                "confidence": result.confidence,
            })

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}")

    total_time = time.time() - start_time

    # Calculate metrics
    print("\n" + "="*60)
    print("HERETIC EVALUATION RESULTS")
    print("="*60)

    # Layer distribution
    layer_counts = Counter(r["stopped_at"] for r in results)
    print("\nLayer Distribution:")
    for layer in ["L0", "L1", "L2", "L3"]:
        count = layer_counts.get(layer, 0)
        pct = count / len(results) * 100 if results else 0
        print(f"  {layer}: {count} ({pct:.1f}%)")

    # Accuracy
    correct = sum(1 for r in results if r["final_label"] == r["expected"])
    accuracy = correct / len(results) * 100 if results else 0
    print(f"\nOverall Accuracy: {correct}/{len(results)} = {accuracy:.1f}%")

    # Confusion matrix
    tp = sum(1 for r in results if r["final_label"] == "harmful" and r["expected"] == "harmful")
    tn = sum(1 for r in results if r["final_label"] == "safe" and r["expected"] == "safe")
    fp = sum(1 for r in results if r["final_label"] == "harmful" and r["expected"] == "safe")
    fn = sum(1 for r in results if r["final_label"] == "safe" and r["expected"] == "harmful")

    print("\nConfusion Matrix:")
    print(f"  TP (caught harmful): {tp}")
    print(f"  TN (allowed safe): {tn}")
    print(f"  FP (blocked safe): {fp}")
    print(f"  FN (missed harmful): {fn}")

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMetrics:")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  F1: {f1*100:.1f}%")

    print(f"\nTotal Time: {total_time:.1f}s ({total_time/len(results)*1000:.1f}ms/sample)")

    # Save results
    output_path = Path("heretic_eval_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "dataset": "Heretic",
            "samples": len(results),
            "config": {
                "l0_threshold": args.l0_threshold,
                "l1_threshold": args.l1_threshold,
                "batch_mode": args.batch,
            },
            "summary": {
                "accuracy": accuracy,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1": f1 * 100,
                "total_time_s": total_time,
            },
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "layer_distribution": dict(layer_counts),
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Show errors
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    # False negatives (missed harmful)
    fn_samples = [r for r in results if r["final_label"] == "safe" and r["expected"] == "harmful"][:3]
    if fn_samples:
        print("\nFalse Negatives (Missed Harmful):")
        for r in fn_samples:
            text = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
            print(f"  - {text}")

    # False positives (blocked safe)
    fp_samples = [r for r in results if r["final_label"] == "harmful" and r["expected"] == "safe"][:3]
    if fp_samples:
        print("\nFalse Positives (Blocked Safe):")
        for r in fp_samples:
            text = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
            print(f"  - {text}")


if __name__ == "__main__":
    main()
