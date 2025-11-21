#!/usr/bin/env python3
"""
Full Cascade Evaluation on Heretic Dataset

Processes each layer separately with proper GPU/Ollama unloading.
All L1 and L2 decisions get audited by L3 judge.

Architecture:
1. L0 processes all 1000 → score, unload
2. L1 processes uncertain → score, unload
3. L2 processes uncertain → score, wait 5 min for Ollama unload
4. L3 audits ALL L1+L2 decisions → final scores

Usage:
    python evaluate_heretic_full.py [--samples N] [--ollama-wait 300]
"""

import gc
import json
import time
import argparse
import torch
from pathlib import Path
from collections import Counter
from datetime import datetime


def clear_gpu():
    """Clear GPU memory completely."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force reset GPU memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    gc.collect()


def unload_ollama_models():
    """Force Ollama to unload all models from VRAM."""
    import requests
    try:
        # Get list of loaded models
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return

        # Try to unload by making a dummy request that forces model swap
        # This is a hack - Ollama doesn't have explicit unload API
        print("  Requesting Ollama to release VRAM...")

        # Set keep_alive to 0 to unload immediately after use
        requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gpt-oss:20b",
                "prompt": "",
                "keep_alive": 0,  # Unload immediately
            },
            timeout=30
        )
        print("  Ollama models set to unload")
        time.sleep(5)  # Give Ollama time to unload
    except Exception as e:
        print(f"  Could not unload Ollama: {e}")


def wait_for_ollama_unload(seconds=300):
    """Wait for Ollama to unload model from VRAM."""
    print(f"\n⏳ Waiting {seconds}s for Ollama to unload model from VRAM...")
    for i in range(seconds // 30):
        time.sleep(30)
        elapsed = (i + 1) * 30
        remaining = seconds - elapsed
        print(f"  {elapsed}s elapsed, {remaining}s remaining...")
    remaining = seconds % 30
    if remaining > 0:
        time.sleep(remaining)
    print("  ✓ Ollama unload wait complete\n")


def run_l0_batch(samples, threshold=0.7):
    """
    Stage 1: L0 Bouncer on all samples.
    """
    print("\n" + "="*60)
    print("STAGE 1: L0 BOUNCER (DeBERTa)")
    print("="*60)

    from l0_bouncer import L0Bouncer
    l0 = L0Bouncer()

    l0_results = []
    l0_uncertain_idx = []

    start = time.time()
    for idx, sample in enumerate(samples):
        text = sample.get("text", "")
        result = l0.classify(text)

        l0_results.append({
            "idx": idx,
            "text": text,
            "expected": sample["label"],
            "l0_label": result["label"],
            "l0_confidence": result["confidence"],
            "l0_safe_prob": result.get("safe_prob", 0),
        })

        # Track uncertain for L1
        if result["confidence"] < threshold:
            l0_uncertain_idx.append(idx)

        if (idx + 1) % 100 == 0:
            print(f"  L0 processed {idx + 1}/{len(samples)}")

    l0_time = time.time() - start

    # Score L0
    l0_correct = sum(1 for r in l0_results if r["l0_label"] == r["expected"])
    l0_confident = [r for r in l0_results if r["l0_confidence"] >= threshold]
    l0_confident_correct = sum(1 for r in l0_confident if r["l0_label"] == r["expected"])

    print(f"\nL0 Results ({l0_time:.1f}s):")
    print(f"  Total accuracy: {l0_correct}/{len(samples)} = {l0_correct/len(samples)*100:.1f}%")
    print(f"  Confident decisions: {len(l0_confident)} ({len(l0_confident)/len(samples)*100:.1f}%)")
    print(f"  Confident accuracy: {l0_confident_correct}/{len(l0_confident)} = {l0_confident_correct/len(l0_confident)*100:.1f}%" if l0_confident else "  No confident decisions")
    print(f"  Escalating to L1: {len(l0_uncertain_idx)} samples")

    # Unload L0
    del l0
    clear_gpu()
    print("  ✓ L0 unloaded from GPU")

    return l0_results, l0_uncertain_idx, {
        "time": l0_time,
        "total": len(samples),
        "correct": l0_correct,
        "confident": len(l0_confident),
        "confident_correct": l0_confident_correct,
        "escalated": len(l0_uncertain_idx),
    }


def run_l1_batch(l0_results, uncertain_idx, threshold=0.7):
    """
    Stage 2: L1 Analyst on uncertain samples.
    """
    if not uncertain_idx:
        return l0_results, [], {"time": 0, "total": 0, "correct": 0, "confident": 0, "escalated": 0}

    print("\n" + "="*60)
    print("STAGE 2: L1 ANALYST (Llama 3.2 3B)")
    print("="*60)

    # Force unload any Ollama models taking VRAM
    unload_ollama_models()
    clear_gpu()

    from l1_analyst import L1Analyst
    l1 = L1Analyst()

    l1_uncertain_idx = []

    start = time.time()
    for i, idx in enumerate(uncertain_idx):
        text = l0_results[idx]["text"]
        result = l1.analyze(text)

        l0_results[idx]["l1_label"] = result["label"]
        l0_results[idx]["l1_confidence"] = result["confidence"]
        l0_results[idx]["l1_reasoning"] = result["reasoning"][:200]

        # Track uncertain for L2
        if result["confidence"] < threshold:
            l1_uncertain_idx.append(idx)

        if (i + 1) % 20 == 0:
            print(f"  L1 processed {i + 1}/{len(uncertain_idx)}")

    l1_time = time.time() - start

    # Score L1
    l1_samples = [l0_results[idx] for idx in uncertain_idx]
    l1_correct = sum(1 for r in l1_samples if r.get("l1_label") == r["expected"])
    l1_confident = [r for r in l1_samples if r.get("l1_confidence", 0) >= threshold]
    l1_confident_correct = sum(1 for r in l1_confident if r["l1_label"] == r["expected"])

    print(f"\nL1 Results ({l1_time:.1f}s):")
    print(f"  Total accuracy: {l1_correct}/{len(uncertain_idx)} = {l1_correct/len(uncertain_idx)*100:.1f}%")
    print(f"  Confident decisions: {len(l1_confident)} ({len(l1_confident)/len(uncertain_idx)*100:.1f}%)")
    print(f"  Confident accuracy: {l1_confident_correct}/{len(l1_confident)} = {l1_confident_correct/len(l1_confident)*100:.1f}%" if l1_confident else "  No confident decisions")
    print(f"  Escalating to L2: {len(l1_uncertain_idx)} samples")

    # Unload L1
    del l1
    clear_gpu()
    print("  ✓ L1 unloaded from GPU")

    return l0_results, l1_uncertain_idx, {
        "time": l1_time,
        "total": len(uncertain_idx),
        "correct": l1_correct,
        "confident": len(l1_confident),
        "confident_correct": l1_confident_correct,
        "escalated": len(l1_uncertain_idx),
    }


def run_l2_batch(results, uncertain_idx, ollama_wait=300):
    """
    Stage 3: L2 Gauntlet on uncertain samples.
    """
    if not uncertain_idx:
        return results, {"time": 0, "total": 0, "correct": 0}

    print("\n" + "="*60)
    print("STAGE 3: L2 GAUNTLET (6 Experts, gpt-oss:20b)")
    print("="*60)

    from l2_gauntlet import L2Gauntlet
    l2 = L2Gauntlet()

    start = time.time()
    for i, idx in enumerate(uncertain_idx):
        text = results[idx]["text"]
        result = l2.analyze(text)

        results[idx]["l2_label"] = result["label"]
        results[idx]["l2_votes"] = result["votes"]
        results[idx]["l2_consensus"] = result["consensus"]
        results[idx]["l2_harmful_experts"] = result["harmful_experts"]

        if (i + 1) % 10 == 0:
            print(f"  L2 processed {i + 1}/{len(uncertain_idx)}")

    l2_time = time.time() - start

    # Score L2
    l2_samples = [results[idx] for idx in uncertain_idx]
    l2_correct = sum(1 for r in l2_samples if r.get("l2_label") == r["expected"])

    print(f"\nL2 Results ({l2_time:.1f}s):")
    print(f"  Total accuracy: {l2_correct}/{len(uncertain_idx)} = {l2_correct/len(uncertain_idx)*100:.1f}%")

    # Unload L2 - wait for Ollama
    del l2
    if ollama_wait > 0:
        wait_for_ollama_unload(ollama_wait)

    return results, {
        "time": l2_time,
        "total": len(uncertain_idx),
        "correct": l2_correct,
    }


def run_l3_audit(results, l1_indices, l2_indices, ollama_wait=300):
    """
    Stage 4: L3 Judge audits ALL L1 and L2 decisions.
    """
    audit_indices = list(set(l1_indices + l2_indices))

    if not audit_indices:
        return results, {"time": 0, "total": 0, "correct": 0, "overrides": 0}

    print("\n" + "="*60)
    print("STAGE 4: L3 JUDGE (gpt-oss:120b)")
    print("="*60)
    print(f"Auditing {len(audit_indices)} samples (all L1 + L2 decisions)")

    from l3_judge import L3Judge
    l3 = L3Judge()

    overrides = 0
    start = time.time()

    for i, idx in enumerate(audit_indices):
        r = results[idx]
        text = r["text"]

        # Build context from previous layers
        context = {
            "l0": f"{r['l0_label']} (conf={r['l0_confidence']:.2f})",
        }
        if "l1_label" in r:
            context["l1"] = f"{r['l1_label']} (conf={r['l1_confidence']:.2f})"
        if "l2_label" in r:
            context["l2"] = f"{r['l2_label']} (votes={r['l2_votes']})"

        result = l3.judge(text, context)

        results[idx]["l3_label"] = result["label"]
        results[idx]["l3_confidence"] = result["confidence"]
        results[idx]["l3_reasoning"] = result["reasoning"][:300]
        results[idx]["l3_audit_id"] = result["audit_id"]

        # Check for override
        prev_label = r.get("l2_label") or r.get("l1_label")
        if result["label"] != prev_label:
            overrides += 1
            results[idx]["l3_override"] = True

        if (i + 1) % 10 == 0:
            print(f"  L3 audited {i + 1}/{len(audit_indices)}")

    l3_time = time.time() - start

    # Score L3
    l3_samples = [results[idx] for idx in audit_indices]
    l3_correct = sum(1 for r in l3_samples if r.get("l3_label") == r["expected"])

    print(f"\nL3 Results ({l3_time:.1f}s):")
    print(f"  Total accuracy: {l3_correct}/{len(audit_indices)} = {l3_correct/len(audit_indices)*100:.1f}%")
    print(f"  Overrides: {overrides} ({overrides/len(audit_indices)*100:.1f}%)")

    # Unload L3
    del l3
    if ollama_wait > 0:
        wait_for_ollama_unload(ollama_wait)

    return results, {
        "time": l3_time,
        "total": len(audit_indices),
        "correct": l3_correct,
        "overrides": overrides,
    }


def compute_final_metrics(results, l0_confident_idx, l1_confident_idx, l2_idx, l3_idx):
    """Compute final cascade metrics."""

    # Determine final label for each sample
    for r in results:
        idx = r["idx"]

        if idx in l3_idx:
            # L3 is final authority
            r["final_label"] = r.get("l3_label", r.get("l2_label", r.get("l1_label", r["l0_label"])))
            r["stopped_at"] = "L3"
        elif idx in l2_idx:
            r["final_label"] = r.get("l2_label", r.get("l1_label", r["l0_label"]))
            r["stopped_at"] = "L2"
        elif idx in l1_confident_idx:
            r["final_label"] = r["l1_label"]
            r["stopped_at"] = "L1"
        else:
            r["final_label"] = r["l0_label"]
            r["stopped_at"] = "L0"

    # Overall accuracy
    correct = sum(1 for r in results if r["final_label"] == r["expected"])
    accuracy = correct / len(results) * 100

    # Confusion matrix
    tp = sum(1 for r in results if r["final_label"] == "harmful" and r["expected"] == "harmful")
    tn = sum(1 for r in results if r["final_label"] == "safe" and r["expected"] == "safe")
    fp = sum(1 for r in results if r["final_label"] == "harmful" and r["expected"] == "safe")
    fn = sum(1 for r in results if r["final_label"] == "safe" and r["expected"] == "harmful")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Layer distribution
    layer_counts = Counter(r["stopped_at"] for r in results)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "layer_distribution": dict(layer_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Full cascade evaluation on Heretic")
    parser.add_argument("--samples", type=int, default=None, help="Limit samples (default: all)")
    parser.add_argument("--l0-threshold", type=float, default=0.7, help="L0 confidence threshold")
    parser.add_argument("--l1-threshold", type=float, default=0.7, help="L1 confidence threshold")
    parser.add_argument("--ollama-wait", type=int, default=360, help="Seconds to wait for Ollama unload (default 6 min)")
    parser.add_argument("--skip-l3", action="store_true", help="Skip L3 audit")
    args = parser.parse_args()

    # Load Heretic data
    test_path = Path("../combined_test.json")
    train_path = Path("../combined_train.json")

    if not test_path.exists():
        print(f"Error: Heretic test data not found at {test_path}")
        return

    print("Loading Heretic dataset...")
    with open(test_path) as f:
        data = json.load(f)

    if train_path.exists():
        with open(train_path) as f:
            train_data = json.load(f)
        data = train_data + data
        print(f"  Combined train + test: {len(data)} samples")
    else:
        print(f"  Test only: {len(data)} samples")

    # Normalize labels
    for item in data:
        if item["label"] == "harmless":
            item["label"] = "safe"

    # Limit samples if requested
    if args.samples and args.samples < len(data):
        data = data[:args.samples]
        print(f"  Limited to {len(data)} samples")

    # Label distribution
    label_dist = Counter(item["label"] for item in data)
    print(f"\nDataset: {len(data)} samples")
    print(f"  Safe: {label_dist['safe']} ({label_dist['safe']/len(data)*100:.1f}%)")
    print(f"  Harmful: {label_dist['harmful']} ({label_dist['harmful']/len(data)*100:.1f}%)")

    print(f"\nConfig:")
    print(f"  L0 threshold: {args.l0_threshold}")
    print(f"  L1 threshold: {args.l1_threshold}")
    print(f"  Ollama wait: {args.ollama_wait}s")
    print(f"  Skip L3: {args.skip_l3}")

    total_start = time.time()

    # Stage 1: L0
    results, l0_uncertain_idx, l0_stats = run_l0_batch(data, args.l0_threshold)
    l0_confident_idx = [i for i in range(len(data)) if i not in l0_uncertain_idx]

    # Stage 2: L1
    results, l1_uncertain_idx, l1_stats = run_l1_batch(results, l0_uncertain_idx, args.l1_threshold)
    l1_confident_idx = [i for i in l0_uncertain_idx if i not in l1_uncertain_idx]

    # Stage 3: L2
    results, l2_stats = run_l2_batch(results, l1_uncertain_idx, args.ollama_wait)

    # Stage 4: L3 (audit all L1 + L2)
    if not args.skip_l3:
        l3_audit_idx = l0_uncertain_idx  # All that went to L1 or beyond
        results, l3_stats = run_l3_audit(results, l1_confident_idx, l1_uncertain_idx, args.ollama_wait)
    else:
        l3_stats = {"time": 0, "total": 0, "correct": 0, "overrides": 0}
        l3_audit_idx = []

    total_time = time.time() - total_start

    # Compute final metrics
    final_metrics = compute_final_metrics(
        results,
        l0_confident_idx,
        l1_confident_idx,
        l1_uncertain_idx,
        l3_audit_idx if not args.skip_l3 else []
    )

    # Print final results
    print("\n" + "="*60)
    print("FINAL CASCADE RESULTS")
    print("="*60)

    print("\nLayer Distribution:")
    for layer in ["L0", "L1", "L2", "L3"]:
        count = final_metrics["layer_distribution"].get(layer, 0)
        pct = count / len(results) * 100 if results else 0
        print(f"  {layer}: {count} ({pct:.1f}%)")

    print(f"\nOverall Accuracy: {final_metrics['correct']}/{final_metrics['total']} = {final_metrics['accuracy']:.1f}%")

    cm = final_metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print(f"  TP (caught harmful): {cm['tp']}")
    print(f"  TN (allowed safe): {cm['tn']}")
    print(f"  FP (blocked safe): {cm['fp']}")
    print(f"  FN (missed harmful): {cm['fn']}")

    print(f"\nMetrics:")
    print(f"  Precision: {final_metrics['precision']:.1f}%")
    print(f"  Recall: {final_metrics['recall']:.1f}%")
    print(f"  F1: {final_metrics['f1']:.1f}%")

    print(f"\nPer-Layer Performance:")
    print(f"  L0: {l0_stats['correct']}/{l0_stats['total']} = {l0_stats['correct']/l0_stats['total']*100:.1f}% ({l0_stats['time']:.1f}s)")
    if l1_stats['total'] > 0:
        print(f"  L1: {l1_stats['correct']}/{l1_stats['total']} = {l1_stats['correct']/l1_stats['total']*100:.1f}% ({l1_stats['time']:.1f}s)")
    if l2_stats['total'] > 0:
        print(f"  L2: {l2_stats['correct']}/{l2_stats['total']} = {l2_stats['correct']/l2_stats['total']*100:.1f}% ({l2_stats['time']:.1f}s)")
    if l3_stats['total'] > 0:
        print(f"  L3: {l3_stats['correct']}/{l3_stats['total']} = {l3_stats['correct']/l3_stats['total']*100:.1f}% ({l3_stats['time']:.1f}s)")
        print(f"      Overrides: {l3_stats['overrides']}")

    print(f"\nTotal Time: {total_time:.1f}s ({total_time/len(results)*1000:.1f}ms/sample avg)")

    # Save results
    output_path = Path(f"heretic_full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": "Heretic",
            "samples": len(results),
            "config": {
                "l0_threshold": args.l0_threshold,
                "l1_threshold": args.l1_threshold,
                "ollama_wait": args.ollama_wait,
                "skip_l3": args.skip_l3,
            },
            "layer_stats": {
                "l0": l0_stats,
                "l1": l1_stats,
                "l2": l2_stats,
                "l3": l3_stats,
            },
            "final_metrics": final_metrics,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Error analysis
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    # False negatives
    fn_samples = [r for r in results if r["final_label"] == "safe" and r["expected"] == "harmful"][:5]
    if fn_samples:
        print("\nFalse Negatives (Missed Harmful):")
        for r in fn_samples:
            text = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
            print(f"  [{r['stopped_at']}] {text}")

    # False positives
    fp_samples = [r for r in results if r["final_label"] == "harmful" and r["expected"] == "safe"][:5]
    if fp_samples:
        print("\nFalse Positives (Blocked Safe):")
        for r in fp_samples:
            text = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
            print(f"  [{r['stopped_at']}] {text}")


if __name__ == "__main__":
    main()
