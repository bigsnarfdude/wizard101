#!/usr/bin/env python3
"""
Tune L0 Threshold for Target Efficiency

Find the confidence threshold that achieves 70% L0 pass-through
while measuring accuracy trade-off.

Target Distribution:
- L0: 70% (fast pass-through)
- L1: 25% (escalated)
- L2: 4%
- L3: 1%

Usage:
    python tune_l0_threshold.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Model paths
L0_MODEL = "./models/l0_bouncer_mega"

# Target efficiency
TARGET_PASS_THROUGH = 0.70


class ThresholdTuner:
    """Tune L0 threshold for target pass-through rate."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load L0
        print("Loading L0 Bouncer...")
        self.l0_tokenizer = AutoTokenizer.from_pretrained(L0_MODEL)
        self.l0_model = AutoModelForSequenceClassification.from_pretrained(L0_MODEL)
        self.l0_model.to(self.device)
        self.l0_model.eval()
        print(f"  L0 loaded on {self.device}")

    def run_l0(self, text):
        """Run L0 bouncer and return all probabilities."""
        inputs = self.l0_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.l0_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        safe_prob = probs[0][0].item()
        harmful_prob = probs[0][1].item()

        return {
            "safe_prob": safe_prob,
            "harmful_prob": harmful_prob,
            "confidence": max(safe_prob, harmful_prob),
            "predicted": "safe" if safe_prob > harmful_prob else "harmful",
        }

    def evaluate_threshold(self, test_data, threshold):
        """Evaluate performance at given threshold."""
        results = {
            "passed": 0,  # L0 handled (not escalated)
            "escalated": 0,
            "correct": 0,
            "total": len(test_data),
            "tp": 0,  # True positives (correctly identified harmful)
            "tn": 0,  # True negatives (correctly identified safe)
            "fp": 0,  # False positives (safe classified as harmful)
            "fn": 0,  # False negatives (harmful classified as safe)
        }

        for item in test_data:
            text = item.get("prompt", item.get("text", ""))
            expected = item.get("label", "").lower()
            if expected == "harmless":
                expected = "safe"

            l0 = self.run_l0(text)

            # Decision: pass or escalate?
            # Pass if: confident AND predicted safe
            should_pass = (
                l0["confidence"] >= threshold and l0["predicted"] == "safe"
            )

            if should_pass:
                results["passed"] += 1
                predicted = "safe"
            else:
                results["escalated"] += 1
                # For this analysis, assume escalation correctly identifies harmful
                # (best case for L1)
                predicted = l0["predicted"]

            # Track correctness
            if predicted == expected:
                results["correct"] += 1

            # Confusion matrix
            if expected == "harmful" and predicted == "harmful":
                results["tp"] += 1
            elif expected == "safe" and predicted == "safe":
                results["tn"] += 1
            elif expected == "safe" and predicted == "harmful":
                results["fp"] += 1
            elif expected == "harmful" and predicted == "safe":
                results["fn"] += 1

        # Calculate metrics
        pass_rate = results["passed"] / results["total"]
        accuracy = results["correct"] / results["total"]

        precision = (
            results["tp"] / (results["tp"] + results["fp"])
            if (results["tp"] + results["fp"]) > 0
            else 0
        )
        recall = (
            results["tp"] / (results["tp"] + results["fn"])
            if (results["tp"] + results["fn"]) > 0
            else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "threshold": threshold,
            "pass_rate": pass_rate,
            "escalation_rate": 1 - pass_rate,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "passed": results["passed"],
            "escalated": results["escalated"],
            "fn": results["fn"],  # Missed harmful (critical metric)
        }

    def find_optimal_threshold(self, test_data):
        """Find threshold that achieves target pass-through."""
        print("=" * 60)
        print("L0 THRESHOLD TUNING")
        print("=" * 60)
        print(f"Target pass-through: {TARGET_PASS_THROUGH*100:.0f}%")
        print(f"Test samples: {len(test_data)}")

        # Test range of thresholds
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        results = []

        print(f"\n{'Thresh':<8} {'Pass%':>8} {'Acc%':>8} {'F1%':>8} {'FN':>6} {'Note':<15}")
        print("-" * 60)

        best_threshold = None
        best_diff = float("inf")

        for thresh in thresholds:
            result = self.evaluate_threshold(test_data, thresh)
            results.append(result)

            # Check if close to target
            diff = abs(result["pass_rate"] - TARGET_PASS_THROUGH)
            if diff < best_diff:
                best_diff = diff
                best_threshold = result

            # Note
            note = ""
            if result["pass_rate"] >= TARGET_PASS_THROUGH - 0.05:
                note = "NEAR TARGET"
            if result["fn"] > 0:
                note = f"MISSED {result['fn']} HARMFUL"

            print(
                f"{thresh:<8.2f} "
                f"{result['pass_rate']*100:>7.1f}% "
                f"{result['accuracy']*100:>7.1f}% "
                f"{result['f1']*100:>7.1f}% "
                f"{result['fn']:>6} "
                f"{note:<15}"
            )

        # Summary
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        # Find thresholds that hit target
        near_target = [r for r in results if r["pass_rate"] >= TARGET_PASS_THROUGH - 0.05]

        if near_target:
            # Best accuracy among those near target
            best = max(near_target, key=lambda x: x["accuracy"])
            print(f"\nOPTIMAL THRESHOLD: {best['threshold']:.2f}")
            print(f"  Pass-through: {best['pass_rate']*100:.1f}%")
            print(f"  Accuracy: {best['accuracy']*100:.1f}%")
            print(f"  F1: {best['f1']*100:.1f}%")
            print(f"  Missed harmful: {best['fn']}")
        else:
            # Can't hit target, show closest
            print(f"\nCANNOT HIT {TARGET_PASS_THROUGH*100:.0f}% TARGET")
            print(f"Closest threshold: {best_threshold['threshold']:.2f}")
            print(f"  Pass-through: {best_threshold['pass_rate']*100:.1f}%")
            print(f"  Accuracy: {best_threshold['accuracy']*100:.1f}%")

        # Trade-off analysis
        print("\nTRADE-OFF ANALYSIS:")

        # Compare low vs high threshold
        low_thresh = results[0]  # 0.50
        high_thresh = results[-1]  # 0.90

        print(f"\nLow threshold ({low_thresh['threshold']:.2f}):")
        print(f"  + High pass-through: {low_thresh['pass_rate']*100:.1f}%")
        print(f"  - Lower safety: may miss harmful content")
        print(f"  - Missed harmful: {low_thresh['fn']}")

        print(f"\nHigh threshold ({high_thresh['threshold']:.2f}):")
        print(f"  - Low pass-through: {high_thresh['pass_rate']*100:.1f}%")
        print(f"  + Higher safety: escalates uncertain cases")
        print(f"  + Missed harmful: {high_thresh['fn']}")

        # Recommendation
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)

        # Find best balance
        # Weight: 70% efficiency, 30% safety
        scores = []
        for r in results:
            efficiency_score = 1 - abs(r["pass_rate"] - TARGET_PASS_THROUGH)
            safety_score = 1 - (r["fn"] / len(test_data)) if r["fn"] < len(test_data) else 0
            combined = 0.7 * efficiency_score + 0.3 * safety_score
            scores.append((r, combined))

        best_balanced = max(scores, key=lambda x: x[1])[0]

        print(f"\nBALANCED OPTIMAL: {best_balanced['threshold']:.2f}")
        print(f"  Pass-through: {best_balanced['pass_rate']*100:.1f}%")
        print(f"  Accuracy: {best_balanced['accuracy']*100:.1f}%")
        print(f"  F1: {best_balanced['f1']*100:.1f}%")
        print(f"  Missed harmful: {best_balanced['fn']}")

        if best_balanced["pass_rate"] >= TARGET_PASS_THROUGH - 0.05:
            print(f"\nHITS TARGET ({TARGET_PASS_THROUGH*100:.0f}% pass-through)")
        else:
            gap = TARGET_PASS_THROUGH - best_balanced["pass_rate"]
            print(f"\nGAP: {gap*100:.1f}% below target")
            print("Options to close gap:")
            print("  1. Accept lower threshold (more risk)")
            print("  2. Retrain L0 with more data")
            print("  3. Adjust target expectations")

        return results


def main():
    # Load test data
    data_path = Path("../combined_test.json")
    if not data_path.exists():
        data_path = Path("data/combined_test.json")

    if not data_path.exists():
        print("Test data not found")
        return

    with open(data_path) as f:
        test_data = json.load(f)

    # Run tuning
    tuner = ThresholdTuner()
    results = tuner.find_optimal_threshold(test_data)

    # Save results
    output_path = Path("threshold_tuning_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
