#!/usr/bin/env python3
"""
Evaluate L1 Analyst (GuardReasoner-8B 4-bit) baseline performance.
"""

import json
import time
from collections import defaultdict
from l1_analyst import L1Analyst

def load_test_data(path="guardreasoner_test_5k.json", limit=500):
    """Load test samples."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Sample for faster eval
    if limit and len(data) > limit:
        import random
        random.seed(42)
        data = random.sample(data, limit)

    return data

def evaluate():
    print("="*60)
    print("L1 ANALYST BASELINE EVALUATION")
    print("="*60)

    # Load model
    analyst = L1Analyst()

    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data(limit=500)
    print(f"Evaluating on {len(test_data)} samples")

    # Metrics
    results = []
    latencies = []

    print("\nRunning evaluation...")
    for i, sample in enumerate(test_data):
        text = sample.get('prompt') or sample.get('text') or sample.get('instruction', '')
        true_label = sample.get('label', '').lower()

        # Normalize labels
        if true_label in ['unsafe', 'harmful', '1', 1]:
            true_label = 'harmful'
        else:
            true_label = 'safe'

        # Predict
        start = time.time()
        result = analyst.analyze(text)
        latency = (time.time() - start) * 1000

        pred_label = result['label']
        latencies.append(latency)

        results.append({
            'true': true_label,
            'pred': pred_label,
            'confidence': result['confidence'],
            'latency_ms': latency,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_data)} samples processed...")

    # Calculate metrics
    tp = sum(1 for r in results if r['true'] == 'harmful' and r['pred'] == 'harmful')
    tn = sum(1 for r in results if r['true'] == 'safe' and r['pred'] == 'safe')
    fp = sum(1 for r in results if r['true'] == 'safe' and r['pred'] == 'harmful')
    fn = sum(1 for r in results if r['true'] == 'harmful' and r['pred'] == 'safe')

    accuracy = (tp + tn) / len(results) if results else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p50_latency = sorted(latencies)[len(latencies)//2] if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  F1 Score:  {f1*100:.1f}%")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}  FP: {fp}")
    print(f"  FN: {fn}  TN: {tn}")

    print(f"\nLatency:")
    print(f"  Mean: {avg_latency:.1f}ms")
    print(f"  P50:  {p50_latency:.1f}ms")
    print(f"  P99:  {p99_latency:.1f}ms")

    # Confidence distribution
    high_conf = sum(1 for r in results if r['confidence'] >= 0.9)
    med_conf = sum(1 for r in results if 0.7 <= r['confidence'] < 0.9)
    low_conf = sum(1 for r in results if r['confidence'] < 0.7)

    print(f"\nConfidence Distribution:")
    print(f"  High (>=0.9): {high_conf} ({high_conf/len(results)*100:.1f}%)")
    print(f"  Med (0.7-0.9): {med_conf} ({med_conf/len(results)*100:.1f}%)")
    print(f"  Low (<0.7):   {low_conf} ({low_conf/len(results)*100:.1f}%)")

    print("\n" + "="*60)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency_mean': avg_latency,
        'latency_p50': p50_latency,
        'latency_p99': p99_latency,
        'samples': len(results),
    }


if __name__ == "__main__":
    evaluate()
