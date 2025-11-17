#!/usr/bin/env python3
"""Experiment 12: Baseline + Minimal on FULL WildGuard (1554 samples)
Validation experiment to confirm 23.0% result holds at scale"""

import sys
import time
sys.path.insert(0, '.')

from eval_benchmark import main as run_eval

if __name__ == '__main__':
    start_time = time.time()

    # Run evaluation on FULL WildGuard benchmark
    sys.argv = ['experiment_12.py', 'wildguard_full_benchmark.json']
    run_eval()

    elapsed = time.time() - start_time
    throughput = (1554 / elapsed) * 3600  # 1554 samples

    print(f"\n{'='*80}")
    print("EXPERIMENT 12 RESULTS")
    print(f"{'='*80}")
    print(f"Model: gpt-oss:20b (Baseline)")
    print(f"Policies: policies_minimal (100-150 tokens)")
    print(f"Dataset: FULL WildGuard (1554 samples)")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Throughput: {throughput:.0f} samples/hour")
    print(f"\nValidation: Comparing to Exp 05 result (23.0% on 300 samples)")
    print(f"{'='*80}\n")
