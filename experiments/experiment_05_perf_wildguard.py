#!/usr/bin/env python3
"""Experiment 05: Performance test on WildGuard benchmark (300 REAL samples)"""

import sys
import time
sys.path.insert(0, '.')

from eval_benchmark import main as run_eval

if __name__ == '__main__':
    start_time = time.time()

    # Run evaluation (pass wildguard_benchmark.json as argument)
    sys.argv = ['experiment_05_perf_wildguard.py', 'wildguard_benchmark.json']
    run_eval()

    elapsed = time.time() - start_time
    throughput = (300 / elapsed) * 3600  # 300 samples

    print(f"\n{'='*80}")
    print("PERFORMANCE METRICS - EXPERIMENT 05")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Throughput: {throughput:.0f} samples/hour")
    print(f"{'='*80}\n")
