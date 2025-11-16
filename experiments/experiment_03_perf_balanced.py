#!/usr/bin/env python3
"""Experiment 03: Performance baseline on balanced benchmark (90 samples)"""

import sys
import time
sys.path.insert(0, '.')

# Just use eval_benchmark's main() with timing wrapper
from eval_benchmark import main as run_eval

if __name__ == '__main__':
    start_time = time.time()

    # Run evaluation (pass balanced_benchmark.json as argument)
    sys.argv = ['experiment_03_perf_balanced.py', 'balanced_benchmark.json']
    run_eval()

    elapsed = time.time() - start_time
    throughput = (90 / elapsed) * 3600  # 90 samples

    print(f"\n{'='*80}")
    print("PERFORMANCE METRICS - EXPERIMENT 03")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Throughput: {throughput:.0f} samples/hour")
    print(f"{'='*80}\n")
