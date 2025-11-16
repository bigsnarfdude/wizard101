#!/usr/bin/env python3
"""Experiment 04: Performance test on larger benchmark (270 samples)"""

import sys
import time
sys.path.insert(0, '.')

from eval_benchmark import main as run_eval

if __name__ == '__main__':
    start_time = time.time()

    # Run evaluation (pass larger_benchmark.json as argument)
    sys.argv = ['experiment_04_perf_larger.py', 'larger_benchmark.json']
    run_eval()

    elapsed = time.time() - start_time
    throughput = (270 / elapsed) * 3600  # 270 samples

    print(f"\n{'='*80}")
    print("PERFORMANCE METRICS - EXPERIMENT 04")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Throughput: {throughput:.0f} samples/hour")
    print(f"{'='*80}\n")
