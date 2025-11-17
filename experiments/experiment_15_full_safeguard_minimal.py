#!/usr/bin/env python3
"""Experiment 15: Safeguard + Minimal on FULL WildGuard (1554 samples)"""

import sys
import time
sys.path.insert(0, '.')

from eval_benchmark import main as run_eval

# Override to use safeguard model
import serial_gauntlet_simple
original_check = serial_gauntlet_simple.check_one_policy

def check_with_safeguard(content, policy_name, policy_text):
    result = original_check.__wrapped__(content, policy_name, policy_text, model="gpt-oss-safeguard:latest")
    return result

if not hasattr(original_check, '__wrapped__'):
    serial_gauntlet_simple.check_one_policy.__wrapped__ = original_check
    serial_gauntlet_simple.check_one_policy = check_with_safeguard

if __name__ == '__main__':
    start_time = time.time()

    print("=" * 80)
    print("EXPERIMENT 15: Safeguard + Minimal Policies (FULL WILDGUARD)")
    print("=" * 80)
    print("Model: gpt-oss-safeguard:latest")
    print("Policies: Minimal (100-150 tokens)")
    print("Dataset: wildguard_full_benchmark.json (1554 samples)")
    print("=" * 80)
    print()

    sys.argv = ['experiment_15.py', 'wildguard_full_benchmark.json']
    run_eval()

    elapsed = time.time() - start_time
    throughput = (1554 / elapsed) * 3600

    print()
    print("=" * 80)
    print("EXPERIMENT 15 RESULTS")
    print("=" * 80)
    print(f"Model: gpt-oss-safeguard:latest")
    print(f"Policies: policies_minimal (100-150 tokens)")
    print(f"Dataset: FULL WildGuard (1554 samples)")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Throughput: {throughput:.0f} samples/hour")
    print()
    print(f"Validation: Comparing to Exp 09 result (21.0% on 300 samples)")
    print("=" * 80)
