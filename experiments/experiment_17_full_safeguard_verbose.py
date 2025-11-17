#!/usr/bin/env python3
"""Experiment 17: Safeguard + Verbose on FULL WildGuard (1554 samples)

This is the CRITICAL validation experiment for the verbose policy finding.
Original Exp 11 achieved 36.0% on 300 samples. Need to confirm this holds at scale.
"""

import sys
import time
sys.path.insert(0, '.')

from eval_benchmark import main as run_eval

# Override to use safeguard model + verbose policies
import serial_gauntlet_simple

# Override model
original_check = serial_gauntlet_simple.check_one_policy

def check_with_safeguard(content, policy_name, policy_text):
    result = original_check.__wrapped__(content, policy_name, policy_text, model="gpt-oss-safeguard:latest")
    return result

if not hasattr(original_check, '__wrapped__'):
    serial_gauntlet_simple.check_one_policy.__wrapped__ = original_check
    serial_gauntlet_simple.check_one_policy = check_with_safeguard

# Override policies
original_load = serial_gauntlet_simple.load_policies

def load_verbose_policies():
    policies = {}
    import os
    policy_dir = 'policies_verbose'
    for filename in os.listdir(policy_dir):
        if filename.endswith('.txt'):
            policy_name = filename.replace('.txt', '')
            with open(os.path.join(policy_dir, filename)) as f:
                policies[policy_name] = f.read()
    return policies

serial_gauntlet_simple.load_policies = load_verbose_policies

if __name__ == '__main__':
    start_time = time.time()

    print("=" * 80)
    print("EXPERIMENT 17: Safeguard + Verbose Policies (FULL WILDGUARD)")
    print("=" * 80)
    print("Model: gpt-oss-safeguard:latest")
    print("Policies: Verbose (800-900 tokens)")
    print("Dataset: wildguard_full_benchmark.json (1554 samples)")
    print()
    print("🔬 CRITICAL VALIDATION EXPERIMENT")
    print("Testing if 36.0% result from Exp 11 holds on full dataset")
    print("=" * 80)
    print()

    sys.argv = ['experiment_17.py', 'wildguard_full_benchmark.json']
    run_eval()

    elapsed = time.time() - start_time
    throughput = (1554 / elapsed) * 3600

    print()
    print("=" * 80)
    print("EXPERIMENT 17 RESULTS")
    print("=" * 80)
    print(f"Model: gpt-oss-safeguard:latest")
    print(f"Policies: policies_verbose (800-900 tokens)")
    print(f"Dataset: FULL WildGuard (1554 samples)")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Throughput: {throughput:.0f} samples/hour")
    print()
    print(f"⚠️  VALIDATION: Comparing to Exp 11 result (36.0% on 300 samples)")
    print("=" * 80)
