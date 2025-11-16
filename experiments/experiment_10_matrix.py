#!/usr/bin/env python3
"""Experiment 10: Safeguard model + Medium policies"""

import sys
import time
sys.path.insert(0, '.')

# Patch model and policy directory
import serial_gauntlet_simple

# Override load_policies to use policies_medium
def load_policies_10():
    policies = {}
    import os
    for filename in os.listdir('policies_medium'):
        if filename.endswith('.txt'):
            policy_name = filename.replace('.txt', '')
            with open(os.path.join('policies_medium', filename)) as f:
                policies[policy_name] = f.read()
    return policies

serial_gauntlet_simple.load_policies = load_policies_10

# Override model
_original_check = serial_gauntlet_simple.check_one_policy

def patched_check(content, policy_name, policy_text, model_param="gpt-oss:20b"):
    return _original_check(content, policy_name, policy_text, "gpt-oss-safeguard:latest")

serial_gauntlet_simple.check_one_policy = patched_check

# Run evaluation
from eval_benchmark import main as run_eval

if __name__ == '__main__':
    start_time = time.time()

    # Run evaluation on WildGuard benchmark
    sys.argv = ['experiment_10.py', 'wildguard_benchmark.json']
    run_eval()

    elapsed = time.time() - start_time
    throughput = (300 / elapsed) * 3600  # 300 samples

    print(f"\n{'='*80}")
    print("EXPERIMENT 10 RESULTS")
    print(f"{'='*80}")
    print(f"Model: gpt-oss-safeguard:latest")
    print(f"Policies: policies_medium")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Throughput: {throughput:.0f} samples/hour")
    print(f"{'='*80}\n")
