#!/usr/bin/env python3
"""Generate experiment matrix scripts (06-11)"""

# Experiment matrix:
#                 Minimal      Medium       Verbose
# Baseline (20b)  Exp 05       Exp 08       Exp 09
# Safeguard       Exp 06       Exp 10       Exp 11

experiments = {
    "06": ("gpt-oss-safeguard:latest", "policies_minimal", "Safeguard model + Minimal policies"),
    "08": ("gpt-oss:20b", "policies_medium", "Baseline model + Medium policies"),
    "09": ("gpt-oss:20b", "policies_verbose", "Baseline model + Verbose policies"),
    "10": ("gpt-oss-safeguard:latest", "policies_medium", "Safeguard model + Medium policies"),
    "11": ("gpt-oss-safeguard:latest", "policies_verbose", "Safeguard model + Verbose policies"),
}

for exp_num, (model, policy_dir, description) in experiments.items():
    script = f'''#!/usr/bin/env python3
"""Experiment {exp_num}: {description}"""

import sys
import time
sys.path.insert(0, '.')

# Patch model and policy directory
import serial_gauntlet_simple

# Override load_policies to use {policy_dir}
def load_policies_{exp_num}():
    policies = {{}}
    import os
    for filename in os.listdir('{policy_dir}'):
        if filename.endswith('.txt'):
            policy_name = filename.replace('.txt', '')
            with open(os.path.join('{policy_dir}', filename)) as f:
                policies[policy_name] = f.read()
    return policies

serial_gauntlet_simple.load_policies = load_policies_{exp_num}

# Override model
_original_check = serial_gauntlet_simple.check_one_policy

def patched_check(content, policy_name, policy_text, model_param="gpt-oss:20b"):
    return _original_check(content, policy_name, policy_text, "{model}")

serial_gauntlet_simple.check_one_policy = patched_check

# Run evaluation
from eval_benchmark import main as run_eval

if __name__ == '__main__':
    start_time = time.time()

    # Run evaluation on WildGuard benchmark
    sys.argv = ['experiment_{exp_num}.py', 'wildguard_benchmark.json']
    run_eval()

    elapsed = time.time() - start_time
    throughput = (300 / elapsed) * 3600  # 300 samples

    print(f"\\n{{'='*80}}")
    print("EXPERIMENT {exp_num} RESULTS")
    print(f"{{'='*80}}")
    print(f"Model: {model}")
    print(f"Policies: {policy_dir}")
    print(f"Total time: {{elapsed/60:.1f}} minutes")
    print(f"Throughput: {{throughput:.0f}} samples/hour")
    print(f"{{'='*80}}\\n")
'''
    
    with open(f"experiment_{exp_num}_matrix.py", "w") as f:
        f.write(script)
    
    print(f"Created experiment_{exp_num}_matrix.py - {description}")

print("\n✓ All experiment scripts generated!")
print("\nExperiment Matrix:")
print("                 Minimal         Medium          Verbose")
print("Baseline (20b)   Exp 05 (DONE)  Exp 08          Exp 09")
print("Safeguard        Exp 06          Exp 10          Exp 11")
