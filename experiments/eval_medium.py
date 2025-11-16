import sys
sys.path.insert(0, '.')
from eval_benchmark import *

# Override load_policies to use medium
def load_policies_medium():
    policies = {}
    policy_dir = 'policies_medium'
    import os
    for filename in os.listdir(policy_dir):
        if filename.endswith('.txt'):
            policy_name = filename.replace('.txt', '')
            with open(os.path.join(policy_dir, filename)) as f:
                policies[policy_name] = f.read()
    return policies

# Patch
import serial_gauntlet_simple
serial_gauntlet_simple.load_policies = load_policies_medium

if __name__ == '__main__':
    main()
