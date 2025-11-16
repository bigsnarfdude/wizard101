#!/usr/bin/env python3
"""Test ONLY safeguard model - compare to existing baseline results"""

import sys
import time
sys.path.insert(0, '.')

# Patch model BEFORE importing
MODEL = "gpt-oss-safeguard:latest"

# Monkey-patch the model in serial_gauntlet_simple
import serial_gauntlet_simple
original_code = serial_gauntlet_simple.check_one_policy.__code__

def check_one_policy_patched(content, policy_name, policy_text, model="gpt-oss:20b"):
    """Override to use safeguard model"""
    return serial_gauntlet_simple.PolicyResult(
        policy_name=policy_name,
        violation=False,
        confidence=0.0,
        latency_ms=0,
        thinking="patched"
    )

# Actually just modify the default
import serial_gauntlet_simple
_original_check = serial_gauntlet_simple.check_one_policy

def patched_check(content, policy_name, policy_text, model="gpt-oss:20b"):
    return _original_check(content, policy_name, policy_text, MODEL)

serial_gauntlet_simple.check_one_policy = patched_check

from eval_benchmark import load_benchmark, load_policies, evaluate_benchmark

def main():
    benchmark_file = "wildguard_benchmark.json"
    
    print(f"Loading benchmark: {benchmark_file}")
    samples = load_benchmark(benchmark_file)
    
    print(f"Loading policies...")
    policies = load_policies()
    
    print(f"\n{'='*80}")
    print(f"TESTING MODEL: {MODEL}")
    print(f"{'='*80}")
    print(f"Samples: {len(samples)}")
    print(f"Policies: {len(policies)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    results = evaluate_benchmark(samples, policies)
    
    end_time = time.time()
    elapsed = end_time - start_time
    throughput = (len(samples) / elapsed) * 3600
    
    print(f"\n{'='*80}")
    print(f"SAFEGUARD MODEL RESULTS")
    print(f"{'='*80}")
    print(f"Model: {MODEL}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Throughput: {throughput:.0f} samples/hour")
    print(f"Multi-policy accuracy: {results['multi_policy_accuracy']:.1f}%")
    print(f"Overall accuracy: {results['overall_accuracy']:.1f}%")
    print(f"{'='*80}\n")
    
    # Save
    import json
    output = {
        "model": MODEL,
        "benchmark": benchmark_file,
        "time_seconds": elapsed,
        "throughput": throughput,
        "results": results
    }
    
    with open('safeguard_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✓ Saved to safeguard_results.json")
    
    print("\nCompare to baseline (gpt-oss:20b) numbers from earlier runs:")
    print("  - Balanced benchmark: 61.1% multi-policy, 98.9% overall")

if __name__ == '__main__':
    main()
