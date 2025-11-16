#!/usr/bin/env python3
"""Compare baseline vs safeguard models on WildGuard benchmark"""

import sys
sys.path.insert(0, '.')
from eval_benchmark import load_benchmark, load_policies, evaluate_benchmark
import time

# Test both models
MODELS = [
    "gpt-oss:20b",                 # Baseline
    "gpt-oss-safeguard:latest"     # Safeguard (20b)
]

def main():
    benchmark_file = "wildguard_benchmark.json"
    
    print(f"Loading benchmark: {benchmark_file}")
    samples = load_benchmark(benchmark_file)
    
    print(f"Loading policies...")
    policies = load_policies()
    
    results_all = {}
    
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model}")
        print(f"{'='*80}")
        print(f"Samples: {len(samples)}")
        print(f"Policies: {len(policies)}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Temporarily patch the model in serial_gauntlet_simple
        import serial_gauntlet_simple
        original_check = serial_gauntlet_simple.check_one_policy
        
        def patched_check(content, policy_name, policy_text, model_param="gpt-oss:20b"):
            return original_check(content, policy_name, policy_text, model)
        
        serial_gauntlet_simple.check_one_policy = patched_check
        
        # Run evaluation
        results = evaluate_benchmark(samples, policies)
        
        # Restore original
        serial_gauntlet_simple.check_one_policy = original_check
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        results_all[model] = {
            "results": results,
            "time_seconds": elapsed,
            "samples_per_hour": (len(samples) / elapsed) * 3600
        }
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model} - COMPLETED")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Throughput: {results_all[model]['samples_per_hour']:.0f} samples/hour")
        print(f"{'='*80}\n")
    
    # Save comparison
    import json
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results_all, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    for model, data in results_all.items():
        print(f"\n{model}:")
        print(f"  Multi-policy accuracy: {data['results']['multi_policy_accuracy']:.1f}%")
        print(f"  Overall accuracy: {data['results']['overall_accuracy']:.1f}%")
        print(f"  Throughput: {data['samples_per_hour']:.0f} samples/hour")
    
    print(f"\n✓ Saved to model_comparison_results.json")

if __name__ == '__main__':
    main()
