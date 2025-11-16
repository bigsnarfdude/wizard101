#!/usr/bin/env python3
"""Performance evaluation: throughput and accuracy on larger benchmark"""

import sys
import time
sys.path.insert(0, '.')

from eval_benchmark import load_benchmark, load_policies, evaluate_benchmark

def main():
    benchmark_file = sys.argv[1] if len(sys.argv) > 1 else 'larger_benchmark.json'
    
    print(f"Loading benchmark: {benchmark_file}")
    samples = load_benchmark(benchmark_file)
    
    print(f"Loading policies...")
    policies = load_policies()
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE EVALUATION")
    print(f"{'='*80}")
    print(f"Samples: {len(samples)}")
    print(f"Policies: {len(policies)}")
    print(f"Model: gpt-oss:20b")
    print(f"{'='*80}\n")
    
    # Track timing
    start_time = time.time()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_timestamp}\n")
    
    # Run evaluation
    results = evaluate_benchmark(samples, policies)
    
    # Calculate timing
    end_time = time.time()
    end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    total_seconds = end_time - start_time
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    
    # Calculate throughput
    samples_per_second = len(samples) / total_seconds
    samples_per_minute = len(samples) / total_minutes
    samples_per_hour = len(samples) / total_hours
    
    seconds_per_sample = total_seconds / len(samples)
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*80}")
    print(f"Start time:          {start_timestamp}")
    print(f"End time:            {end_timestamp}")
    print(f"Total time:          {total_minutes:.1f} minutes ({total_hours:.2f} hours)")
    print(f"Seconds per sample:  {seconds_per_sample:.1f}s")
    print(f"Throughput:          {samples_per_hour:.0f} samples/hour")
    print(f"                     {samples_per_minute:.1f} samples/minute")
    print(f"                     {samples_per_second:.2f} samples/second")
    print(f"{'='*80}\n")
    
    # Save results with timing
    import json
    output = {
        "benchmark_file": benchmark_file,
        "timestamp": start_timestamp,
        "performance": {
            "total_samples": len(samples),
            "total_seconds": total_seconds,
            "total_minutes": total_minutes,
            "seconds_per_sample": seconds_per_sample,
            "samples_per_hour": samples_per_hour,
            "samples_per_minute": samples_per_minute
        },
        "accuracy": results
    }
    
    output_file = benchmark_file.replace('.json', '_performance.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved performance results to {output_file}\n")

if __name__ == '__main__':
    main()
