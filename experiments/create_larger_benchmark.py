#!/usr/bin/env python3
"""Create larger benchmark for performance testing (300 samples)"""

import json

# Load balanced benchmark
with open('balanced_benchmark.json') as f:
    balanced_benchmark = json.load(f)

# Repeat the balanced benchmark 3x with slight variations
larger_benchmark = []

# Add original 90 samples
larger_benchmark.extend(balanced_benchmark)

# Create 210 more samples by repeating with variations
for i in range(2):  # 2 more copies = 180 more samples
    for sample in balanced_benchmark:
        content = sample["content"]
        
        # Add variations to violation samples only
        if sample["labels"]:
            if i == 0 and not content.startswith("extremely"):
                content = "extremely " + content
            elif i == 1 and not content.startswith("very"):
                content = "very " + content
        
        larger_benchmark.append({
            "content": content,
            "labels": sample["labels"],
            "source": f"synthetic_var_{i+1}"
        })

# Take first 300 samples
larger_benchmark = larger_benchmark[:300]

print(f"Created larger benchmark with {len(larger_benchmark)} samples")

# Verify distribution
from collections import Counter
all_labels = []
for sample in larger_benchmark:
    all_labels.extend(sample['labels'])

label_counts = Counter(all_labels)
print("\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label:20} {count:3} samples")

# Save
with open('larger_benchmark.json', 'w') as f:
    json.dump(larger_benchmark, f, indent=2)

print("\n✓ Saved to larger_benchmark.json")
print(f"\nEstimated time at 12s per sample: {len(larger_benchmark) * 12 / 60:.1f} minutes")
print(f"Estimated throughput: {3600 / 12:.0f} samples/hour")
