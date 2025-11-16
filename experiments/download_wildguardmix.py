#!/usr/bin/env python3
"""Download WildGuardMix dataset from HuggingFace"""

from datasets import load_dataset
import json

print("Downloading WildGuardMix test split...")
print("This may take a few minutes...")

# Load test dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")

print(f"\nDataset loaded!")
print(f"Test set size: {len(dataset['test'])} samples")
print(f"Columns: {dataset['test'].column_names}")

# Show first few examples
print("\nFirst 3 examples:")
for i in range(min(3, len(dataset['test']))):
    example = dataset['test'][i]
    print(f"\n--- Example {i+1} ---")
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 150:
            print(f"{key}: {value[:150]}...")
        else:
            print(f"{key}: {value}")

# Save to JSON for inspection
print("\nSaving sample to wildguardmix_sample.json...")
sample = [dataset['test'][i] for i in range(min(10, len(dataset['test'])))]
with open('wildguardmix_sample.json', 'w') as f:
    json.dump(sample, f, indent=2)

print("✓ Saved 10 samples to wildguardmix_sample.json")
print(f"\nFull dataset: {len(dataset['test'])} test samples available")
