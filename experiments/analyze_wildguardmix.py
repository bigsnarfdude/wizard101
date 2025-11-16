#!/usr/bin/env python3
"""Analyze WildGuardMix categories and mapping to our 6 policies"""

from datasets import load_dataset
from collections import Counter

print("Loading WildGuardMix test set...")
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")
test_data = dataset['test']

print(f"Total samples: {len(test_data)}\n")

# Analyze subcategories
subcategories = [sample['subcategory'] for sample in test_data]
subcategory_counts = Counter(subcategories)

print("Subcategories:")
for cat, count in sorted(subcategory_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:40} {count:4} samples")

# Analyze harmful vs unharmful
harm_labels = [sample['prompt_harm_label'] for sample in test_data]
harm_counts = Counter(harm_labels)
print(f"\nHarm labels:")
for label, count in harm_counts.items():
    print(f"  {label}: {count}")

# Show examples from each subcategory
print("\n" + "="*80)
print("SAMPLE PROMPTS BY SUBCATEGORY")
print("="*80)

for subcat in sorted(set(subcategories)):
    # Find first harmful example in this subcategory
    example = next((s for s in test_data if s['subcategory'] == subcat and s['prompt_harm_label'] == 'harmful'), None)
    if example:
        print(f"\n{subcat}:")
        print(f"  Prompt: {example['prompt'][:150]}...")
        print(f"  Harm: {example['prompt_harm_label']}")
