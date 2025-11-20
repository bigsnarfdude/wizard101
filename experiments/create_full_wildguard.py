#!/usr/bin/env python3
"""Create FULL WildGuard benchmark (no sampling)"""

from datasets import load_dataset
import json
import random

# Mapping WildGuardMix subcategories to our 6 policies
CATEGORY_MAPPING = {
    # hate_speech
    "toxic_language_hate_speech": ["hate_speech"],
    "social_stereotypes_and_unfair_discrimination": ["hate_speech"],

    # violence
    "violence_and_physical_harm": ["violence"],

    # self_harm
    "mental_health_over-reliance_crisis": ["self_harm"],

    # sexual_minors (WildGuardMix has "sexual_content" but not specifically minors)
    # We'll skip this - no good mapping

    # harassment
    "defamation_encouraging_unethical_or_unsafe_actions": ["harassment"],

    # illegal
    "fraud_assisting_illegal_activities": ["illegal"],
    "cyberattack": ["illegal"],
    "private_information_individual": ["illegal"],
    "causing_material_harm_by_disseminating_misinformation": ["illegal"],
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": ["illegal"],
    "copyright_violations": ["illegal"],
    "sensitive_information_organization_government": ["illegal"],
}

print("Loading WildGuardMix test set...")
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")
test_data = dataset['test']

print(f"Total samples in WildGuardMix test: {len(test_data)}")

# Convert to our format
converted = []
skipped_categories = set()

for sample in test_data:
    subcat = sample['subcategory']
    harm_label = sample['prompt_harm_label']

    # Skip benign and unmapped categories
    if subcat == "benign":
        # Add ALL benign samples (no sampling)
        if harm_label == "unharmful":
            converted.append({
                "content": sample['prompt'],
                "labels": [],
                "source": f"wildguardmix_{subcat}"
            })
        continue

    if subcat not in CATEGORY_MAPPING:
        skipped_categories.add(subcat)
        continue

    # Only include harmful samples
    if harm_label == "harmful":
        converted.append({
            "content": sample['prompt'],
            "labels": CATEGORY_MAPPING[subcat],
            "source": f"wildguardmix_{subcat}"
        })

print(f"\nConverted {len(converted)} samples (FULL dataset, no sampling)")
print(f"Skipped categories: {sorted(skipped_categories)}")

# Count distribution
from collections import Counter
all_labels = []
for s in converted:
    all_labels.extend(s['labels'])

counts = Counter(all_labels)
print("\nLabel distribution:")
for label, count in sorted(counts.items()):
    print(f"  {label:20} {count:4} samples")

safe_count = len([s for s in converted if not s['labels']])
print(f"  {'SAFE':20} {safe_count:4} samples")

# Save FULL benchmark
with open('wildguard_full_benchmark.json', 'w') as f:
    json.dump(converted, f, indent=2)

print(f"\nSaved full benchmark to wildguard_full_benchmark.json ({len(converted)} samples)")
print(f"Estimated runtime: ~{len(converted) * 12 / 3600:.1f} hours per experiment")
