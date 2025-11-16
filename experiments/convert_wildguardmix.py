#!/usr/bin/env python3
"""Convert WildGuardMix to our 6-policy format"""

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

print("Loading WildGuardMix...")
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")
test_data = dataset['test']

# Convert to our format
converted = []
skipped_categories = set()

for sample in test_data:
    subcat = sample['subcategory']
    harm_label = sample['prompt_harm_label']
    
    # Skip benign and unmapped categories
    if subcat == "benign":
        # Add some benign samples
        if harm_label == "unharmful" and random.random() < 0.15:  # 15% sample rate
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

print(f"\nConverted {len(converted)} samples")
print(f"Skipped categories: {sorted(skipped_categories)}")

# Sample 300 samples (balanced mix)
random.seed(42)
if len(converted) > 300:
    sampled = random.sample(converted, 300)
else:
    sampled = converted

print(f"Final benchmark: {len(sampled)} samples")

# Count distribution
from collections import Counter
all_labels = []
for s in sampled:
    all_labels.extend(s['labels'])

counts = Counter(all_labels)
print("\nLabel distribution:")
for label, count in sorted(counts.items()):
    print(f"  {label:20} {count:3} samples")

safe_count = len([s for s in sampled if not s['labels']])
print(f"  {'SAFE':20} {safe_count:3} samples")

# Save
with open('wildguard_benchmark.json', 'w') as f:
    json.dump(sampled, f, indent=2)

print(f"\n✓ Saved to wildguard_benchmark.json")
