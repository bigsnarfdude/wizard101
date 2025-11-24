#!/usr/bin/env python3
"""
Build Large Test Set from GuardReasoner Data

Creates test set from samples NOT used in train_12k.json

Usage:
    python build_test_set.py
"""

import json
import random
from pathlib import Path
from collections import Counter


def extract_label(output_text):
    """Extract label from GuardReasoner output."""
    text = output_text.lower()
    if "request: harmful" in text:
        return "harmful"
    elif "request: unharmful" in text:
        return "safe"
    elif "request: benign" in text:
        return "safe"
    return None


def extract_prompt(input_text):
    """Extract prompt from GuardReasoner input."""
    text = input_text.strip()
    if text.startswith("Human user:"):
        text = text[11:].strip()
    if "\nAI assistant:" in text:
        text = text.split("\nAI assistant:")[0].strip()
    return text


def main():
    # Load all GuardReasoner data
    all_data_path = Path("../guardreasoner/guardreasoner_data/all_combined.json")
    with open(all_data_path) as f:
        all_data = json.load(f)
    print(f"Total GuardReasoner samples: {len(all_data):,}")

    # Load training data to exclude
    train_path = Path("train_12k.json")
    with open(train_path) as f:
        train_data = json.load(f)
    print(f"Training samples to exclude: {len(train_data):,}")

    # Create set of training prompts for fast lookup
    train_prompts = set(item["text"] for item in train_data)

    # Convert and filter
    test_samples = []
    for item in all_data:
        prompt = extract_prompt(item["input"])
        label = extract_label(item["output"])

        # Skip if in training set or invalid
        if not prompt or not label or len(prompt) < 10:
            continue
        if prompt in train_prompts:
            continue

        test_samples.append({
            "prompt": prompt,
            "label": label,
            "source": item.get("source_split", "guardreasoner")
        })

    print(f"\nAvailable test samples: {len(test_samples):,}")

    # Check distribution
    labels = Counter(s["label"] for s in test_samples)
    print(f"  Safe: {labels['safe']:,}")
    print(f"  Harmful: {labels['harmful']:,}")

    # Create balanced test set
    TARGET_SIZE = 10000  # Large - we have 97k available

    safe = [s for s in test_samples if s["label"] == "safe"]
    harmful = [s for s in test_samples if s["label"] == "harmful"]

    random.seed(42)
    n_each = TARGET_SIZE // 2
    sampled_safe = random.sample(safe, min(n_each, len(safe)))
    sampled_harmful = random.sample(harmful, min(n_each, len(harmful)))

    final_test = sampled_safe + sampled_harmful
    random.shuffle(final_test)

    # Final stats
    final_labels = Counter(s["label"] for s in final_test)
    print(f"\nFinal test set: {len(final_test):,} samples")
    print(f"  Safe: {final_labels['safe']:,} ({final_labels['safe']/len(final_test)*100:.1f}%)")
    print(f"  Harmful: {final_labels['harmful']:,} ({final_labels['harmful']/len(final_test)*100:.1f}%)")

    # Save
    output_path = Path("guardreasoner_test_10k.json")
    with open(output_path, "w") as f:
        json.dump(final_test, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
