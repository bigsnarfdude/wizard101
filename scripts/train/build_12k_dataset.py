#!/usr/bin/env python3
"""
Build 12K Training Dataset from GuardReasoner Data

Extract and convert GuardReasoner format to L0 bouncer format.

Usage:
    python build_12k_dataset.py
"""

import json
import re
import random
from pathlib import Path
from collections import Counter

DATA_PATH = Path("../guardreasoner/guardreasoner_data/all_combined.json")
OUTPUT_FILE = "train_12k.json"
TARGET_SAMPLES = 12000


def extract_label(output_text):
    """Extract safe/unsafe label from GuardReasoner reasoning output."""
    text = output_text.lower()

    # Look for "Answers:" section at end
    # Format: "Request: harmful" or "Request: unharmful"
    if "request: harmful" in text:
        return "harmful"
    elif "request: unharmful" in text:
        return "safe"
    elif "request: benign" in text:
        return "safe"

    return None


def extract_prompt(input_text):
    """Extract user prompt from GuardReasoner input format."""
    # Remove "Human user:" prefix
    text = input_text.strip()
    if text.startswith("Human user:"):
        text = text[11:].strip()

    # If there's an AI response, just take the user part
    if "\nAI assistant:" in text:
        text = text.split("\nAI assistant:")[0].strip()

    return text


def main():
    print("=" * 60)
    print("BUILDING 12K TRAINING DATASET")
    print("=" * 60)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} samples")

    # Convert format
    print("\nConverting format...")
    converted = []

    for item in data:
        prompt = extract_prompt(item["input"])
        label = extract_label(item["output"])

        if prompt and label and len(prompt) > 10:
            converted.append({
                "text": prompt,
                "label": label,
                "source": item.get("source_split", "guardreasoner")
            })

    print(f"Converted {len(converted):,} samples")

    # Check distribution
    labels = Counter(d["label"] for d in converted)
    print(f"  Safe: {labels['safe']:,}")
    print(f"  Harmful: {labels['harmful']:,}")

    # Balance and sample
    print(f"\nBalancing to {TARGET_SAMPLES:,} samples...")

    safe = [d for d in converted if d["label"] == "safe"]
    harmful = [d for d in converted if d["label"] == "harmful"]

    # Sample balanced
    n_each = TARGET_SAMPLES // 2

    random.seed(42)
    sampled_safe = random.sample(safe, min(n_each, len(safe)))
    sampled_harmful = random.sample(harmful, min(n_each, len(harmful)))

    final_data = sampled_safe + sampled_harmful
    random.shuffle(final_data)

    # Final stats
    final_labels = Counter(d["label"] for d in final_data)
    print(f"\nFinal dataset: {len(final_data):,} samples")
    print(f"  Safe: {final_labels['safe']:,} ({final_labels['safe']/len(final_data)*100:.1f}%)")
    print(f"  Harmful: {final_labels['harmful']:,} ({final_labels['harmful']/len(final_data)*100:.1f}%)")

    # Save
    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Compare to mega
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print("Original (800 samples)")
    print("Mega (2,540 samples) - 3.2x")
    print(f"NEW (12,000 samples) - 15x")

    return final_data


if __name__ == "__main__":
    main()
