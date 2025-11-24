#!/usr/bin/env python3
"""
Build Mega Training Dataset for L0 Bouncer

Combines all available datasets into one unified training set.

Sources:
- combined_train.json (800 samples)
- wildguard_full_benchmark.json (1554 samples)
- larger_benchmark.json (1650 samples)

Output: mega_train.json (~4000+ samples)
"""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("../")
OUTPUT_FILE = "mega_train.json"


def normalize_label(item):
    """Convert various label formats to safe/harmful."""

    # Check for explicit label field
    if "label" in item:
        label = item["label"].lower()
        if label in ["harmful", "unsafe"]:
            return "harmful"
        elif label in ["safe", "harmless", "benign"]:
            return "safe"

    # Check for labels array (WildGuard format)
    if "labels" in item:
        labels = item["labels"]
        if labels:  # Non-empty = harmful
            return "harmful"
        else:  # Empty = safe
            return "safe"

    # Check source (WildGuard format)
    if "source" in item:
        source = item["source"].lower()
        if "benign" in source:
            return "safe"
        else:
            return "harmful"

    # Default to safe if uncertain
    return "safe"


def get_text(item):
    """Extract text from various formats."""
    if "prompt" in item:
        return item["prompt"]
    if "text" in item:
        return item["text"]
    if "content" in item:
        return item["content"]
    return ""


def load_dataset(path, name):
    """Load and normalize a dataset."""
    if not path.exists():
        print(f"  {name}: NOT FOUND")
        return []

    with open(path) as f:
        data = json.load(f)

    normalized = []
    for item in data:
        text = get_text(item)
        if not text:
            continue

        label = normalize_label(item)
        normalized.append({
            "text": text,
            "label": label,
            "source": name,
        })

    # Stats
    labels = Counter(d["label"] for d in normalized)
    print(f"  {name}: {len(normalized)} samples ({labels['safe']} safe, {labels['harmful']} harmful)")

    return normalized


def main():
    print("=" * 60)
    print("BUILDING MEGA TRAINING DATASET")
    print("=" * 60)

    all_data = []

    # Load all datasets
    print("\nLoading datasets:")

    # Primary datasets
    all_data.extend(load_dataset(DATA_DIR / "combined_train.json", "combined_train"))
    all_data.extend(load_dataset(DATA_DIR / "wildguard_full_benchmark.json", "wildguard_full"))
    all_data.extend(load_dataset(DATA_DIR / "larger_benchmark.json", "larger_benchmark"))
    all_data.extend(load_dataset(DATA_DIR / "wildguard_benchmark.json", "wildguard_bench"))

    # Deduplicate by text
    print("\nDeduplicating...")
    seen = set()
    unique_data = []
    for item in all_data:
        text_hash = hash(item["text"][:100])  # First 100 chars
        if text_hash not in seen:
            seen.add(text_hash)
            unique_data.append(item)

    print(f"  Before: {len(all_data)}")
    print(f"  After: {len(unique_data)}")

    # Final stats
    labels = Counter(d["label"] for d in unique_data)
    sources = Counter(d["source"] for d in unique_data)

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)
    print(f"\nTotal samples: {len(unique_data)}")
    print(f"  Safe: {labels['safe']} ({labels['safe']/len(unique_data)*100:.1f}%)")
    print(f"  Harmful: {labels['harmful']} ({labels['harmful']/len(unique_data)*100:.1f}%)")

    print(f"\nBy source:")
    for source, count in sources.most_common():
        print(f"  {source}: {count}")

    # Save
    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(unique_data, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")

    # Compare to original
    print("\n" + "=" * 60)
    print("IMPROVEMENT")
    print("=" * 60)
    original = 800
    print(f"Original training: {original} samples")
    print(f"Mega training: {len(unique_data)} samples")
    print(f"Improvement: {len(unique_data)/original:.1f}x more data")

    return unique_data


if __name__ == "__main__":
    main()
