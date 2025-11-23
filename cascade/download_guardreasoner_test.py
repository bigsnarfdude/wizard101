#!/usr/bin/env python3
"""
Download GuardReasoner Test Dataset from HuggingFace

Test set: https://huggingface.co/datasets/yueliu1999/GuardReasonerTest

Usage:
    python download_guardreasoner_test.py
"""

import json
from pathlib import Path
from datasets import load_dataset


def main():
    print("Downloading GuardReasoner Test Dataset...")

    # Load from HuggingFace
    dataset = load_dataset("yueliu1999/GuardReasonerTest")

    print(f"Dataset splits: {list(dataset.keys())}")

    # Process and save
    output_dir = Path("../guardreasoner/guardreasoner_data")
    output_dir.mkdir(exist_ok=True)

    all_test = []

    for split_name in dataset.keys():
        split_data = dataset[split_name]
        print(f"\n{split_name}: {len(split_data)} samples")

        # Convert to list
        samples = [dict(item) for item in split_data]
        all_test.extend(samples)

        # Save individual split
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"  Saved to {output_path}")

    # Save combined test set
    combined_path = output_dir / "all_test_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_test, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total test samples: {len(all_test)}")
    print(f"Combined test set: {combined_path}")

    # Also create a simpler format for cascade evaluation
    simple_test = []
    for item in all_test:
        # Extract prompt from input
        input_text = item.get("input", "")
        if input_text.startswith("Human user:"):
            prompt = input_text[11:].strip()
            if "\nAI assistant:" in prompt:
                prompt = prompt.split("\nAI assistant:")[0].strip()
        else:
            prompt = input_text

        # Extract label from output
        output_text = item.get("output", "").lower()
        if "request: harmful" in output_text:
            label = "harmful"
        elif "request: unharmful" in output_text:
            label = "safe"
        else:
            continue  # Skip if can't determine label

        simple_test.append({
            "prompt": prompt,
            "label": label,
            "source": item.get("source_split", "guardreasoner_test")
        })

    # Save simple format
    simple_path = Path("guardreasoner_test.json")
    with open(simple_path, "w") as f:
        json.dump(simple_test, f, indent=2)

    print(f"\nSimple format: {simple_path}")
    print(f"  {len(simple_test)} samples")

    # Count labels
    safe_count = sum(1 for s in simple_test if s["label"] == "safe")
    harmful_count = len(simple_test) - safe_count
    print(f"  Safe: {safe_count} ({safe_count/len(simple_test)*100:.1f}%)")
    print(f"  Harmful: {harmful_count} ({harmful_count/len(simple_test)*100:.1f}%)")


if __name__ == "__main__":
    main()
