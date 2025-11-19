#!/usr/bin/env python3
"""
Download and explore GuardReasonerTrain dataset for 3-task training.

This script downloads the dataset and shows statistics.

Usage:
    python download_3task_dataset.py
"""

from datasets import load_dataset
import json

DATASET_NAME = "yueliu1999/GuardReasonerTrain"

def main():
    print("=" * 60)
    print("Downloading GuardReasonerTrain Dataset")
    print("=" * 60)

    # Download dataset
    print(f"\nLoading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME)

    # Show splits
    print("\nAvailable splits:")
    for split_name in ds.keys():
        split = ds[split_name]
        print(f"  - {split_name}: {len(split)} samples")

    total = sum(len(ds[s]) for s in ds.keys())
    print(f"\nTotal samples: {total}")

    # Show sample from WildGuardTrainR (main split)
    print("\n" + "=" * 60)
    print("Sample from WildGuardTrainR")
    print("=" * 60)

    sample = ds["WildGuardTrainR"][0]

    print("\nFields available:")
    for key in sample.keys():
        print(f"  - {key}")

    print("\n--- Instruction (truncated) ---")
    print(sample["instruction"][:500] + "...")

    print("\n--- Input (truncated) ---")
    print(sample["input"][:500] + "...")

    print("\n--- Output (truncated) ---")
    print(sample["output"][:500] + "...")

    print("\n--- Labels ---")
    print(f"  prompt_harm_label: {sample.get('prompt_harm_label', 'N/A')}")
    print(f"  response_refusal_label: {sample.get('response_refusal_label', 'N/A')}")
    print(f"  response_harm_label: {sample.get('response_harm_label', 'N/A')}")

    # Label distribution
    print("\n" + "=" * 60)
    print("Label Distribution (WildGuardTrainR)")
    print("=" * 60)

    split = ds["WildGuardTrainR"]

    # Count labels
    prompt_harm = {"harmful": 0, "unharmful": 0}
    refusal = {"refusal": 0, "compliance": 0}
    response_harm = {"harmful": 0, "unharmful": 0}

    for example in split:
        ph = example.get("prompt_harm_label", "")
        rf = example.get("response_refusal_label", "")
        rh = example.get("response_harm_label", "")

        if ph in prompt_harm:
            prompt_harm[ph] += 1
        if rf in refusal:
            refusal[rf] += 1
        if rh in response_harm:
            response_harm[rh] += 1

    print("\nPrompt Harm:")
    for k, v in prompt_harm.items():
        pct = v / len(split) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    print("\nRefusal:")
    for k, v in refusal.items():
        pct = v / len(split) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    print("\nResponse Harm:")
    for k, v in response_harm.items():
        pct = v / len(split) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    # Save sample for inspection
    output_file = "sample_3task_data.json"
    samples = [ds["WildGuardTrainR"][i] for i in range(5)]
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n✅ Saved 5 samples to {output_file}")

    print("\n" + "=" * 60)
    print("Dataset ready for training!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review sample_3task_data.json")
    print("  2. Run: python train_3task_test.py --dry-run")
    print("  3. Run: python train_3task_test.py --test")

if __name__ == "__main__":
    main()
