#!/usr/bin/env python3
"""Download SALAD-Bench dataset from HuggingFace."""

from datasets import load_dataset
import json
from pathlib import Path


def download_salad_bench():
    """Download SALAD-Bench dataset from HuggingFace."""

    output_dir = Path(__file__).parent.parent.parent / "data" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download base set (primary evaluation)
    print("Downloading SALAD-Bench base_set (21K samples)...")
    base = load_dataset("OpenSafetyLab/Salad-Data", name="base_set", split="train")

    # Convert to our format
    samples = []
    for row in base:
        samples.append({
            "id": row.get("qid"),
            "prompt": row.get("question"),
            "source": row.get("source"),
            "category_1": row.get("1-category"),
            "category_2": row.get("2-category"),
            "category_3": row.get("3-category"),
            "label": "harmful"  # All SALAD-Bench questions are harmful
        })

    output_path = output_dir / "salad_bench_base.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples):,} samples to {output_path}")

    # Download attack-enhanced set
    print("\nDownloading SALAD-Bench attack_enhanced_set (5K samples)...")
    attack = load_dataset("OpenSafetyLab/Salad-Data", name="attack_enhanced_set", split="train")

    attack_samples = []
    for row in attack:
        attack_samples.append({
            "id": row.get("aid"),
            "prompt": row.get("augq"),
            "method": row.get("method"),
            "label": "harmful"
        })

    attack_path = output_dir / "salad_bench_attack.json"
    with open(attack_path, "w") as f:
        json.dump(attack_samples, f, indent=2)

    print(f"Saved {len(attack_samples):,} samples to {attack_path}")

    # Summary
    print("\n" + "=" * 50)
    print("SALAD-Bench Download Complete")
    print("=" * 50)
    print(f"Base set:   {len(samples):,} samples")
    print(f"Attack set: {len(attack_samples):,} samples")
    print(f"Total:      {len(samples) + len(attack_samples):,} samples")
    print(f"Location:   {output_dir}")


if __name__ == "__main__":
    download_salad_bench()
