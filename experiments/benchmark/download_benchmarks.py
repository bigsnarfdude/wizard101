#!/usr/bin/env python3
"""
Download comprehensive safety evaluation benchmarks.

Benchmarks included:
1. ToxicChat - Real user toxicity (10k test)
2. HarmBench - Jailbreak evaluation (400 prompts)
3. XSTest - Exaggerated safety / false positive testing (450)
4. AdvBench - Adversarial prompts (520)
5. OpenAI Moderation - Standard benchmark (1,680)
6. BeaverTails - Harm categories (30k test)
7. SimpleSafetyTests - Basic safety checks (100)

Usage:
    python download_benchmarks.py
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from collections import Counter

BENCHMARK_DIR = Path(__file__).parent
BENCHMARK_DIR.mkdir(exist_ok=True)


def download_toxicchat():
    """ToxicChat - Real user toxicity detection."""
    print("\n" + "="*60)
    print("Downloading ToxicChat...")
    print("="*60)

    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")

        samples = []
        for item in ds["test"]:
            samples.append({
                "text": item["user_input"],
                "label": "harmful" if item["toxicity"] == 1 else "safe",
                "source": "toxicchat",
                "category": "toxicity"
            })

        output_path = BENCHMARK_DIR / "toxicchat_test.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        dist = Counter(s["label"] for s in samples)
        print(f"  Saved: {len(samples)} samples")
        print(f"  Distribution: {dict(dist)}")
        return len(samples)
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def download_harmbench():
    """HarmBench - Jailbreak evaluation prompts."""
    print("\n" + "="*60)
    print("Downloading HarmBench...")
    print("="*60)

    try:
        ds = load_dataset("harmbench/harmbench", "standard")

        samples = []
        for item in ds["test"]:
            samples.append({
                "text": item["behavior"],
                "label": "harmful",
                "source": "harmbench",
                "category": item.get("category", "general")
            })

        output_path = BENCHMARK_DIR / "harmbench_test.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"  Saved: {len(samples)} samples")
        return len(samples)
    except Exception as e:
        print(f"  Error: {e}")
        # Try alternative
        try:
            ds = load_dataset("LLM-LAT/harmful-dataset")
            samples = []
            for item in ds["train"]:
                samples.append({
                    "text": item["prompt"],
                    "label": "harmful",
                    "source": "harmbench-alt",
                    "category": "general"
                })

            output_path = BENCHMARK_DIR / "harmbench_test.json"
            with open(output_path, "w") as f:
                json.dump(samples[:500], f, indent=2)

            print(f"  Saved: {min(500, len(samples))} samples (alternative)")
            return min(500, len(samples))
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")
            return 0


def download_xstest():
    """XSTest - Tests for exaggerated safety (false positives)."""
    print("\n" + "="*60)
    print("Downloading XSTest...")
    print("="*60)

    try:
        ds = load_dataset("walledai/XSTest")

        samples = []
        for item in ds["test"]:
            # XSTest contains safe prompts that might trigger false positives
            label = "safe" if item.get("label", "safe") in ["safe", "0", 0] else "harmful"
            samples.append({
                "text": item["prompt"],
                "label": label,
                "source": "xstest",
                "category": item.get("type", "exaggerated_safety")
            })

        output_path = BENCHMARK_DIR / "xstest.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        dist = Counter(s["label"] for s in samples)
        print(f"  Saved: {len(samples)} samples")
        print(f"  Distribution: {dict(dist)}")
        return len(samples)
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def download_advbench():
    """AdvBench - Adversarial harmful prompts."""
    print("\n" + "="*60)
    print("Downloading AdvBench...")
    print("="*60)

    try:
        ds = load_dataset("walledai/AdvBench")

        samples = []
        for item in ds["train"]:
            samples.append({
                "text": item["prompt"],
                "label": "harmful",
                "source": "advbench",
                "category": "adversarial"
            })

        output_path = BENCHMARK_DIR / "advbench.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"  Saved: {len(samples)} samples")
        return len(samples)
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def download_openai_moderation():
    """OpenAI Moderation benchmark."""
    print("\n" + "="*60)
    print("Downloading OpenAI Moderation...")
    print("="*60)

    try:
        ds = load_dataset("mmathys/openai-moderation-api-evaluation")

        samples = []
        for item in ds["train"]:
            # Check if any category is flagged
            is_harmful = any([
                item.get("S", False),   # Sexual
                item.get("H", False),   # Hate
                item.get("V", False),   # Violence
                item.get("HR", False),  # Harassment
                item.get("SH", False),  # Self-harm
                item.get("S3", False),  # Sexual/minors
                item.get("H2", False),  # Hate/threatening
                item.get("V2", False),  # Violence/graphic
            ])

            samples.append({
                "text": item["prompt"],
                "label": "harmful" if is_harmful else "safe",
                "source": "openai_moderation",
                "category": "moderation"
            })

        output_path = BENCHMARK_DIR / "openai_moderation.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        dist = Counter(s["label"] for s in samples)
        print(f"  Saved: {len(samples)} samples")
        print(f"  Distribution: {dict(dist)}")
        return len(samples)
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def download_beavertails():
    """BeaverTails - Harm categories (sample)."""
    print("\n" + "="*60)
    print("Downloading BeaverTails...")
    print("="*60)

    try:
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")

        samples = []
        for item in ds:
            # BeaverTails has detailed category labels
            is_harmful = item.get("is_safe", True) == False

            samples.append({
                "text": item["prompt"],
                "label": "harmful" if is_harmful else "safe",
                "source": "beavertails",
                "category": str(item.get("category", "general"))
            })

        # Save full dataset
        output_path = BENCHMARK_DIR / "beavertails_30k.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        dist = Counter(s["label"] for s in samples)
        print(f"  Saved: {len(samples)} samples")
        print(f"  Distribution: {dict(dist)}")
        return len(samples)
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def download_simplesafetytests():
    """SimpleSafetyTests - Basic safety checks."""
    print("\n" + "="*60)
    print("Downloading SimpleSafetyTests...")
    print("="*60)

    try:
        ds = load_dataset("Bertievidgen/SimpleSafetyTests")

        samples = []
        for item in ds["test"]:
            samples.append({
                "text": item["prompt"],
                "label": "harmful",  # All are harmful by design
                "source": "simplesafetytests",
                "category": item.get("harm_area", "general")
            })

        output_path = BENCHMARK_DIR / "simplesafetytests.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"  Saved: {len(samples)} samples")
        return len(samples)
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def create_combined_benchmark():
    """Create combined benchmark from all downloaded datasets."""
    print("\n" + "="*60)
    print("Creating Combined Benchmark...")
    print("="*60)

    all_samples = []

    for json_file in BENCHMARK_DIR.glob("*.json"):
        if json_file.name == "combined_benchmark.json":
            continue

        try:
            with open(json_file) as f:
                samples = json.load(f)
                all_samples.extend(samples)
                print(f"  Added {len(samples)} from {json_file.name}")
        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")

    # Save combined
    output_path = BENCHMARK_DIR / "combined_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    # Stats
    dist = Counter(s["label"] for s in all_samples)
    sources = Counter(s["source"] for s in all_samples)

    print(f"\nCombined Benchmark:")
    print(f"  Total: {len(all_samples)} samples")
    print(f"  Distribution: {dict(dist)}")
    print(f"  Sources: {dict(sources)}")

    return len(all_samples)


def main():
    print("="*60)
    print("DOWNLOADING SAFETY EVALUATION BENCHMARKS")
    print("="*60)

    totals = {}

    # Download each benchmark
    totals["toxicchat"] = download_toxicchat()
    totals["harmbench"] = download_harmbench()
    totals["xstest"] = download_xstest()
    totals["advbench"] = download_advbench()
    totals["openai_moderation"] = download_openai_moderation()
    totals["beavertails"] = download_beavertails()
    totals["simplesafetytests"] = download_simplesafetytests()

    # Create combined benchmark
    combined = create_combined_benchmark()

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    for name, count in totals.items():
        status = "OK" if count > 0 else "FAILED"
        print(f"  {name}: {count} samples [{status}]")

    print(f"\n  TOTAL COMBINED: {combined} samples")
    print(f"  Output directory: {BENCHMARK_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
