#!/usr/bin/env python3
"""
Quick script to generate 10K high-quality reasoning samples using Gemini.

This demonstrates the full pipeline:
1. Download source data from HuggingFace
2. Generate responses where needed
3. Create reasoning traces
4. Validate quality
5. Save dataset

Estimated time: 2-3 hours
Estimated cost: $2-3 with Gemini 2.0 Flash
"""

import json
from datasets import load_dataset
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from generate_with_gemini import GeminiReasoningGenerator


def download_source_data(num_samples=10000):
    """Download diverse source prompts from HuggingFace."""
    print("Downloading source data from HuggingFace...")

    # Option 1: Use WildGuard (high quality, adversarial)
    print("Loading WildGuardMix...")
    wildguard = load_dataset("allenai/wildguardmix", split="train", streaming=True)
    wildguard_samples = []

    for i, item in enumerate(wildguard):
        if i >= num_samples // 2:  # Get half from WildGuard
            break
        wildguard_samples.append({
            "prompt": item.get("prompt", ""),
            "response": item.get("response", None)  # May be None - will generate
        })

    print(f"Loaded {len(wildguard_samples)} WildGuard samples")

    # Option 2: Use ToxicChat (toxic language detection)
    print("Loading ToxicChat...")
    toxic = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    toxic_samples = []

    for i, item in enumerate(toxic):
        if i >= num_samples // 4:  # Get 25% from ToxicChat
            break
        toxic_samples.append({
            "prompt": item.get("user_input", ""),
            "response": None  # Will generate
        })

    print(f"Loaded {len(toxic_samples)} ToxicChat samples")

    # Option 3: Use your existing datasets
    print("Loading existing harmful_behaviors...")
    try:
        with open("../harmful_behaviors_train.json") as f:
            harmful = json.load(f)[:num_samples // 4]
    except FileNotFoundError:
        harmful = []
        print("harmful_behaviors_train.json not found, skipping")

    # Combine sources
    all_samples = wildguard_samples + toxic_samples + harmful

    print(f"\nTotal source samples: {len(all_samples)}")
    return all_samples[:num_samples]


def main():
    """Generate 10K reasoning samples."""
    print("="*70)
    print("GEMINI REASONING DATASET GENERATOR")
    print("="*70)
    print("\nTarget: 10,000 high-quality samples")
    print("Estimated time: 2-3 hours")
    print("Estimated cost: $2-3\n")

    # Step 1: Download source data
    print("STEP 1: Downloading source data...")
    source_data = download_source_data(num_samples=10000)

    # Save source data
    source_path = Path("../data/gemini_source_data.json")
    source_path.parent.mkdir(parents=True, exist_ok=True)

    with open(source_path, 'w') as f:
        json.dump(source_data, f, indent=2)

    print(f"Saved source data to {source_path}")

    # Step 2: Generate reasoning traces
    print("\nSTEP 2: Generating reasoning traces with Gemini...")
    generator = GeminiReasoningGenerator(
        model_name="gemini-2.0-flash-exp",
        rate_limit_delay=0.3  # Fast but respectful
    )

    results = generator.process_dataset(source_data)

    # Step 3: Save results
    output_path = Path("../data/gemini_reasoning_10k.json")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Generated {len(results)} high-quality samples")
    print(f"✅ Saved to {output_path}")

    # Step 4: Show statistics
    generator.print_stats()

    # Step 5: Quality analysis
    print("\nQUALITY ANALYSIS")
    print("="*70)

    # Check label distribution
    label_counts = {
        "harmful_prompts": 0,
        "unharmful_prompts": 0,
        "refusals": 0,
        "compliance": 0,
        "harmful_responses": 0,
        "unharmful_responses": 0
    }

    for sample in results:
        if sample["prompt_harm_label"] == "harmful":
            label_counts["harmful_prompts"] += 1
        else:
            label_counts["unharmful_prompts"] += 1

        if sample["response_refusal_label"] == "refusal":
            label_counts["refusals"] += 1
        else:
            label_counts["compliance"] += 1

        if sample["response_harm_label"] == "harmful":
            label_counts["harmful_responses"] += 1
        else:
            label_counts["unharmful_responses"] += 1

    print(f"Harmful prompts: {label_counts['harmful_prompts']} ({label_counts['harmful_prompts']/len(results)*100:.1f}%)")
    print(f"Unharmful prompts: {label_counts['unharmful_prompts']} ({label_counts['unharmful_prompts']/len(results)*100:.1f}%)")
    print(f"Refusals: {label_counts['refusals']} ({label_counts['refusals']/len(results)*100:.1f}%)")
    print(f"Compliance: {label_counts['compliance']} ({label_counts['compliance']/len(results)*100:.1f}%)")
    print(f"Harmful responses: {label_counts['harmful_responses']} ({label_counts['harmful_responses']/len(results)*100:.1f}%)")
    print(f"Unharmful responses: {label_counts['unharmful_responses']} ({label_counts['unharmful_responses']/len(results)*100:.1f}%)")

    # Show examples
    print("\n" + "="*70)
    print("EXAMPLE SAMPLES")
    print("="*70)

    for i, sample in enumerate(results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {sample['original_prompt'][:100]}...")
        print(f"Response: {sample['generated_response'][:100]}...")
        print(f"Labels: {sample['prompt_harm_label']} / {sample['response_refusal_label']} / {sample['response_harm_label']}")
        print(f"Reasoning preview:\n{sample['output'][:200]}...")

    print("\n" + "="*70)
    print("✅ COMPLETE! Dataset ready for training.")
    print("="*70)

    print(f"\nNext steps:")
    print(f"1. Review samples in {output_path}")
    print(f"2. Combine with GuardReasonerTrain dataset (optional)")
    print(f"3. Start R-SFT training with combined dataset")
    print(f"4. Expected improvement: +5-10% accuracy with more diverse data")


if __name__ == "__main__":
    main()
