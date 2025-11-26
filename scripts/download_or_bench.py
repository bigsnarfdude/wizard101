#!/usr/bin/env python3
"""
Download OR-Bench (Over-Refusal Benchmark) dataset and normalize to JSON format.

OR-Bench by bench-llm:
Tests LLM over-refusal behavior with prompts that seem harmful but are actually safe.

Source: https://huggingface.co/datasets/bench-llm/or-bench

Dataset subsets:
- or-bench-80k: 80,400 samples - seemingly toxic but SAFE (tests over-refusal)
- or-bench-hard-1k: 1,320 samples - harder edge cases, SAFE
- or-bench-toxic: 655 samples - actually HARMFUL

Total: 82,375 samples
"""

import json
import os
from datasets import load_dataset


def get_token():
    """Get HuggingFace token from standard locations."""
    for token_path in ['~/.huggingface/token', '~/.cache/huggingface/token']:
        expanded = os.path.expanduser(token_path)
        if os.path.exists(expanded):
            with open(expanded) as f:
                return f.read().strip()
    return None


def download_or_bench():
    """Download and normalize OR-Bench dataset."""

    token = get_token()
    normalized = []
    categories = set()

    # Subset configurations with their ground truth labels
    subsets = [
        ("or-bench-80k", "safe"),      # Seemingly toxic but safe
        ("or-bench-hard-1k", "safe"),  # Harder edge cases, still safe
        ("or-bench-toxic", "harmful"), # Actually toxic
    ]

    for subset_name, ground_truth_label in subsets:
        print(f"\nDownloading OR-Bench subset: {subset_name}...")
        try:
            ds = load_dataset(
                "bench-llm/or-bench",
                subset_name,
                token=token
            )
            data = ds['train']
            print(f"  Loaded {len(data)} samples (label={ground_truth_label})")

            for sample in data:
                prompt = sample.get('prompt', '').strip()
                category = sample.get('category', 'unknown')

                if not prompt:
                    continue

                categories.add(category)

                normalized.append({
                    'text': prompt,
                    'label': ground_truth_label,
                    'source': f'or-bench-{subset_name}',
                    'category': category
                })
        except Exception as e:
            print(f"  Error loading {subset_name}: {e}")

    # Count by label
    safe_count = sum(1 for s in normalized if s['label'] == 'safe')
    harmful_count = sum(1 for s in normalized if s['label'] == 'harmful')

    print(f"\nNormalized dataset:")
    print(f"  Total samples: {len(normalized)}")
    print(f"  Safe: {safe_count}")
    print(f"  Harmful: {harmful_count}")
    print(f"  Categories: {sorted(categories)}")

    # Save combined dataset
    output_path = '/Users/vincent/development/wizard101/data/benchmark/or_bench.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")

    # Show samples
    print("\nSample SAFE entry (should not be refused):")
    for s in normalized:
        if s['label'] == 'safe':
            print(f"  Text: {s['text'][:100]}...")
            break

    print("\nSample HARMFUL entry (should be refused):")
    for s in normalized:
        if s['label'] == 'harmful':
            print(f"  Text: {s['text'][:100]}...")
            break

    return output_path, len(normalized)


if __name__ == '__main__':
    output_path, count = download_or_bench()
    print(f"\nDone! {count} samples saved to {output_path}")
