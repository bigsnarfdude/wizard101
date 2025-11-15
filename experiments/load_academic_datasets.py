#!/usr/bin/env python3
"""
Academic Benchmark Dataset Loader for Multi-Policy Safety Evaluation

Uses well-known, peer-reviewed datasets from academic research:
1. ToxicChat (LMSYS, 2023) - Real-world toxicity
2. XSTest (Röttger et al., NAACL 2024) - Over-refusal detection
3. SimpleSafetyTests (Vidgen et al., 2023) - Critical safety risks
4. HarmBench (Mazeika et al., 2024) - Automated red-teaming
5. StrongREJECT (Souly et al., 2024) - Jailbreak evaluation

All datasets are publicly available on HuggingFace.
"""

from datasets import load_dataset
from typing import Dict, List, Tuple
import random
import json


def load_toxicchat(num_samples: int = 100) -> List[Dict[str, any]]:
    """
    ToxicChat: Real-world User-AI Interaction Toxicity

    Paper: Lin et al. (2023) "ToxicChat: Unveiling Hidden Challenges of Toxicity
           Detection in Real-World User-AI Conversation"
    Source: https://huggingface.co/datasets/lmsys/toxic-chat
    Size: 10,166 test samples

    Coverage: Hate speech, harassment (general toxicity)

    Args:
        num_samples: Number of samples to load

    Returns:
        List of dicts with content, labels, source
    """
    print(f"Loading ToxicChat dataset ({num_samples} samples)...")
    print("  Paper: Lin et al. (2023)")
    print("  Source: lmsys/toxic-chat")

    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    test_data = ds['test']

    # Sample balanced data
    toxic_samples = [x for x in test_data if x['toxicity'] == 1]
    safe_samples = [x for x in test_data if x['toxicity'] == 0]

    n = num_samples // 2
    toxic_subset = random.sample(toxic_samples, min(n, len(toxic_samples)))
    safe_subset = random.sample(safe_samples, min(n, len(safe_samples)))

    data = []

    for item in toxic_subset:
        # Map toxicity to our 6 policies (primarily hate_speech + harassment)
        data.append({
            "content": item['user_input'],
            "labels": ["hate_speech", "harassment"],  # General toxicity
            "source": "ToxicChat",
            "true_label": "toxic"
        })

    for item in safe_subset:
        data.append({
            "content": item['user_input'],
            "labels": [],
            "source": "ToxicChat",
            "true_label": "safe"
        })

    random.shuffle(data)

    print(f"  ✓ Loaded {len(data)} samples (toxic: {len(toxic_subset)}, safe: {len(safe_subset)})")
    return data


def load_xstest(num_samples: int = 100) -> List[Dict[str, any]]:
    """
    XSTest: Exaggerated Safety Behavior Detection

    Paper: Röttger et al. (NAACL 2024) "XSTest: A Test Suite for Identifying
           Exaggerated Safety Behaviours in Large Language Models"
    Source: https://huggingface.co/datasets/walledai/XSTest
    Size: 450 prompts (250 safe, 200 unsafe)

    Coverage: Tests over-refusal (false positives)

    Args:
        num_samples: Number of samples to load

    Returns:
        List of dicts with content, labels, source
    """
    print(f"\nLoading XSTest dataset ({num_samples} samples)...")
    print("  Paper: Röttger et al. (NAACL 2024)")
    print("  Source: walledai/XSTest")

    # Try multiple sources (some may be gated)
    try:
        ds = load_dataset("Paul/XSTest")
    except:
        try:
            ds = load_dataset("walledai/XSTest")
        except:
            ds = load_dataset("natolambert/xstest-v2-copy")

    # XSTest structure: single 'train' split with 'label' field
    all_prompts = list(ds['train'])

    # Filter by safe/unsafe label
    safe_prompts = [x for x in all_prompts if x['label'] == 'safe']
    unsafe_prompts = [x for x in all_prompts if x['label'] == 'unsafe']

    # Sample balanced
    n_safe = min(num_samples // 2, len(safe_prompts))
    n_unsafe = min(num_samples - n_safe, len(unsafe_prompts))

    safe_subset = random.sample(safe_prompts, n_safe)
    unsafe_subset = random.sample(unsafe_prompts, n_unsafe)

    data = []

    # Safe prompts should NOT be flagged (testing over-refusal)
    for item in safe_subset:
        data.append({
            "content": item['prompt'],
            "labels": [],  # Should be safe!
            "source": "XSTest",
            "true_label": "safe",
            "xstest_type": item.get('type', 'unknown')
        })

    # Unsafe prompts should be flagged
    for item in unsafe_subset:
        # Map XSTest categories to our 6 policies
        category = item.get('type', 'unknown')
        labels = map_xstest_category(category)

        data.append({
            "content": item['prompt'],
            "labels": labels,
            "source": "XSTest",
            "true_label": "unsafe",
            "xstest_type": category
        })

    random.shuffle(data)

    print(f"  ✓ Loaded {len(data)} samples (safe: {n_safe}, unsafe: {n_unsafe})")
    return data


def load_simplesafetytests(num_samples: int = 100) -> List[Dict[str, any]]:
    """
    SimpleSafetyTests: Critical Safety Risk Identification

    Paper: Vidgen et al. (2023) "SimpleSafetyTests: a Test Suite for Identifying
           Critical Safety Risks in Large Language Models"
    Source: https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests
    Size: 100 prompts across 5 harm areas

    Coverage: Suicide/self-harm, physical harm, illegal items, scams, child abuse

    Args:
        num_samples: Number of samples to load (max 100)

    Returns:
        List of dicts with content, labels, source
    """
    print(f"\nLoading SimpleSafetyTests ({min(num_samples, 100)} samples)...")
    print("  Paper: Vidgen et al. (2023)")
    print("  Source: Bertievidgen/SimpleSafetyTests")

    ds = load_dataset("Bertievidgen/SimpleSafetyTests")

    # SimpleSafetyTests has 5 harm areas
    all_prompts = list(ds['test'])

    # Sample up to num_samples (max 100 available)
    subset = random.sample(all_prompts, min(num_samples, len(all_prompts)))

    data = []

    for item in subset:
        # Map SST harm areas to our 6 policies
        harm_area = item.get('harm_area', 'unknown')
        labels = map_sst_harm_area(harm_area)

        data.append({
            "content": item['prompt'],
            "labels": labels,
            "source": "SimpleSafetyTests",
            "true_label": "unsafe",  # All SST prompts are unsafe
            "harm_area": harm_area
        })

    print(f"  ✓ Loaded {len(data)} samples (all unsafe)")
    return data


def load_harmbench(num_samples: int = 100) -> List[Dict[str, any]]:
    """
    HarmBench: Automated Red-Teaming Evaluation

    Paper: Mazeika et al. (2024) "HarmBench: A Standardized Evaluation Framework
           for Automated Red Teaming and Robust Refusal"
    Source: https://huggingface.co/datasets/walledai/HarmBench
    Size: 400 prompts across 7 semantic categories

    Coverage: Cybercrime, weapons, copyright, misinformation, harassment, illegal

    Args:
        num_samples: Number of samples to load

    Returns:
        List of dicts with content, labels, source
    """
    print(f"\nLoading HarmBench ({num_samples} samples)...")
    print("  Paper: Mazeika et al. (2024)")
    print("  Source: walledai/HarmBench")

    try:
        ds = load_dataset("walledai/HarmBench")
        all_prompts = list(ds['test'])

        # Sample
        subset = random.sample(all_prompts, min(num_samples, len(all_prompts)))

        data = []

        for item in subset:
            # Map HarmBench categories to our 6 policies
            category = item.get('category', 'unknown')
            labels = map_harmbench_category(category)

            data.append({
                "content": item['prompt'],
                "labels": labels,
                "source": "HarmBench",
                "true_label": "unsafe",
                "harmbench_category": category
            })

        print(f"  ✓ Loaded {len(data)} samples")
        return data

    except Exception as e:
        print(f"  ⚠ Could not load HarmBench: {e}")
        print(f"  Using fallback...")
        return []


# Category mapping functions

def map_xstest_category(category: str) -> List[str]:
    """Map XSTest categories to our 6 policies"""
    mapping = {
        # Safe prompts (should map to [])
        'homonyms': [],
        'figurative': [],
        'safe_targets': [],
        'historical_events': [],

        # Unsafe prompts
        'violence': ['violence'],
        'hate': ['hate_speech'],
        'self-harm': ['self_harm'],
        'sexual': ['sexual_minors'],
        'harassment': ['harassment'],
        'illegal': ['illegal'],
    }
    return mapping.get(category.lower(), ['illegal'])  # Default unsafe


def map_sst_harm_area(harm_area: str) -> List[str]:
    """Map SimpleSafetyTests harm areas to our 6 policies"""
    mapping = {
        'Suicide and Self-Harm': ['self_harm'],
        'Physical harm': ['violence'],
        'Illegal and Highly Regulated Items': ['illegal'],
        'Scams and Fraud': ['illegal', 'harassment'],
        'Child Abuse': ['sexual_minors', 'violence'],
    }
    return mapping.get(harm_area, ['illegal'])


def map_harmbench_category(category: str) -> List[str]:
    """Map HarmBench categories to our 6 policies"""
    mapping = {
        'Cybercrime & Unauthorized Intrusion': ['illegal'],
        'Chemical & Biological Weapons/Drugs': ['illegal', 'violence'],
        'Copyright Violations': ['illegal'],
        'Misinformation & Disinformation': ['harassment'],
        'Harassment & Bullying': ['harassment'],
        'Illegal Activities': ['illegal'],
        'General Harm': ['violence'],
    }
    return mapping.get(category, ['illegal'])


def create_academic_benchmark(
    toxicchat_n: int = 40,
    xstest_n: int = 30,
    sst_n: int = 20,
    harmbench_n: int = 10
) -> List[Dict[str, any]]:
    """
    Create comprehensive benchmark from academic datasets

    Args:
        toxicchat_n: ToxicChat samples (real-world toxicity)
        xstest_n: XSTest samples (over-refusal detection)
        sst_n: SimpleSafetyTests samples (critical risks)
        harmbench_n: HarmBench samples (adversarial)

    Returns:
        Combined benchmark dataset
    """
    print("\n" + "="*80)
    print("CREATING ACADEMIC BENCHMARK DATASET")
    print("="*80)

    all_data = []

    # Load each dataset
    if toxicchat_n > 0:
        all_data.extend(load_toxicchat(toxicchat_n))

    if xstest_n > 0:
        all_data.extend(load_xstest(xstest_n))

    if sst_n > 0:
        all_data.extend(load_simplesafetytests(sst_n))

    if harmbench_n > 0:
        harmbench_data = load_harmbench(harmbench_n)
        if harmbench_data:
            all_data.extend(harmbench_data)

    # Shuffle
    random.shuffle(all_data)

    print("\n" + "="*80)
    print(f"BENCHMARK READY: {len(all_data)} samples")
    print("="*80)
    print(f"  ToxicChat: {toxicchat_n}")
    print(f"  XSTest: {xstest_n}")
    print(f"  SimpleSafetyTests: {sst_n}")
    print(f"  HarmBench: {harmbench_n}")

    # Distribution by source
    sources = {}
    for item in all_data:
        src = item['source']
        sources[src] = sources.get(src, 0) + 1

    print(f"\nActual distribution:")
    for src, count in sources.items():
        print(f"  {src}: {count}")

    return all_data


def save_dataset(data: List[Dict[str, any]], filename: str):
    """Save dataset to JSON file"""
    filepath = f"/Users/vincent/development/wizard101/experiments/{filename}"

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Dataset saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Create academic benchmark (100 samples)
    benchmark = create_academic_benchmark(
        toxicchat_n=40,   # Real-world toxicity
        xstest_n=30,      # Over-refusal detection
        sst_n=20,         # Critical safety risks
        harmbench_n=10    # Adversarial red-teaming
    )

    # Save to file
    save_dataset(benchmark, "academic_benchmark.json")

    # Show samples
    print("\n" + "="*80)
    print("SAMPLE TEST CASES")
    print("="*80)

    for i, item in enumerate(benchmark[:5], 1):
        print(f"\n{i}. [{item['source']}] {item['content'][:70]}...")
        print(f"   Labels: {item['labels']}")
        print(f"   True Label: {item['true_label']}")
