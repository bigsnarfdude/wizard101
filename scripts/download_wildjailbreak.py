#!/usr/bin/env python3
"""
Download WildJailbreak dataset and normalize to JSON format.

WildJailbreak by Liang et al. (Dec 2024):
"WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models"

Source: https://huggingface.co/datasets/allenai/wildjailbreak

Note: If WildJailbreak access is denied, this script will fall back to WildGuardMix
which contains similar data from the same team (AllenAI).
"""

import json
import os
import csv
from huggingface_hub import hf_hub_download
from datasets import load_dataset


def get_token():
    """Get HuggingFace token from standard locations."""
    for token_path in ['~/.huggingface/token', '~/.cache/huggingface/token']:
        expanded = os.path.expanduser(token_path)
        if os.path.exists(expanded):
            with open(expanded) as f:
                return f.read().strip()
    return None


def download_wildjailbreak(token):
    """Try to download WildJailbreak dataset."""
    train_data = []
    eval_data = []

    # Download train.tsv using huggingface_hub
    print("Downloading WildJailbreak training set...")
    train_file = hf_hub_download(
        repo_id="allenai/wildjailbreak",
        filename="train/train.tsv",
        repo_type="dataset",
        token=token
    )

    # Parse train.tsv
    print("Parsing training data...")
    with open(train_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            train_data.append(row)

    print(f"Loaded {len(train_data)} samples from training set")

    # Download eval.tsv
    print("Downloading WildJailbreak eval set...")
    eval_file = hf_hub_download(
        repo_id="allenai/wildjailbreak",
        filename="eval/eval.tsv",
        repo_type="dataset",
        token=token
    )

    # Parse eval.tsv
    print("Parsing eval data...")
    with open(eval_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            eval_data.append(row)

    print(f"Loaded {len(eval_data)} samples from eval set")

    return train_data, eval_data


def download_wildguardmix(token):
    """Download WildGuardMix as fallback."""
    print("\n" + "="*60)
    print("WildJailbreak access denied - using WildGuardMix instead")
    print("WildGuardMix is from the same team (AllenAI) with similar data")
    print("="*60 + "\n")

    print("Downloading WildGuardMix training set...")
    train_ds = load_dataset(
        'allenai/wildguardmix',
        'wildguardtrain',
        token=token
    )
    print(f"Loaded {len(train_ds['train'])} samples from training set")

    print("Downloading WildGuardMix test set...")
    test_ds = load_dataset(
        'allenai/wildguardmix',
        'wildguardtest',
        token=token
    )
    print(f"Loaded {len(test_ds['test'])} samples from test set")

    return train_ds['train'], test_ds['test']


def normalize_wildjailbreak():
    """Download and normalize WildJailbreak dataset."""

    token = get_token()
    if not token:
        raise ValueError("No HuggingFace token found")

    # Try WildJailbreak first, fall back to WildGuardMix
    use_wildguardmix = False
    try:
        train_data, eval_data = download_wildjailbreak(token)
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "restricted" in error_msg.lower() or "403" in error_msg:
            train_data, eval_data = download_wildguardmix(token)
            use_wildguardmix = True
        else:
            raise

    normalized = []

    if use_wildguardmix:
        # Process WildGuardMix format
        for sample in train_data:
            prompt = sample.get('prompt', '').strip()
            harm_label = sample.get('prompt_harm_label', '')

            if not prompt:
                continue

            if harm_label == 'harmful':
                label = 'harmful'
            elif harm_label == 'unharmful':
                label = 'safe'
            else:
                continue

            normalized.append({
                'text': prompt,
                'label': label
            })

        for sample in eval_data:
            prompt = sample.get('prompt', '').strip()
            harm_label = sample.get('prompt_harm_label', '')

            if not prompt:
                continue

            if harm_label == 'harmful':
                label = 'harmful'
            elif harm_label == 'unharmful':
                label = 'safe'
            else:
                continue

            normalized.append({
                'text': prompt,
                'label': label
            })
    else:
        # Process WildJailbreak format
        for sample in train_data:
            data_type = sample.get('data_type', '')

            # Determine label
            if 'harmful' in data_type:
                label = 'harmful'
            elif 'benign' in data_type:
                label = 'safe'
            else:
                # Skip unknown types
                continue

            # Determine text: use adversarial if available, otherwise vanilla
            adversarial = sample.get('adversarial', '').strip()
            vanilla = sample.get('vanilla', '').strip()

            if adversarial:
                text = adversarial
            elif vanilla:
                text = vanilla
            else:
                # Skip empty samples
                continue

            normalized.append({
                'text': text,
                'label': label
            })

        # Process eval set
        for sample in eval_data:
            data_type = sample.get('data_type', '')

            # Determine label
            if 'harmful' in data_type:
                label = 'harmful'
            elif 'benign' in data_type:
                label = 'safe'
            else:
                continue

            # Determine text
            adversarial = sample.get('adversarial', '').strip()
            vanilla = sample.get('vanilla', '').strip()

            if adversarial:
                text = adversarial
            elif vanilla:
                text = vanilla
            else:
                continue

            normalized.append({
                'text': text,
                'label': label
            })

    # Count labels
    harmful_count = sum(1 for item in normalized if item['label'] == 'harmful')
    safe_count = sum(1 for item in normalized if item['label'] == 'safe')

    print(f"\nNormalized dataset:")
    print(f"  Total samples: {len(normalized)}")
    print(f"  Harmful: {harmful_count}")
    print(f"  Safe: {safe_count}")

    # Save to JSON
    output_path = '/Users/vincent/development/wizard101/data/benchmark/wildjailbreak.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")

    # Show samples
    print("\nSample harmful entry:")
    for item in normalized:
        if item['label'] == 'harmful':
            print(f"  Text: {item['text'][:100]}...")
            break

    print("\nSample safe entry:")
    for item in normalized:
        if item['label'] == 'safe':
            print(f"  Text: {item['text'][:100]}...")
            break

    return output_path, len(normalized)


if __name__ == '__main__':
    output_path, count = normalize_wildjailbreak()
    print(f"\nDone! {count} samples saved to {output_path}")
