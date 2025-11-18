#!/usr/bin/env python3
"""
Create DPO Preference Dataset from Hard Samples

Takes mined hard samples and creates preference pairs for HS-DPO training.
Each pair has:
- chosen: correct reasoning + label
- rejected: incorrect reasoning + label
- weight: adaptive weight based on difficulty

Based on GuardReasoner paper methodology
"""

import json
import random
from pathlib import Path
from datetime import datetime
import argparse

# Paths
BASE_DIR = Path(__file__).parent
HARD_SAMPLES_PATH = BASE_DIR / "data/hard_samples_exp18.json"
OUTPUT_PATH = BASE_DIR / "data/guardreasoner_dpo_pairs.json"

# Configuration
SEED = 42
GAMMA = 0.5  # Normalization parameter for adaptive weighting


def normalize_weight(difficulty, gamma=GAMMA):
    """
    Normalize difficulty score to weight
    Paper uses: α = 1 + Norm(k_incorrect - k_correct, γ)

    difficulty ranges from -k to +k (e.g., -4 to +4 for k=4)
    We normalize to [0, 1] then scale
    """
    # For k=4: difficulty ranges from -4 to +4
    # Map to [0, 1]: (difficulty + k) / (2*k)
    k = 4  # We used k=4 for mining
    normalized = (difficulty + k) / (2 * k)

    # Apply gamma scaling
    weight = 1 + normalized * gamma
    return weight


def format_response(reasoning, label):
    """
    Format reasoning + label into response text
    Matches the training format from R-SFT
    """
    response = f"{reasoning}\n\nLabel: {label}"
    return response


def create_dpo_pairs(hard_samples):
    """
    Create DPO preference pairs from hard samples

    For each hard sample:
    1. Select one correct output as "chosen"
    2. Select one incorrect output as "rejected"
    3. Calculate adaptive weight based on difficulty
    """
    print(f"Creating DPO preference pairs from {len(hard_samples)} hard samples...")

    dpo_pairs = []
    stats = {
        'total_hard_samples': len(hard_samples),
        'pairs_created': 0,
        'skipped': 0,
    }

    for sample in hard_samples:
        # Get correct and incorrect outputs
        correct_outputs = [o for o in sample['outputs'] if o.get('correct', False)]
        incorrect_outputs = [o for o in sample['outputs'] if not o.get('correct', False)]

        # Validate we have both (should always be true for hard samples)
        if not correct_outputs or not incorrect_outputs:
            print(f"⚠️  Skipping sample - missing correct or incorrect outputs")
            stats['skipped'] += 1
            continue

        # Randomly select one from each group
        chosen = random.choice(correct_outputs)
        rejected = random.choice(incorrect_outputs)

        # Calculate adaptive weight
        difficulty = sample['difficulty']
        weight = normalize_weight(difficulty, GAMMA)

        # Create DPO pair
        dpo_pair = {
            'prompt': sample['prompt'],
            'chosen': format_response(chosen['reasoning'], chosen['label']),
            'rejected': format_response(rejected['reasoning'], rejected['label']),
            'weight': weight,
            'difficulty': difficulty,
            'k_correct': sample['k_correct'],
            'k_incorrect': sample['k_incorrect'],
            'ground_truth': sample['ground_truth'],
        }

        dpo_pairs.append(dpo_pair)
        stats['pairs_created'] += 1

    return dpo_pairs, stats


def main():
    parser = argparse.ArgumentParser(description="Create DPO preference dataset from hard samples")
    parser.add_argument('--gamma', type=float, default=GAMMA, help='Gamma parameter for weight normalization')
    args = parser.parse_args()

    print("="*80)
    print("CREATE DPO PREFERENCE DATASET")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Gamma: {args.gamma}")

    # Set seed
    random.seed(SEED)

    # Load hard samples
    if not HARD_SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Hard samples not found at {HARD_SAMPLES_PATH}")

    with open(HARD_SAMPLES_PATH) as f:
        data = json.load(f)

    hard_samples = data['hard_samples']
    print(f"\n✅ Loaded {len(hard_samples)} hard samples")
    print(f"   Mining metadata: {data['metadata']['mining_method']} with k={data['metadata']['k']}")

    # Create DPO pairs
    dpo_pairs, stats = create_dpo_pairs(hard_samples)

    # Print statistics
    print("\n" + "="*80)
    print("CREATION RESULTS")
    print("="*80)
    print(f"Total hard samples: {stats['total_hard_samples']}")
    print(f"DPO pairs created: {stats['pairs_created']}")
    print(f"Skipped: {stats['skipped']}")

    # Weight distribution
    if dpo_pairs:
        weights = [p['weight'] for p in dpo_pairs]
        print(f"\nWeight distribution:")
        print(f"  Min: {min(weights):.3f}")
        print(f"  Max: {max(weights):.3f}")
        print(f"  Avg: {sum(weights)/len(weights):.3f}")

        # Difficulty distribution
        difficulties = [p['difficulty'] for p in dpo_pairs]
        print(f"\nDifficulty distribution:")
        print(f"  Min: {min(difficulties)}")
        print(f"  Max: {max(difficulties)}")
        print(f"  Avg: {sum(difficulties)/len(difficulties):.2f}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'experiment_id': 18,
            'source': 'hard_samples_exp18.json',
            'gamma': args.gamma,
            'seed': SEED,
            'timestamp': datetime.now().isoformat(),
        },
        'stats': stats,
        'dpo_pairs': dpo_pairs,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Saved {len(dpo_pairs)} DPO pairs to {OUTPUT_PATH}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
