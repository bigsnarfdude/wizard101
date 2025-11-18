#!/usr/bin/env python3
"""
Hard Sample Mining for HS-DPO

Generate k outputs per sample and identify ambiguous samples
(samples with both correct AND incorrect outputs)

Based on GuardReasoner paper methodology
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
from tqdm import tqdm
from datetime import datetime
import argparse

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/exp_18_rsft_lora"
TRAIN_DATA_PATH = BASE_DIR / "data/guardreasoner_train_chatml.json"  # ChatML training data
OUTPUT_PATH = BASE_DIR / "data/hard_samples_exp18.json"

# Configuration
K_OUTPUTS = 4  # Generate 4 outputs per sample (paper uses 4-8)
TEMPERATURE = 0.8  # High temperature for diversity
MAX_NEW_TOKENS = 512
SEED = 42


def load_model():
    """Load the 1-epoch R-SFT model"""
    print(f"Loading model from {MODEL_PATH}...")

    base_model_name = "unsloth/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    return model, tokenizer


def load_training_data():
    """Load training data for hard sample mining (ChatML format)"""
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_PATH}")

    with open(TRAIN_DATA_PATH) as f:
        data = json.load(f)

    # Extract samples in simplified format from ChatML
    simplified_data = []
    for item in data:
        conversations = item.get('conversations', [])

        # Find user message and assistant response
        user_content = None
        assistant_content = None

        for conv in conversations:
            if conv.get('role') == 'user':
                user_content = conv.get('content', '')
            elif conv.get('role') == 'assistant':
                assistant_content = conv.get('content', '')

        if user_content and assistant_content:
            # Extract ground truth label from assistant response
            label = 'unknown'
            assistant_lower = assistant_content.lower()

            # Check for explicit labels in reasoning
            if 'harmful' in assistant_lower and 'not harmful' not in assistant_lower and 'unharmful' not in assistant_lower:
                label = 'harmful'
            elif 'safe' in assistant_lower or 'unharmful' in assistant_lower or 'harmless' in assistant_lower:
                label = 'safe'

            simplified_data.append({
                'text': user_content,
                'label': label,
                'original_response': assistant_content
            })

    print(f"✅ Loaded {len(simplified_data)} training samples from ChatML format")
    return simplified_data


def create_prompt(sample):
    """Create input prompt for model"""
    content = sample.get('text', sample.get('prompt', ''))

    prompt = f"""<|im_start|>user
Analyze this content for safety:

"{content}"

Is this content safe or harmful? Provide step-by-step reasoning.<|im_end|>
<|im_start|>assistant
Let me analyze this step by step:

"""
    return prompt


def parse_response(response):
    """Parse model response to extract reasoning and label"""
    lines = response.strip().split('\n')

    # Try to find explicit label
    label = None
    for line in lines:
        line_lower = line.lower()
        if 'label:' in line_lower:
            if 'harmful' in line_lower:
                label = 'harmful'
            elif 'safe' in line_lower or 'harmless' in line_lower:
                label = 'safe'
            break

    # If no explicit label, check last few lines
    if label is None:
        last_text = ' '.join(lines[-3:]).lower()
        if 'harmful' in last_text and 'not harmful' not in last_text and 'unharmful' not in last_text:
            label = 'harmful'
        elif 'safe' in last_text or 'benign' in last_text or 'harmless' in last_text:
            label = 'safe'
        else:
            label = 'unknown'

    reasoning = response.strip()
    return reasoning, label


def generate_output(model, tokenizer, sample, seed_offset=0):
    """Generate a single output for a sample"""
    prompt = create_prompt(sample)

    # Set seed for reproducibility
    torch.manual_seed(SEED + seed_offset)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):]

    reasoning, predicted_label = parse_response(response)

    return {
        'reasoning': reasoning,
        'label': predicted_label,
        'full_response': response
    }


def mine_hard_samples(model, tokenizer, train_data, k=K_OUTPUTS):
    """Mine hard samples by generating k outputs per sample"""
    print(f"\n{'='*80}")
    print(f"HARD SAMPLE MINING: Generating {k} outputs per sample")
    print(f"{'='*80}")

    hard_samples = []
    stats = {
        'total': 0,
        'hard': 0,
        'easy_correct': 0,
        'easy_incorrect': 0,
    }

    for sample in tqdm(train_data, desc="Mining hard samples"):
        # Normalize ground truth label
        ground_truth = sample.get('label', sample.get('ground_truth', 'unknown'))
        if ground_truth == 'harmless':
            ground_truth = 'safe'

        if ground_truth not in ['safe', 'harmful']:
            continue  # Skip samples without valid labels

        # Generate k outputs
        outputs = []
        for i in range(k):
            try:
                output = generate_output(model, tokenizer, sample, seed_offset=i*1000)

                # Check correctness
                correct = (output['label'] == ground_truth)
                output['correct'] = correct
                outputs.append(output)
            except Exception as e:
                print(f"\n⚠️  Error generating output: {e}")
                continue

        # Count correct and incorrect
        correct_count = sum(1 for o in outputs if o.get('correct', False))
        incorrect_count = len(outputs) - correct_count

        stats['total'] += 1

        # Classify sample difficulty
        if correct_count > 0 and incorrect_count > 0:
            # HARD SAMPLE: Has both correct and incorrect outputs
            stats['hard'] += 1
            difficulty = incorrect_count - correct_count

            hard_samples.append({
                'prompt': sample.get('text', sample.get('prompt', '')),
                'ground_truth': ground_truth,
                'outputs': outputs,
                'k_correct': correct_count,
                'k_incorrect': incorrect_count,
                'difficulty': difficulty,  # Higher = harder
                'difficulty_score': incorrect_count / len(outputs),  # 0.0 to 1.0
            })
        elif correct_count == len(outputs):
            # Easy sample: all correct
            stats['easy_correct'] += 1
        elif incorrect_count == len(outputs):
            # Easy sample: all incorrect (model consistently wrong)
            stats['easy_incorrect'] += 1

    return hard_samples, stats


def main():
    parser = argparse.ArgumentParser(description="Mine hard samples for HS-DPO")
    parser.add_argument('--k', type=int, default=K_OUTPUTS, help='Number of outputs per sample')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit number of samples to process')
    args = parser.parse_args()

    print("="*80)
    print("HARD SAMPLE MINING FOR HS-DPO")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"K outputs per sample: {args.k}")

    # Set seed
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    model, tokenizer = load_model()

    # Load training data
    train_data = load_training_data()

    if args.max_samples:
        train_data = random.sample(train_data, min(args.max_samples, len(train_data)))
        print(f"Limited to {len(train_data)} samples")

    # Mine hard samples
    hard_samples, stats = mine_hard_samples(model, tokenizer, train_data, k=args.k)

    # Sort by difficulty (hardest first)
    hard_samples.sort(key=lambda x: x['difficulty'], reverse=True)

    # Print statistics
    print("\n" + "="*80)
    print("MINING RESULTS")
    print("="*80)
    print(f"Total samples processed: {stats['total']}")
    print(f"Hard samples found: {stats['hard']} ({stats['hard']/stats['total']*100:.1f}%)")
    print(f"Easy (all correct): {stats['easy_correct']} ({stats['easy_correct']/stats['total']*100:.1f}%)")
    print(f"Easy (all incorrect): {stats['easy_incorrect']} ({stats['easy_incorrect']/stats['total']*100:.1f}%)")

    # Show difficulty distribution
    if hard_samples:
        difficulties = [s['difficulty'] for s in hard_samples]
        print(f"\nDifficulty distribution:")
        print(f"  Min: {min(difficulties)}")
        print(f"  Max: {max(difficulties)}")
        print(f"  Avg: {sum(difficulties)/len(difficulties):.2f}")

        # Show hardest samples
        print(f"\nTop 5 hardest samples:")
        for i, sample in enumerate(hard_samples[:5], 1):
            print(f"{i}. Difficulty={sample['difficulty']}, "
                  f"Correct={sample['k_correct']}, "
                  f"Incorrect={sample['k_incorrect']}")
            print(f"   Prompt: {sample['prompt'][:100]}...")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'experiment_id': 18,
            'mining_method': 'best-of-n',
            'k': args.k,
            'temperature': TEMPERATURE,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(MODEL_PATH),
        },
        'stats': stats,
        'hard_samples': hard_samples,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Saved {len(hard_samples)} hard samples to {OUTPUT_PATH}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
