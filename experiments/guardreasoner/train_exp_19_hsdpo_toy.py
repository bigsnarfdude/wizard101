#!/usr/bin/env python3
"""
HS-DPO Training (Toy Experiment)

Train HS-DPO on small preference dataset (63 pairs from 100-sample test)
to validate the pipeline before scaling to full dataset.

Based on GuardReasoner paper methodology
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from datetime import datetime
import argparse

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/exp_18_rsft_lora"
DPO_DATA_PATH = BASE_DIR / "data/guardreasoner_dpo_pairs.json"
OUTPUT_PATH = BASE_DIR / "models/exp_19_hsdpo_toy_lora"

# HS-DPO Configuration (from paper)
LEARNING_RATE = 5e-6  # 10x lower than R-SFT
BATCH_SIZE = 2  # Small for toy experiment
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 8
NUM_EPOCHS = 2  # Paper uses 2 epochs
BETA = 0.01  # KL constraint
MAX_LENGTH = 2048
SEED = 42


def load_dpo_data():
    """Load DPO preference pairs"""
    if not DPO_DATA_PATH.exists():
        raise FileNotFoundError(f"DPO data not found at {DPO_DATA_PATH}")

    with open(DPO_DATA_PATH) as f:
        data = json.load(f)

    dpo_pairs = data['dpo_pairs']
    print(f"✅ Loaded {len(dpo_pairs)} DPO preference pairs")

    # Convert to HuggingFace Dataset format
    dataset_dict = {
        'prompt': [p['prompt'] for p in dpo_pairs],
        'chosen': [p['chosen'] for p in dpo_pairs],
        'rejected': [p['rejected'] for p in dpo_pairs],
    }

    # Note: TRL DPOTrainer doesn't support sample weights in the dataset
    # We'll need to implement custom weighting if needed
    # For now, we'll train without explicit weights

    dataset = Dataset.from_dict(dataset_dict)
    return dataset, data['metadata']


def load_model_and_tokenizer():
    """Load R-SFT model and tokenizer"""
    print(f"Loading R-SFT model from {MODEL_PATH}...")

    base_model_name = "unsloth/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model in 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load R-SFT LoRA adapter
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    # Merge LoRA weights into base model for DPO training
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Prepare for new LoRA training (HS-DPO)
    model = prepare_model_for_kbit_training(model)

    # Add new LoRA adapter for HS-DPO
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    print(f"✅ Model ready with new LoRA adapter")

    return model, tokenizer


def format_prompt(prompt):
    """Format prompt for training (ChatML format)"""
    formatted = f"""<|im_start|>user
Analyze this content for safety:

"{prompt}"

Is this content safe or harmful? Provide step-by-step reasoning.<|im_end|>
<|im_start|>assistant
Let me analyze this step by step:

"""
    return formatted


def main():
    parser = argparse.ArgumentParser(description="HS-DPO toy training")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--beta', type=float, default=BETA, help='DPO beta (KL constraint)')
    args = parser.parse_args()

    print("="*80)
    print("HS-DPO TOY TRAINING (Experiment 19)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Beta: {args.beta}")

    # Set seed
    torch.manual_seed(SEED)

    # Load data
    dataset, metadata = load_dpo_data()
    print(f"\nDPO dataset metadata:")
    print(f"  Source: {metadata['source']}")
    print(f"  Gamma: {metadata['gamma']}")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # DPO Training configuration
    training_args = DPOConfig(
        output_dir=str(OUTPUT_PATH),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=MAX_LENGTH,
        max_prompt_length=512,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
    )

    # Initialize DPO trainer
    print("\nInitializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # TRL uses processing_class instead of tokenizer
    )

    # Train
    print("\n" + "="*80)
    print("STARTING HS-DPO TRAINING")
    print("="*80)
    trainer.train()

    # Save final model
    print(f"\nSaving model to {OUTPUT_PATH}...")
    trainer.save_model(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved to: {OUTPUT_PATH}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
