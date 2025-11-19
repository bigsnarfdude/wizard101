#!/usr/bin/env python3
"""
Experiment 20: 3-Task GuardReasoner Training

This script implements proper 3-task classification matching the paper:
- Task 1: Prompt harmfulness (harmful/unharmful)
- Task 2: Refusal detection (refusal/compliance)
- Task 3: Response harmfulness (harmful/unharmful)

Uses GuardReasonerTrain dataset from HuggingFace (128K samples).

Previous experiments:
- Exp 18: R-SFT with binary classification (11K samples) - 59% acc
- Exp 19: HS-DPO toy pipeline (100 samples)
- Exp 20: 3-task format with GuardReasonerTrain (THIS)

Expected results:
- Task 1 F1: 0.80-0.85 (paper: 0.875)
- Task 2 F1: 0.75-0.80 (paper: 0.810)
- Task 3 F1: 0.75-0.80 (paper: 0.825)
- Overall F1: 0.75-0.80 (paper: 0.84)

Usage:
    # Test with small subset (100 samples)
    python train_exp_20_3task.py --test

    # Full training
    python train_exp_20_3task.py

    # Dry run to check setup
    python train_exp_20_3task.py --dry-run
"""

import os
import json
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./models/exp_20_3task"
DATASET_NAME = "yueliu1999/GuardReasonerTrain"

# Training hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
GRAD_ACCUM = 64
EPOCHS = 1  # Test with 1 epoch
MAX_SEQ_LENGTH = 2048

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_3task_dataset(num_samples=None):
    """Load GuardReasonerTrain dataset with 3-task format."""
    print(f"Loading dataset: {DATASET_NAME}")

    # Load the dataset
    ds = load_dataset(DATASET_NAME)

    # Use WildGuardTrainR split (largest, 86K samples)
    train_data = ds["WildGuardTrainR"]

    if num_samples:
        train_data = train_data.select(range(min(num_samples, len(train_data))))
        print(f"Using {len(train_data)} samples for testing")
    else:
        print(f"Using full dataset: {len(train_data)} samples")

    return train_data


def format_3task_prompt(example):
    """Format example into 3-task training format."""

    # The dataset already has the proper format with:
    # - instruction: system prompt with 3-task template
    # - input: prompt + response
    # - output: reasoning + 3 labels

    # Combine into training format
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""

    return {"text": full_prompt}


def setup_model_and_tokenizer():
    """Set up model with LoRA for efficient training."""
    print(f"Loading model: {MODEL_ID}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    parser = argparse.ArgumentParser(description="3-Task GuardReasoner Training")
    parser.add_argument("--test", action="store_true", help="Run test with 100 samples")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--dry-run", action="store_true", help="Setup only, don't train")
    args = parser.parse_args()

    # Determine sample size
    num_samples = 100 if args.test else args.samples

    print("=" * 50)
    print("3-Task GuardReasoner Training Setup")
    print("=" * 50)

    # Load dataset
    train_data = load_3task_dataset(num_samples)

    # Show sample
    print("\nSample data point:")
    print("-" * 50)
    sample = train_data[0]
    print(f"Instruction: {sample['instruction'][:200]}...")
    print(f"Input: {sample['input'][:200]}...")
    print(f"Output: {sample['output'][:200]}...")
    print(f"Labels: prompt={sample.get('prompt_harm_label', 'N/A')}, "
          f"refusal={sample.get('response_refusal_label', 'N/A')}, "
          f"response={sample.get('response_harm_label', 'N/A')}")
    print("-" * 50)

    # Format dataset
    print("\nFormatting dataset for training...")
    train_data = train_data.map(format_3task_prompt)

    if args.dry_run:
        print("\n✅ Dry run complete. Dataset loaded and formatted.")
        print(f"Ready to train on {len(train_data)} samples")
        return

    # Setup model
    model, tokenizer = setup_model_and_tokenizer()

    # Tokenize
    print("\nTokenizing dataset...")
    tokenized_data = train_data.map(
        lambda x: tokenize_function(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=train_data.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    trainer.train()

    # Save
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
