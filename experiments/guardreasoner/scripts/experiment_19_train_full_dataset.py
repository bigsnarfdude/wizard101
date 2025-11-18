#!/usr/bin/env python3
"""
Experiment 19: GuardReasoner R-SFT Training with Full Dataset (127K samples)

This uses the exact GuardReasonerTrain dataset from the paper.
Expected: 75-80% accuracy after 3 epochs (vs current 59% with 11K samples)

Dataset: 127,544 samples (WildGuard + Aegis + BeaverTails + ToxicChat)
Model: LLaMA-3.2-3B-Instruct with 4-bit LoRA
Training: 3 epochs, ~270 hours (~11 days)
"""

import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_ID = "yueliu1999/GuardReasonerTrain"
OUTPUT_DIR = "guardreasoner-llama3.2-3b-lora-full"
EXPERIMENT_NAME = "exp_19_full_dataset"

print("="*60)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Model: {MODEL_ID}")
print(f"Dataset: {DATASET_ID}")
print("="*60)

# 1. Load Tokenizer
print(f"\n[1/8] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Important for training

# 2. Load Dataset
print(f"\n[2/8] Loading dataset...")
dataset = load_dataset(DATASET_ID)

print(f"Available splits: {list(dataset.keys())}")
for split_name, split_data in dataset.items():
    print(f"  {split_name}: {len(split_data):,} samples")

# Combine all splits
print("\nCombining all splits...")
train_dataset = concatenate_datasets([
    dataset['WildGuardTrainR'],
    dataset['AegisTrainR'],
    dataset['BeaverTailsTrainR'],
    dataset['ToxicChatTrainR']
])
print(f"Total training samples: {len(train_dataset):,}")

# 3. Format Dataset for Llama 3
print(f"\n[3/8] Formatting dataset...")

def format_chat_template(row):
    """Format GuardReasoner 3-task data for Llama 3 chat template."""
    messages = [
        {"role": "system", "content": row["instruction"]},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": row["output"]}
    ]

    # Apply Llama 3 chat template
    row["text"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return row

train_dataset = train_dataset.map(
    format_chat_template,
    desc="Formatting samples"
)

# Show example
print("\n📝 Example formatted sample:")
print("="*60)
print(train_dataset[0]["text"][:500] + "...")
print("="*60)

# 4. Check sequence lengths (important for reasoning traces!)
print(f"\n[4/8] Checking sequence lengths...")
lengths = []
sample_size = min(1000, len(train_dataset))
for i in range(sample_size):
    tokens = tokenizer(train_dataset[i]["text"], truncation=False)
    lengths.append(len(tokens["input_ids"]))

print(f"Sequence length statistics ({sample_size} samples):")
print(f"  Mean: {np.mean(lengths):.0f}")
print(f"  Median: {np.median(lengths):.0f}")
print(f"  75th percentile: {np.percentile(lengths, 75):.0f}")
print(f"  95th percentile: {np.percentile(lengths, 95):.0f}")
print(f"  99th percentile: {np.percentile(lengths, 99):.0f}")
print(f"  Max: {np.max(lengths):.0f}")

# Recommend max_seq_length
recommended_length = int(np.percentile(lengths, 95))
max_seq_length = min(2048, max(recommended_length, 1024))
print(f"\n✅ Using max_seq_length: {max_seq_length}")

# Count truncations
truncated = sum(1 for l in lengths if l > max_seq_length)
print(f"⚠️  Samples that will be truncated: {truncated}/{sample_size} ({truncated/sample_size*100:.1f}%)")

# 5. Split for evaluation
print(f"\n[5/8] Creating train/eval split...")
train_eval = train_dataset.train_test_split(test_size=0.01, seed=42)
train_data = train_eval["train"]
eval_data = train_eval["test"]
print(f"  Train: {len(train_data):,} samples")
print(f"  Eval: {len(eval_data):,} samples")

# 6. Load Model in 4-bit (QLoRA)
print(f"\n[6/8] Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    # attn_implementation="flash_attention_2",  # Uncomment if flash attention installed
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Required for gradient checkpointing

# 7. LoRA Configuration
print(f"\n[7/8] Configuring LoRA...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)

# 8. Training Arguments
print(f"\n[8/8] Setting up training...")

# Calculate steps
steps_per_epoch = len(train_data) // (2 * 64)  # batch_size * grad_accum
total_steps = steps_per_epoch * 3

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Training schedule
    num_train_epochs=3,

    # Batch size (effective batch = 128, matching Exp 18)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=64,

    # Learning rate (matching Exp 18)
    learning_rate=5e-5,
    weight_decay=0.01,

    # Precision (bf16 for 40-series GPUs)
    fp16=False,
    bf16=True,

    # Logging
    logging_steps=10,
    logging_dir=f"{OUTPUT_DIR}/logs",

    # Saving
    save_strategy="epoch",
    save_total_limit=3,

    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",

    # Optimizer
    optim="paged_adamw_32bit",

    # Dataset config
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,

    # Misc
    warmup_steps=100,
    gradient_checkpointing=True,
    report_to="none",  # Change to "wandb" if you use W&B

    # Reproducibility
    seed=42,
)

print(f"\n📊 Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Total samples: {len(train_data):,}")
print(f"  Steps per epoch: ~{steps_per_epoch:,}")
print(f"  Total steps: ~{total_steps:,}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Max seq length: {training_args.max_seq_length}")
print(f"  Estimated time: ~270 hours (~11 days)")

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

# Start Training
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
print(f"Start time: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*60 + "\n")

trainer.train()

# Save the Adapter
print(f"\n💾 Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training info
import json
training_info = {
    "experiment": EXPERIMENT_NAME,
    "model": MODEL_ID,
    "dataset": DATASET_ID,
    "total_samples": len(train_data),
    "epochs": training_args.num_train_epochs,
    "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "learning_rate": training_args.learning_rate,
    "max_seq_length": max_seq_length,
    "lora_r": peft_config.r,
    "lora_alpha": peft_config.lora_alpha,
}

with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("\n✅ Training complete!")
print(f"📁 Model saved to: {OUTPUT_DIR}")
print(f"📊 Training info saved to: {OUTPUT_DIR}/training_info.json")
print("\n🎯 Next steps:")
print("  1. Run evaluation: python3 evaluate_exp_19.py")
print("  2. Compare to Exp 18 results (59% baseline)")
print("  3. Expected: 75-80% accuracy")
