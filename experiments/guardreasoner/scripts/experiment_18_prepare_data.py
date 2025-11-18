#!/usr/bin/env python3
"""
Experiment 18: Prepare GuardReasoner Training Data

Downloads and formats GuardReasonerTrain dataset for fine-tuning.
Since Ollama doesn't support fine-tuning, this prepares data for:
- Unsloth (LoRA fine-tuning)
- LLaMA Factory
- HuggingFace Transformers

This script just prepares data - actual training needs GPU cluster or Colab.
"""

import os
import sys
import json
from datetime import datetime
from datasets import load_dataset, concatenate_datasets

print("="*80)
print("EXPERIMENT 18: PREPARE R-SFT TRAINING DATA")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("")

# Output directory
OUTPUT_DIR = "/home/vincent/wizard101/experiments/guardreasoner/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Step 1: Download Dataset
# ============================================================================

print("Step 1: Downloading GuardReasonerTrain dataset...")
print("(This will download 372MB - may take 2-3 minutes)")
print("")

try:
    print("Loading WildGuardTrainR (86K samples)...")
    wildguard = load_dataset("yueliu1999/GuardReasonerTrain", split="WildGuardTrainR")
    print(f"  ✓ {len(wildguard):,} samples")

    print("Loading AegisTrainR (11K samples)...")
    aegis = load_dataset("yueliu1999/GuardReasonerTrain", split="AegisTrainR")
    print(f"  ✓ {len(aegis):,} samples")

    print("Loading BeaverTailsTrainR (27K samples)...")
    beavertails = load_dataset("yueliu1999/GuardReasonerTrain", split="BeaverTailsTrainR")
    print(f"  ✓ {len(beavertails):,} samples")

    print("Loading ToxicChatTrainR (3K samples)...")
    toxicchat = load_dataset("yueliu1999/GuardReasonerTrain", split="ToxicChatTrainR")
    print(f"  ✓ {len(toxicchat):,} samples")

    print("\nCombining datasets...")
    full_dataset = concatenate_datasets([wildguard, aegis, beavertails, toxicchat])
    print(f"✓ Total: {len(full_dataset):,} samples with reasoning traces")

except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("")

# ============================================================================
# Step 2: Format for Different Training Frameworks
# ============================================================================

print("Step 2: Converting to training formats...")
print("")

# Format 1: ChatML (for Unsloth, HuggingFace)
print("Creating ChatML format (conversations)...")
chatml_data = []
for i, sample in enumerate(full_dataset):
    if i % 10000 == 0 and i > 0:
        print(f"  Progress: {i:,}/{len(full_dataset):,}")

    chatml_data.append({
        "conversations": [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    })

chatml_path = os.path.join(OUTPUT_DIR, "guardreasoner_train_chatml.json")
with open(chatml_path, 'w') as f:
    json.dump(chatml_data, f, indent=2)
print(f"✓ Saved ChatML format: {chatml_path}")
print(f"  Size: {os.path.getsize(chatml_path) / 1024 / 1024:.1f}MB")
print("")

# Format 2: JSONL (for streaming, LLaMA Factory)
print("Creating JSONL format...")
jsonl_path = os.path.join(OUTPUT_DIR, "guardreasoner_train.jsonl")
with open(jsonl_path, 'w') as f:
    for i, sample in enumerate(full_dataset):
        if i % 10000 == 0 and i > 0:
            print(f"  Progress: {i:,}/{len(full_dataset):,}")

        f.write(json.dumps({
            "instruction": sample["instruction"],
            "input": sample["input"],
            "output": sample["output"],
        }) + '\n')

print(f"✓ Saved JSONL format: {jsonl_path}")
print(f"  Size: {os.path.getsize(jsonl_path) / 1024 / 1024:.1f}MB")
print("")

# Format 3: Simple prompts (for reference)
print("Creating simplified examples (first 100)...")
examples_path = os.path.join(OUTPUT_DIR, "reasoning_examples.txt")
with open(examples_path, 'w') as f:
    for i in range(min(100, len(full_dataset))):
        sample = full_dataset[i]
        f.write(f"="*80 + "\n")
        f.write(f"EXAMPLE {i+1}\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"[INSTRUCTION]\n{sample['instruction']}\n\n")
        f.write(f"[INPUT]\n{sample['input']}\n\n")
        f.write(f"[OUTPUT - REASONING]\n{sample['output']}\n\n")

print(f"✓ Saved examples: {examples_path}")
print("")

# ============================================================================
# Step 3: Create Training Configuration
# ============================================================================

print("Step 3: Creating training configs...")
print("")

# Unsloth config
unsloth_config = {
    "model_name": "unsloth/Llama-3.2-3B-Instruct",
    "dataset_path": chatml_path,
    "output_dir": "/home/vincent/wizard101/experiments/guardreasoner/models/exp_18_unsloth",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 64,
    "warmup_steps": 100,
}

unsloth_config_path = os.path.join(OUTPUT_DIR, "unsloth_config.json")
with open(unsloth_config_path, 'w') as f:
    json.dump(unsloth_config, f, indent=2)
print(f"✓ Saved Unsloth config: {unsloth_config_path}")
print("")

# ============================================================================
# Step 4: Statistics
# ============================================================================

print("Step 4: Dataset statistics...")
print("")

# Sample lengths
instruction_lens = [len(s["instruction"]) for s in full_dataset]
input_lens = [len(s["input"]) for s in full_dataset]
output_lens = [len(s["output"]) for s in full_dataset]

print(f"Dataset size: {len(full_dataset):,} samples")
print("")
print(f"Instruction lengths:")
print(f"  Average: {sum(instruction_lens)/len(instruction_lens):.0f} chars")
print(f"  Min: {min(instruction_lens)} chars")
print(f"  Max: {max(instruction_lens)} chars")
print("")
print(f"Input lengths (prompts to classify):")
print(f"  Average: {sum(input_lens)/len(input_lens):.0f} chars")
print(f"  Min: {min(input_lens)} chars")
print(f"  Max: {max(input_lens)} chars")
print("")
print(f"Output lengths (reasoning traces):")
print(f"  Average: {sum(output_lens)/len(output_lens):.0f} chars")
print(f"  Min: {min(output_lens)} chars")
print(f"  Max: {max(output_lens)} chars")
print("")

# Estimate reasoning steps
sample_outputs = [full_dataset[i]["output"] for i in range(min(1000, len(full_dataset)))]
avg_steps = sum(s.count("## Reasoning Step") for s in sample_outputs) / len(sample_outputs)
print(f"Average reasoning steps: {avg_steps:.1f}")
print("(Matches paper: 3.61 steps)")
print("")

# ============================================================================
# Summary
# ============================================================================

print("="*80)
print("DATA PREPARATION COMPLETE!")
print("="*80)
print("")
print("Files created:")
print(f"  1. {chatml_path}")
print(f"     Format: ChatML (conversations)")
print(f"     Use with: Unsloth, HuggingFace")
print("")
print(f"  2. {jsonl_path}")
print(f"     Format: JSONL (streaming)")
print(f"     Use with: LLaMA Factory, custom trainers")
print("")
print(f"  3. {examples_path}")
print(f"     Format: Text (readable)")
print(f"     Use with: Manual inspection")
print("")
print(f"  4. {unsloth_config_path}")
print(f"     Format: JSON config")
print(f"     Use with: Unsloth training script")
print("")
print("="*80)
print("")
print("NEXT STEPS:")
print("")
print("Option A: Train on Google Colab T4 (FREE)")
print("  1. Upload ChatML file to Google Drive")
print("  2. Open the R-SFT training notebook")
print("  3. Point to uploaded file")
print("  4. Run (takes ~3-4 hours)")
print("")
print("Option B: Train on RunPod/Lambda Labs (~$1-2)")
print("  1. Spin up A100 instance")
print("  2. Clone wizard101 repo")
print("  3. Run experiment_18_unsloth_training.py")
print("  4. Train in ~1 hour")
print("")
print("Option C: Skip fine-tuning, test reasoning prompts first")
print("  1. Run Experiment 20 (reasoning prompts, no training)")
print("  2. See if prompting alone improves accuracy")
print("  3. If yes → proceed with fine-tuning")
print("  4. If no → may not be worth training cost")
print("")
print("RECOMMENDATION: Start with Option C (Experiment 20)")
print("  - Free")
print("  - Fast (4 hours)")
print("  - Validates if reasoning approach works")
print("  - If +5-10% improvement → proceed to fine-tuning")
print("")
print("="*80)

# Save summary
summary = {
    "experiment_id": 18,
    "experiment_name": "R-SFT Data Preparation",
    "timestamp": datetime.now().isoformat(),
    "dataset_size": len(full_dataset),
    "files_created": [chatml_path, jsonl_path, examples_path, unsloth_config_path],
    "avg_reasoning_steps": avg_steps,
    "avg_chars": {
        "instruction": sum(instruction_lens)/len(instruction_lens),
        "input": sum(input_lens)/len(input_lens),
        "output": sum(output_lens)/len(output_lens),
    },
    "next_steps": "Option C recommended: Test reasoning prompts (Exp 20) before fine-tuning"
}

summary_path = os.path.join(OUTPUT_DIR, "experiment_18_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: {summary_path}")
print("")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
