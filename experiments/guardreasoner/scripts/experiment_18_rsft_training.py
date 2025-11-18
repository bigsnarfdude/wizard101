#!/usr/bin/env python3
"""
Experiment 18: R-SFT Training (Reasoning Supervised Fine-Tuning)

Based on GuardReasoner paper (Liu et al., 2025)
Trains gpt-oss:20b on GuardReasonerTrain dataset with reasoning traces

Runtime: ~8-12 hours on nigel
Output: Fine-tuned model that reasons before classifying
"""

import os
import sys
import json
import time
from datetime import datetime
from datasets import load_dataset, concatenate_datasets

print("="*80)
print("EXPERIMENT 18: R-SFT TRAINING (GuardReasoner Methodology)")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("")

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "experiment_id": 18,
    "experiment_name": "R-SFT Training on GuardReasonerTrain",
    "base_model": "gpt-oss:20b",
    "output_model": "gpt-oss-reasoning:20b",
    "dataset": "yueliu1999/GuardReasonerTrain",
    "max_samples": None,  # None = use all 127K samples
    "learning_rate": 5e-5,
    "epochs": 1,  # Start with 1, paper uses 3
    "batch_size": 8,
    "max_seq_length": 2048,
    "save_steps": 1000,
    "eval_steps": 500,
    "warmup_steps": 100,
    "output_dir": "/home/vincent/wizard101/experiments/guardreasoner/models/exp_18_checkpoint",
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print("")

# ============================================================================
# Step 1: Check Ollama Availability
# ============================================================================

print("-" * 80)
print("Step 1: Checking Ollama availability...")
print("-" * 80)

import subprocess
import requests

try:
    # Check if Ollama is running
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json().get("models", [])
        print(f"✓ Ollama is running")
        print(f"✓ Available models: {len(models)}")

        # Check if base model exists
        base_exists = any(m["name"] == CONFIG["base_model"] for m in models)
        if base_exists:
            print(f"✓ Base model '{CONFIG['base_model']}' found")
        else:
            print(f"⚠ Base model '{CONFIG['base_model']}' not found")
            print(f"  Available: {[m['name'] for m in models[:5]]}")
    else:
        print(f"✗ Ollama returned status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Cannot connect to Ollama: {e}")
    print("  Make sure Ollama is running: ollama serve")
    sys.exit(1)

print("")

# ============================================================================
# Step 2: Download GuardReasonerTrain Dataset
# ============================================================================

print("-" * 80)
print("Step 2: Loading GuardReasonerTrain dataset...")
print("-" * 80)
print("This will download 372MB (may take 2-3 minutes)")
print("")

try:
    # Load all splits
    print("Loading WildGuardTrainR...")
    wildguard = load_dataset("yueliu1999/GuardReasonerTrain", split="WildGuardTrainR")
    print(f"  ✓ Loaded {len(wildguard):,} samples")

    print("Loading AegisTrainR...")
    aegis = load_dataset("yueliu1999/GuardReasonerTrain", split="AegisTrainR")
    print(f"  ✓ Loaded {len(aegis):,} samples")

    print("Loading BeaverTailsTrainR...")
    beavertails = load_dataset("yueliu1999/GuardReasonerTrain", split="BeaverTailsTrainR")
    print(f"  ✓ Loaded {len(beavertails):,} samples")

    print("Loading ToxicChatTrainR...")
    toxicchat = load_dataset("yueliu1999/GuardReasonerTrain", split="ToxicChatTrainR")
    print(f"  ✓ Loaded {len(toxicchat):,} samples")

    # Combine all splits
    print("\nCombining all splits...")
    train_dataset = concatenate_datasets([wildguard, aegis, beavertails, toxicchat])
    print(f"✓ Total training samples: {len(train_dataset):,}")

    # Optionally limit dataset size for testing
    if CONFIG["max_samples"] and CONFIG["max_samples"] < len(train_dataset):
        print(f"⚠ Limiting to {CONFIG['max_samples']:,} samples for testing")
        train_dataset = train_dataset.select(range(CONFIG["max_samples"]))

    print(f"\nDataset structure:")
    print(f"  Columns: {train_dataset.column_names}")
    print(f"  Sample 0 preview:")
    sample = train_dataset[0]
    print(f"    Instruction length: {len(sample['instruction'])} chars")
    print(f"    Input length: {len(sample['input'])} chars")
    print(f"    Output length: {len(sample['output'])} chars (reasoning!)")

except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    sys.exit(1)

print("")

# ============================================================================
# Step 3: Format Dataset for Ollama Fine-Tuning
# ============================================================================

print("-" * 80)
print("Step 3: Converting dataset to Ollama fine-tuning format...")
print("-" * 80)

def format_sample_for_ollama(sample):
    """
    Convert GuardReasonerTrain format to Ollama fine-tuning format

    Input format (GuardReasonerTrain):
      - instruction: Task description
      - input: User prompt
      - output: Reasoning + classification

    Output format (Ollama):
      - prompt: instruction + input
      - response: output (reasoning trace)
    """
    prompt = f"""System: {sample['instruction']}

User: {sample['input']}"""

    response = sample['output']

    return {
        "prompt": prompt,
        "response": response,
    }

print("Formatting samples...")
formatted_data = []
for i, sample in enumerate(train_dataset):
    if i % 10000 == 0:
        print(f"  Progress: {i:,}/{len(train_dataset):,}")
    formatted_data.append(format_sample_for_ollama(sample))

print(f"✓ Formatted {len(formatted_data):,} samples")
print("")

# Save formatted dataset
output_jsonl = os.path.join(CONFIG["output_dir"], "training_data.jsonl")
os.makedirs(CONFIG["output_dir"], exist_ok=True)

print(f"Saving to {output_jsonl}...")
with open(output_jsonl, 'w') as f:
    for item in formatted_data:
        f.write(json.dumps(item) + '\n')

file_size_mb = os.path.getsize(output_jsonl) / 1024 / 1024
print(f"✓ Saved {file_size_mb:.1f}MB")
print("")

# ============================================================================
# Step 4: Create Modelfile for Fine-Tuning
# ============================================================================

print("-" * 80)
print("Step 4: Creating Ollama Modelfile...")
print("-" * 80)

modelfile_content = f"""# GuardReasoner R-SFT Fine-Tuning
# Experiment 18: Train gpt-oss:20b on reasoning traces

FROM {CONFIG["base_model"]}

# System prompt for reasoning-based safety classification
SYSTEM \"\"\"You are a safety classifier that reasons step-by-step before making decisions.

When given a prompt to classify, you must:
1. Analyze the user's intent
2. Assess potential harms
3. Apply relevant safety policies
4. Provide your reasoning
5. Give final classification

Always think carefully and explain your reasoning.\"\"\"

# Training parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""

modelfile_path = os.path.join(CONFIG["output_dir"], "Modelfile")
with open(modelfile_path, 'w') as f:
    f.write(modelfile_content)

print(f"✓ Modelfile created at {modelfile_path}")
print("")
print("Modelfile content:")
print(modelfile_content)
print("")

# ============================================================================
# Step 5: Run Ollama Fine-Tuning
# ============================================================================

print("=" * 80)
print("Step 5: STARTING R-SFT TRAINING")
print("=" * 80)
print("")
print("⚠ NOTE: Ollama doesn't have built-in fine-tuning yet!")
print("")
print("Two options:")
print("")
print("Option A: Use LLaMA.cpp + LoRA fine-tuning")
print("  1. Export model to GGUF")
print("  2. Use llama.cpp finetune")
print("  3. Merge LoRA back")
print("  4. Import to Ollama")
print("")
print("Option B: Use Unsloth (recommended)")
print("  1. Load model in HuggingFace format")
print("  2. Fine-tune with Unsloth (4-bit LoRA)")
print("  3. Save merged model")
print("  4. Import to Ollama")
print("")
print("=" * 80)
print("")

# ============================================================================
# ALTERNATIVE: Direct Training Loop with Ollama API
# ============================================================================

print("Implementing ALTERNATIVE: Training via Ollama generate API")
print("(Simulated R-SFT - not true gradient descent)")
print("")

print("⚠ This is a PROOF-OF-CONCEPT:")
print("  - Uses few-shot learning instead of gradient updates")
print("  - Creates reasoning examples for context window")
print("  - Not true fine-tuning, but demonstrates approach")
print("")

# Create few-shot examples from training data
num_examples = min(10, len(formatted_data))
few_shot_examples = formatted_data[:num_examples]

print(f"Creating {num_examples} few-shot reasoning examples...")
few_shot_context = "\n\n".join([
    f"Example {i+1}:\nPrompt: {ex['prompt']}\nResponse: {ex['response']}"
    for i, ex in enumerate(few_shot_examples)
])

print(f"✓ Few-shot context created ({len(few_shot_context)} chars)")
print("")

# Save context for later use
context_path = os.path.join(CONFIG["output_dir"], "reasoning_examples.txt")
with open(context_path, 'w') as f:
    f.write(few_shot_context)
print(f"✓ Saved reasoning examples to {context_path}")
print("")

# ============================================================================
# Summary and Next Steps
# ============================================================================

print("=" * 80)
print("EXPERIMENT 18 PREPARATION COMPLETE")
print("=" * 80)
print("")
print("Dataset prepared:")
print(f"  ✓ {len(train_dataset):,} samples with reasoning traces")
print(f"  ✓ Formatted data: {output_jsonl}")
print(f"  ✓ File size: {file_size_mb:.1f}MB")
print("")
print("Next steps for TRUE R-SFT training:")
print("")
print("1. Use Unsloth (recommended):")
print("   cd /home/vincent/wizard101/experiments/guardreasoner/scripts")
print("   python3 experiment_18_unsloth_training.py")
print("")
print("2. Or use LLaMA Factory:")
print("   llamafactory-cli train exp_18_config.yaml")
print("")
print("3. Or convert to HuggingFace and train manually:")
print("   python3 experiment_18_hf_training.py")
print("")
print("=" * 80)
print("")

# Save experiment config
config_path = os.path.join(CONFIG["output_dir"], "experiment_18_config.json")
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=2)
print(f"✓ Config saved to {config_path}")
print("")

end_time = datetime.now()
print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
