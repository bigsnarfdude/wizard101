#!/usr/bin/env python3
"""
Properly download GuardReasoner-8B with all sharded files
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("="*70)
print("DOWNLOADING GUARDREASONER-8B")
print("="*70)
print("Model: yueliu1999/GuardReasoner-8B")
print("Size: ~16GB (4 sharded files)")
print()

model_id = "yueliu1999/GuardReasoner-8B"

print("Step 1: Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("✅ Tokenizer downloaded")
print()

print("Step 2: Downloading model weights (this will take 5-10 minutes)...")
print("Progress will be shown below:")
print()

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print()
print("="*70)
print("✅ MODEL DOWNLOADED SUCCESSFULLY!")
print("="*70)
print(f"Cached at: ~/.cache/huggingface/hub/")
print()
print("You can now run:")
print("  python eval_official_guardreasoner.py --model 8b --samples 50")
print("="*70)
