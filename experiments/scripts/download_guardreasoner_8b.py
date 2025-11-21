#!/usr/bin/env python3
"""
Download GuardReasoner-8B and cache it for bitsandbytes 4-bit inference.

This downloads the model to HuggingFace cache so it can be loaded
with BitsAndBytesConfig for 4-bit quantization on-the-fly.

Usage:
    python download_guardreasoner_8b.py

The model will be loaded in eval_layered_batch.py using:
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    model_id = "yueliu1999/GuardReasoner-8B"

    print(f"{'='*60}")
    print("Downloading GuardReasoner-8B")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print()

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("✓ Tokenizer downloaded")

    # Configure 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Download and load model in 4-bit
    print()
    print("Downloading and loading model in 4-bit (this downloads ~16GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ Model downloaded and loaded")

    # Test inference
    print()
    print("Testing inference...")
    inputs = tokenizer("Is this harmful?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test output: {response[:100]}...")
    print("✓ Inference test passed")

    # Show memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"\nGPU memory used: {memory_used:.1f} GB")

    print()
    print(f"{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print()
    print("Model cached in HuggingFace hub cache.")
    print("Will be loaded with 4-bit quantization in eval_layered_batch.py")


if __name__ == "__main__":
    main()
