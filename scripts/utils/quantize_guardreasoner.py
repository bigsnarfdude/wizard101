#!/usr/bin/env python3
"""
Quantize GuardReasoner-8B to 4-bit and save as safetensors.

This creates a proper 4-bit quantized model that can be loaded directly
without needing bitsandbytes at runtime (the quantization is baked in).

Run on a machine with enough RAM (16GB+) or GPU.
Output can then be uploaded to HuggingFace for easy loading.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

def main():
    model_id = "yueliu1999/GuardReasoner-8B"
    output_dir = "./models/guardreasoner-8b-4bit"

    print(f"Quantizing {model_id} to 4-bit...")
    print(f"Output directory: {output_dir}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model with 4-bit quantization
    print("Loading model with 4-bit quantization (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Save the quantized model
    print(f"Saving quantized model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"\nDone! Quantized model saved to {output_dir}")
    print("\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload vincentoh/guardreasoner-8b-4bit {output_dir}")

    # Show size comparison
    import subprocess
    result = subprocess.run(['du', '-sh', output_dir], capture_output=True, text=True)
    print(f"\nModel size: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
