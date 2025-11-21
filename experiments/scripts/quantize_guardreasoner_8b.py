#!/usr/bin/env python3
"""
Quantize GuardReasoner-8B to GPTQ INT4 for CUDA inference.

This creates a fast inference model using GPTQ with Marlin kernel support.
Requires ~32GB system RAM during quantization.
Result: ~4-5GB VRAM for inference.

Usage:
    python quantize_guardreasoner_8b.py

Output:
    ./models/GuardReasoner-8B-GPTQ/
"""

import os
import random
import numpy as np
import torch
from pathlib import Path

def main():
    # Check for auto-gptq
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("Installing auto-gptq...")
        os.system("pip install auto-gptq --no-build-isolation")
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Model configuration
    model_id = "yueliu1999/GuardReasoner-8B"
    output_dir = Path(__file__).parent.parent / "models" / "GuardReasoner-8B-GPTQ"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("GPTQ Quantization: GuardReasoner-8B → INT4")
    print(f"{'='*60}")
    print(f"Source: {model_id}")
    print(f"Output: {output_dir}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Prepare calibration dataset
    print("Preparing calibration data (128 samples from wikitext)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    # Create calibration samples
    nsamples = 128
    seqlen = 2048
    calibration_samples = []

    for _ in range(nsamples):
        i = random.randint(0, encodings.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        input_ids = encodings.input_ids[:, i:j]
        attention_mask = torch.ones_like(input_ids)
        calibration_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

    print(f"Created {len(calibration_samples)} calibration samples")
    print()

    # Quantization configuration (Marlin kernel compatible)
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,      # Required for Marlin kernel
        desc_act=False,      # Required for Marlin kernel
        model_file_base_name="model",
        damp_percent=0.1,
    )

    # Load model for quantization
    print("Loading model for quantization (requires ~32GB RAM)...")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id,
        quantize_config,
        trust_remote_code=True
    )

    # Quantize
    print()
    print("Quantizing model (this takes 10-30 minutes)...")
    model.quantize(calibration_samples)

    # Save
    print()
    print(f"Saving quantized model to {output_dir}...")
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print()
    print(f"{'='*60}")
    print("QUANTIZATION COMPLETE")
    print(f"{'='*60}")
    print()
    print(f"Model saved to: {output_dir}")
    print()
    print("Usage:")
    print("  from auto_gptq import AutoGPTQForCausalLM")
    print("  model = AutoGPTQForCausalLM.from_quantized(")
    print(f"      '{output_dir}',")
    print("      device='cuda:0',")
    print("      use_marlin=True,  # Fast inference")
    print("      trust_remote_code=True")
    print("  )")
    print()

    # Test loading
    print("Testing load with Marlin kernel...")
    try:
        test_model = AutoGPTQForCausalLM.from_quantized(
            str(output_dir),
            device="cuda:0",
            use_marlin=True,
            trust_remote_code=True
        )
        print("✓ Model loads successfully with Marlin kernel")

        # Quick inference test
        inputs = tokenizer("Is this harmful?", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = test_model.generate(**inputs, max_new_tokens=50)
        print("✓ Inference test passed")

        del test_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠ Test failed: {e}")
        print("Model may still be usable without Marlin kernel")


if __name__ == "__main__":
    main()
