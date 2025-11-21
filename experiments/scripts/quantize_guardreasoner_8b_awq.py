#!/usr/bin/env python3
"""
Quantize GuardReasoner-8B to AWQ INT4 for CUDA inference.

AWQ (Activation-aware Weight Quantization) provides fast inference
with minimal quality loss.

Usage:
    python quantize_guardreasoner_8b_awq.py

Output:
    ./models/GuardReasoner-8B-AWQ/
"""

import os
import torch
from pathlib import Path

def main():
    # Check for autoawq
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("Installing autoawq...")
        os.system("pip install autoawq")
        from awq import AutoAWQForCausalLM

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Model configuration
    model_id = "yueliu1999/GuardReasoner-8B"
    output_dir = Path(__file__).parent.parent / "models" / "GuardReasoner-8B-AWQ"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("AWQ Quantization: GuardReasoner-8B → INT4")
    print(f"{'='*60}")
    print(f"Source: {model_id}")
    print(f"Output: {output_dir}")
    print()

    # Quantization configuration
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"  # Use GEMM for batched inference
    }

    # Load model
    print("Loading model (requires ~32GB RAM)...")
    model = AutoAWQForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load calibration data
    print("Loading calibration data...")
    def get_calib_data():
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
        return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20][:256]

    calib_data = get_calib_data()
    print(f"Using {len(calib_data)} calibration samples")
    print()

    # Quantize
    print("Quantizing model (this takes 10-30 minutes)...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

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
    print("  from awq import AutoAWQForCausalLM")
    print("  model = AutoAWQForCausalLM.from_quantized(")
    print(f"      '{output_dir}',")
    print("      fuse_layers=True,")
    print("      trust_remote_code=True")
    print("  )")
    print()

    # Test loading
    print("Testing load...")
    try:
        del model
        torch.cuda.empty_cache()

        test_model = AutoAWQForCausalLM.from_quantized(
            str(output_dir),
            fuse_layers=True,
            trust_remote_code=True
        )
        print("✓ Model loads successfully")

        # Quick inference test
        inputs = tokenizer("Is this harmful?", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = test_model.generate(**inputs, max_new_tokens=50)
        print("✓ Inference test passed")

        del test_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠ Test failed: {e}")


if __name__ == "__main__":
    main()
