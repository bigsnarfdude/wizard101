#!/usr/bin/env python3
"""
Convert GuardReasoner models to MLX format with 4-bit quantization
This creates a fast, Mac-optimized version of the model
"""

import argparse
from pathlib import Path
from mlx_lm import convert

def main():
    parser = argparse.ArgumentParser(description="Convert GuardReasoner to MLX format")
    parser.add_argument(
        "--model",
        type=str,
        default="yueliu1999/GuardReasoner-3B",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="q4",
        choices=["q4", "q8", "none"],
        help="Quantization: q4 (4-bit), q8 (8-bit), none (float16)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./mlx_models/guardreasoner-3b-4bit",
        help="Output directory"
    )

    args = parser.parse_args()

    print("="*70)
    print("CONVERTING GUARDREASONER TO MLX FORMAT")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantize}")
    print(f"Output: {args.output}")
    print()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Converting... (this may take 5-10 minutes)")

    # Convert with quantization
    quantize_arg = None if args.quantize == "none" else args.quantize

    convert(
        hf_path=args.model,
        mlx_path=str(output_path),
        quantize=quantize_arg
    )

    print()
    print("="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)
    print(f"Model saved to: {output_path}")
    print()
    print("To use this model:")
    print(f"  python eval_mlx_quantized.py --model-path {output_path}")
    print()

    # Show model size
    import os
    total_size = sum(
        os.path.getsize(output_path / f)
        for f in output_path.rglob('*')
        if f.is_file()
    )
    print(f"Model size: {total_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
