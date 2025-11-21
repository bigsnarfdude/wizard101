#!/usr/bin/env python3
"""
Convert GuardReasoner-8B to Ollama format for faster inference.

GuardReasoner-8B is based on Llama-3.1-8B, so it can be converted to GGUF
and imported into Ollama.

Two methods:
1. Direct safetensors import (Ollama 0.1.14+)
2. GGUF conversion via llama.cpp

Usage:
    # Method 1: Direct import (recommended)
    python convert_guardreasoner_to_ollama.py --method direct

    # Method 2: GGUF conversion
    python convert_guardreasoner_to_ollama.py --method gguf

    # Just create Modelfile
    python convert_guardreasoner_to_ollama.py --modelfile-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODEL_NAME = "yueliu1999/GuardReasoner-8B"
OLLAMA_MODEL_NAME = "guardreasoner-8b"

# GuardReasoner official system prompt
SYSTEM_PROMPT = """You are a safety classifier. Given a user message, classify it as either "harmful" or "safe".

Analyze the request carefully:
1. Identify what is being requested
2. Consider if it could enable harm
3. Evaluate legitimate vs malicious use cases

Respond with only: "harmful" or "safe"
"""

def create_modelfile(model_path: str = None, quantization: str = "Q4_K_M") -> str:
    """Create Ollama Modelfile for GuardReasoner-8B."""

    if model_path:
        # Direct safetensors import
        content = f"""FROM {model_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>\"\"\"

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0
PARAMETER num_predict 50

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"
"""
    else:
        # GGUF file
        content = f"""FROM ./guardreasoner-8b-{quantization.lower()}.gguf

TEMPLATE \"\"\"{{{{ if .System }}}}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>\"\"\"

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0
PARAMETER num_predict 50

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"
"""

    return content


def method_direct_import():
    """Import GuardReasoner-8B directly from safetensors (Ollama 0.1.14+)."""

    print("=" * 60)
    print("Method 1: Direct Safetensors Import")
    print("=" * 60)

    # Check Ollama version
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Ollama not found. Install from https://ollama.ai")
        return False

    print("\nSteps to import GuardReasoner-8B:")
    print("-" * 60)

    print("""
1. Download the model from HuggingFace:

   pip install huggingface_hub
   huggingface-cli download yueliu1999/GuardReasoner-8B --local-dir ./guardreasoner-8b

2. Create Modelfile (already created):

   See: Modelfile.guardreasoner

3. Create Ollama model:

   ollama create guardreasoner-8b -f Modelfile.guardreasoner

4. Test the model:

   ollama run guardreasoner-8b "How do I pick a lock?"
""")

    # Create Modelfile
    modelfile_path = Path(__file__).parent / "Modelfile.guardreasoner"
    modelfile_content = create_modelfile(model_path="./guardreasoner-8b")

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"\nModelfile created: {modelfile_path}")

    # Create download script
    download_script = """#!/bin/bash
# Download GuardReasoner-8B from HuggingFace

echo "Installing huggingface_hub..."
pip install huggingface_hub

echo "Downloading GuardReasoner-8B..."
huggingface-cli download yueliu1999/GuardReasoner-8B --local-dir ./guardreasoner-8b

echo "Creating Ollama model..."
ollama create guardreasoner-8b -f Modelfile.guardreasoner

echo "Testing model..."
ollama run guardreasoner-8b "What is 2+2?"

echo "Done!"
"""

    script_path = Path(__file__).parent / "download_and_import.sh"
    with open(script_path, "w") as f:
        f.write(download_script)
    script_path.chmod(0o755)

    print(f"Download script created: {script_path}")
    print("\nRun: ./download_and_import.sh")

    return True


def method_gguf_conversion():
    """Convert GuardReasoner-8B to GGUF via llama.cpp."""

    print("=" * 60)
    print("Method 2: GGUF Conversion via llama.cpp")
    print("=" * 60)

    print("""
This method requires llama.cpp for conversion.

Steps:
------

1. Clone llama.cpp:

   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make

2. Download the model:

   pip install huggingface_hub
   huggingface-cli download yueliu1999/GuardReasoner-8B --local-dir ./guardreasoner-8b

3. Convert to GGUF:

   python convert_hf_to_gguf.py ./guardreasoner-8b --outfile guardreasoner-8b-f16.gguf --outtype f16

4. Quantize (optional, for smaller size):

   ./llama-quantize guardreasoner-8b-f16.gguf guardreasoner-8b-q4_k_m.gguf Q4_K_M

5. Create Ollama model:

   # Copy GGUF to this directory
   cp guardreasoner-8b-q4_k_m.gguf /path/to/vincents/

   # Create model
   ollama create guardreasoner-8b -f Modelfile.guardreasoner.gguf

6. Test:

   ollama run guardreasoner-8b "How do I pick a lock?"
""")

    # Create Modelfile for GGUF
    modelfile_path = Path(__file__).parent / "Modelfile.guardreasoner.gguf"
    modelfile_content = create_modelfile(quantization="Q4_K_M")

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"\nModelfile created: {modelfile_path}")

    # Create full conversion script
    conversion_script = """#!/bin/bash
# Convert GuardReasoner-8B to GGUF format

set -e

echo "=== GuardReasoner-8B to GGUF Conversion ==="

# Check if llama.cpp exists
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make -j
    cd ..
fi

# Download model
if [ ! -d "guardreasoner-8b" ]; then
    echo "Downloading GuardReasoner-8B..."
    pip install huggingface_hub
    huggingface-cli download yueliu1999/GuardReasoner-8B --local-dir ./guardreasoner-8b
fi

# Convert to GGUF
echo "Converting to GGUF (F16)..."
cd llama.cpp
python convert_hf_to_gguf.py ../guardreasoner-8b --outfile ../guardreasoner-8b-f16.gguf --outtype f16

# Quantize
echo "Quantizing to Q4_K_M..."
./llama-quantize ../guardreasoner-8b-f16.gguf ../guardreasoner-8b-q4_k_m.gguf Q4_K_M
cd ..

# Create Ollama model
echo "Creating Ollama model..."
ollama create guardreasoner-8b -f Modelfile.guardreasoner.gguf

# Test
echo "Testing model..."
ollama run guardreasoner-8b "What is 2+2?"

echo "=== Conversion Complete ==="
echo "Model: guardreasoner-8b"
echo "GGUF: guardreasoner-8b-q4_k_m.gguf"
"""

    script_path = Path(__file__).parent / "convert_to_gguf.sh"
    with open(script_path, "w") as f:
        f.write(conversion_script)
    script_path.chmod(0o755)

    print(f"Conversion script created: {script_path}")
    print("\nRun: ./convert_to_gguf.sh")

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert GuardReasoner-8B to Ollama")
    parser.add_argument("--method", choices=["direct", "gguf"], default="direct",
                        help="Conversion method: direct (safetensors) or gguf")
    parser.add_argument("--modelfile-only", action="store_true",
                        help="Only create Modelfile, skip instructions")

    args = parser.parse_args()

    print("=" * 60)
    print("GuardReasoner-8B to Ollama Conversion")
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Base: Llama-3.1-8B (supported by Ollama)")
    print(f"Target: {OLLAMA_MODEL_NAME}")

    if args.modelfile_only:
        # Just create both Modelfiles
        modelfile_path = Path(__file__).parent / "Modelfile.guardreasoner"
        with open(modelfile_path, "w") as f:
            f.write(create_modelfile(model_path="./guardreasoner-8b"))
        print(f"\nCreated: {modelfile_path}")

        modelfile_gguf_path = Path(__file__).parent / "Modelfile.guardreasoner.gguf"
        with open(modelfile_gguf_path, "w") as f:
            f.write(create_modelfile(quantization="Q4_K_M"))
        print(f"Created: {modelfile_gguf_path}")
        return

    if args.method == "direct":
        method_direct_import()
    else:
        method_gguf_conversion()

    print("\n" + "=" * 60)
    print("Expected Performance After Conversion")
    print("=" * 60)
    print("""
Current (HuggingFace + bitsandbytes):
  - Inference: ~8s per sample
  - Memory: 5GB VRAM
  - Bottleneck: Python overhead, unoptimized kernels

After Ollama conversion:
  - Inference: ~0.5-1s per sample (8-16x faster)
  - Memory: ~4-5GB VRAM (Q4_K_M)
  - Benefits: Native Metal/CUDA, optimized kernels

Note: Speed depends on hardware. Numbers based on M1/M2 Mac or
decent CUDA GPU.
""")


if __name__ == "__main__":
    main()
