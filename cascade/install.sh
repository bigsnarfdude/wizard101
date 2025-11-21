#!/bin/bash
# Safety Cascade Installation Script
# Requires: CUDA-capable GPU with 24GB+ VRAM

set -e

echo "=============================================="
echo "Safety Cascade - Installation Script"
echo "=============================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not found"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    # Check VRAM
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$VRAM" -lt 24000 ]; then
        echo ""
        echo "WARNING: Detected ${VRAM}MB VRAM. Recommended: 24GB+"
        echo "L1 and higher tiers may not load properly."
    else
        echo "VRAM: ${VRAM}MB - OK"
    fi
else
    echo ""
    echo "WARNING: NVIDIA GPU not detected. CPU mode will be very slow."
fi

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Downloading Models"
echo "=============================================="

# Create models directory
mkdir -p models

# Download L0 Bouncer from HuggingFace
echo ""
echo "Downloading L0 Bouncer (DeBERTa, ~90MB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vincentoh/deberta-v3-xsmall-l0-bouncer-full',
    local_dir='./models/l0_bouncer',
    local_dir_use_symlinks=False
)
print('L0 Bouncer downloaded successfully!')
"

# Download L1 base model and LoRA
echo ""
echo "Downloading L1 Analyst base model (Llama 3.2 3B, ~2GB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='unsloth/Llama-3.2-3B-Instruct',
    local_dir='./models/llama-3.2-3b',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.gguf', '*.ggml']
)
print('Llama 3.2 3B downloaded successfully!')
"

echo ""
echo "Downloading L1 LoRA adapter (GuardReasoner, ~50MB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vincentoh/Llama-3.2-3B-GuardReasoner-Exp18',
    local_dir='./models/guardreasoner_lora',
    local_dir_use_symlinks=False
)
print('GuardReasoner LoRA downloaded successfully!')
"

echo ""
echo "=============================================="
echo "Ollama Setup (for L2/L3)"
echo "=============================================="

# Check for Ollama
if command -v ollama &> /dev/null; then
    echo "Ollama found!"

    echo ""
    echo "Pulling gpt-oss:20b for L2 Gauntlet..."
    ollama pull gpt-oss:20b || echo "WARNING: Could not pull gpt-oss:20b"

    echo ""
    echo "Note: For L3 Judge, you'll need gpt-oss:120b"
    echo "Run: ollama pull gpt-oss:120b (requires ~70GB)"
else
    echo ""
    echo "WARNING: Ollama not found. L2 and L3 tiers will not work."
    echo "Install Ollama from: https://ollama.ai"
    echo ""
    echo "After installing Ollama, run:"
    echo "  ollama pull gpt-oss:20b   # For L2 Gauntlet"
    echo "  ollama pull gpt-oss:120b  # For L3 Judge (optional)"
fi

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Quick test:"
echo "  python3 example.py"
echo ""
echo "Full cascade:"
echo "  python3 -c 'from cascade import SafetyCascade; c = SafetyCascade(); print(c.classify(\"Hello\"))'"
echo ""
echo "Requirements:"
echo "  - L0 only: 4GB+ VRAM"
echo "  - L0+L1: 8GB+ VRAM"
echo "  - Full cascade: 24GB+ VRAM + Ollama"
echo ""
