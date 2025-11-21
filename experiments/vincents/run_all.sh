#!/bin/bash
# Vincent's Full Cascade Evaluation
# Self-contained - all outputs go to ./outputs/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "VINCENT'S CASCADE EVALUATION"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Activate venv
source "$SCRIPT_DIR/../guardreasoner/venv/bin/activate"

cd "$SCRIPT_DIR"

# Run L0
echo "Running L0..."
python eval_layered_batch.py --layer l0 --benchmark all

# Run L1
echo "Running L1..."
python eval_layered_batch.py --layer l1

# Run L2
echo "Running L2..."
python eval_layered_batch.py --layer l2

# Combine and score
echo "Combining results..."
python eval_layered_batch.py --combine

echo ""
echo "============================================================"
echo "COMPLETE - Results in $OUTPUT_DIR"
echo "============================================================"
