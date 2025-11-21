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

cd "$SCRIPT_DIR/.."

# Run L0
echo "Running L0..."
python eval_layered_batch.py --layer l0 --benchmark all --output "$OUTPUT_DIR/l0_results.json"

# Run L1
echo "Running L1..."
python eval_layered_batch.py --layer l1 --input "$OUTPUT_DIR/l0_results.json" --output "$OUTPUT_DIR/l1_results.json"

# Run L2
echo "Running L2..."
python eval_layered_batch.py --layer l2 --input "$OUTPUT_DIR/l1_results.json" --output "$OUTPUT_DIR/l2_results.json"

# Combine and score
echo "Combining results..."
python eval_layered_batch.py --combine --input "$OUTPUT_DIR/l2_results.json" --output "$OUTPUT_DIR/final_scores.json"

echo ""
echo "============================================================"
echo "COMPLETE - Results in $OUTPUT_DIR"
echo "============================================================"
