#!/bin/bash
# Test if L1 catches L0's dangerous misses

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

source "$SCRIPT_DIR/../guardreasoner/venv/bin/activate"

cd "$SCRIPT_DIR/.."

# Copy final_scores if needed
if [ ! -f "$OUTPUT_DIR/final_scores.json" ] && [ -f "cascade_evaluations/final_scores.json" ]; then
    cp cascade_evaluations/final_scores.json "$OUTPUT_DIR/"
fi

python eval_blindspot_test.py

# Move results to outputs
if [ -f "cascade_evaluations/blindspot_test_results.json" ]; then
    mv cascade_evaluations/blindspot_test_results.json "$OUTPUT_DIR/"
fi
