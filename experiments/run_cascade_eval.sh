#!/bin/bash
# Run full cascade evaluation with GPU monitoring
# Usage: ./run_cascade_eval.sh

set -e

echo "============================================================"
echo "CASCADE EVALUATION - HarmBench + XSTest + SimpleSafetyTests"
echo "============================================================"
echo ""

cd ~/wizard101/experiments

# Create results directory
mkdir -p cascade_evaluations

echo "=== Initial GPU State ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
echo ""

# Step 1: L0 Bouncer
echo "============================================================"
echo "STEP 1: Running L0 Bouncer on all benchmarks (1,050 samples)"
echo "============================================================"
python eval_layered_batch.py --layer l0 --benchmark all

echo ""
echo "=== GPU State after L0 (should be released) ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
echo ""

# Check if L1 is needed
L1_COUNT=$(python3 -c "import json; d=json.load(open('cascade_evaluations/l0_results.json')); print(sum(1 for r in d if r.get('needs_l1', False)))")
echo "Samples needing L1: $L1_COUNT"

if [ "$L1_COUNT" -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 2: Running L1 Analyst on uncertain samples ($L1_COUNT)"
    echo "============================================================"
    python eval_layered_batch.py --layer l1

    echo ""
    echo "=== GPU State after L1 (should be released) ==="
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
    echo ""

    # Check if L2 is needed
    L2_COUNT=$(python3 -c "import json; d=json.load(open('cascade_evaluations/l1_results.json')); print(sum(1 for r in d if r.get('needs_l2', False)))")
    echo "Samples needing L2: $L2_COUNT"

    if [ "$L2_COUNT" -gt 0 ]; then
        echo ""
        echo "============================================================"
        echo "STEP 3: Running L2 Gauntlet on disagreements ($L2_COUNT)"
        echo "============================================================"
        python eval_layered_batch.py --layer l2

        echo ""
        echo "=== GPU State after L2 (should be released) ==="
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
        echo ""
    fi
else
    echo "L0 handled all samples confidently. Skipping L1/L2."
    # Copy L0 results as final
    cp cascade_evaluations/l0_results.json cascade_evaluations/l2_results.json
fi

# Step 4: Combine and Score
echo ""
echo "============================================================"
echo "STEP 4: Combining results and calculating scores"
echo "============================================================"
python eval_layered_batch.py --combine

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - cascade_evaluations/l0_results.json"
echo "  - cascade_evaluations/l1_results.json"
echo "  - cascade_evaluations/l2_results.json"
echo "  - cascade_evaluations/final_scores.json"
echo ""
echo "=== Final GPU State ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
