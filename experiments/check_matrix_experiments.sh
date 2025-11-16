#!/bin/bash
# Monitor experiment matrix (06, 08-11)

echo "=================================="
echo "EXPERIMENT MATRIX STATUS"
echo "=================================="
echo ""

for exp in 06 08 09 10 11; do
    echo "Experiment $exp:"
    echo "-----------------------------------"
    ssh vincent@nigel.birs.ca "cd ~/wizard101/experiments && tail -10 exp_${exp}.log 2>/dev/null || echo 'No log file yet'"
    echo ""
done

echo "=================================="
echo "Running processes:"
ssh vincent@nigel.birs.ca "ps aux | grep 'experiment_.*_matrix.py' | grep -v grep || echo 'No processes running'"
echo "=================================="
