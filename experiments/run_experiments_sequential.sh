#!/bin/bash
# Run experiments 08-11 sequentially after 06 completes

cd /Users/vincent/development/wizard101/experiments

echo "Waiting for experiment 06 to complete..."
while ssh vincent@nigel.birs.ca "ps aux | grep 'experiment_06_matrix.py' | grep -v grep" > /dev/null; do
    sleep 60  # Check every minute
done

echo "Experiment 06 complete. Starting sequential runs..."

# Run experiment 08
echo "Starting experiment 08 (Baseline + Medium)..."
ssh vincent@nigel.birs.ca "cd ~/wizard101/experiments && python3 experiment_08_matrix.py > exp_08.log 2>&1"
echo "Experiment 08 complete."

# Run experiment 09
echo "Starting experiment 09 (Baseline + Verbose)..."
ssh vincent@nigel.birs.ca "cd ~/wizard101/experiments && python3 experiment_09_matrix.py > exp_09.log 2>&1"
echo "Experiment 09 complete."

# Run experiment 10
echo "Starting experiment 10 (Safeguard + Medium)..."
ssh vincent@nigel.birs.ca "cd ~/wizard101/experiments && python3 experiment_10_matrix.py > exp_10.log 2>&1"
echo "Experiment 10 complete."

# Run experiment 11
echo "Starting experiment 11 (Safeguard + Verbose)..."
ssh vincent@nigel.birs.ca "cd ~/wizard101/experiments && python3 experiment_11_matrix.py > exp_11.log 2>&1"
echo "Experiment 11 complete."

echo ""
echo "=================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=================================="
echo "Collecting results..."

ssh vincent@nigel.birs.ca "cd ~/wizard101/experiments && grep 'MULTI-POLICY ACCURACY' exp_*.log"
