#!/bin/bash
# Monitor training progress for Experiment 18 (Full 3-epoch R-SFT)

LOG_FILE="$HOME/wizard101/experiments/guardreasoner/logs/exp_18_full_3epoch.log"
TOTAL_STEPS=3987

echo "======================================================================"
echo "EXPERIMENT 18: Full 3-Epoch R-SFT Training Monitor"
echo "======================================================================"
echo ""

# Check if training is running
if screen -ls | grep -q "guardreasoner-train"; then
    echo "✅ Training screen session is ACTIVE"
else
    echo "❌ Training screen session NOT FOUND"
    exit 1
fi

echo ""
echo "GPU Status:"
echo "----------------------------------------------------------------------"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU Util: %s%%  |  Mem Util: %s%%  |  Temp: %s°C  |  VRAM: %sMB / %sMB\n", $1, $2, $3, $4, $5}'

echo ""
echo "Training Progress:"
echo "----------------------------------------------------------------------"

# Extract latest step from log
LATEST_LINE=$(tail -1 "$LOG_FILE")
if echo "$LATEST_LINE" | grep -q "it/s"; then
    # Extract step number from progress bar
    CURRENT_STEP=$(echo "$LATEST_LINE" | grep -oP '\|\s*\K[0-9]+(?=/[0-9]+)')

    if [ -n "$CURRENT_STEP" ]; then
        PERCENT=$(echo "scale=2; $CURRENT_STEP * 100 / $TOTAL_STEPS" | bc)
        REMAINING=$(echo "$TOTAL_STEPS - $CURRENT_STEP" | bc)

        echo "  Current Step: $CURRENT_STEP / $TOTAL_STEPS"
        echo "  Progress: $PERCENT%"
        echo "  Remaining: $REMAINING steps"

        # Extract time per iteration
        TIME_PER_IT=$(echo "$LATEST_LINE" | grep -oP '[0-9.]+(?=s/it)')
        if [ -n "$TIME_PER_IT" ]; then
            REMAINING_HOURS=$(echo "scale=1; $REMAINING * $TIME_PER_IT / 3600" | bc)
            echo "  Estimated remaining time: $REMAINING_HOURS hours"
        fi
    else
        echo "  Parsing progress..."
    fi
fi

echo ""
echo "Recent Log (last 20 lines):"
echo "----------------------------------------------------------------------"
tail -20 "$LOG_FILE"

echo ""
echo "======================================================================"
echo "Commands:"
echo "  Watch live: tail -f $LOG_FILE"
echo "  Attach to screen: screen -r guardreasoner-train"
echo "  Check GPU: nvidia-smi"
echo "======================================================================"
