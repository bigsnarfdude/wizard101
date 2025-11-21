#!/bin/bash
# Check performance evaluation progress

echo "Checking performance evaluation status..."
echo "=========================================="

ssh user@server "cd ~/wizard101/experiments && tail -50 eval_performance_run.log"

echo ""
echo "=========================================="
echo "To monitor in real-time: ssh user@server 'cd ~/wizard101/experiments && tail -f eval_performance_run.log'"
