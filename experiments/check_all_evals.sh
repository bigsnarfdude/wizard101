#!/bin/bash
# Monitor all 3 performance evaluations

echo "=================================="
echo "ALL PERFORMANCE EVALUATIONS STATUS"
echo "=================================="
echo ""

echo "1. BALANCED (90 samples, synthetic)"
echo "-----------------------------------"
ssh user@server "cd ~/wizard101/experiments && tail -20 eval_perf_balanced.log"
echo ""

echo "2. LARGER (270 samples, synthetic)"
echo "-----------------------------------"
ssh user@server "cd ~/wizard101/experiments && tail -20 eval_perf_larger.log"
echo ""

echo "3. WILDGUARD (300 samples, REAL)"
echo "-----------------------------------"
ssh user@server "cd ~/wizard101/experiments && tail -20 eval_perf_wildguard.log"
echo ""

echo "=================================="
echo "To monitor live: ssh user@server 'cd ~/wizard101/experiments && tail -f eval_perf_*.log'"
echo "=================================="
