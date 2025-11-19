# GuardReasoner Evaluation - Running in Background

**Started**: 2025-11-19 07:28:42
**Status**: ✅ Download + evaluation running automatically

## What's Happening

**Process PID**: 40616 (running)
**Script**: `eval_official_format.py`
**Location**: `/Users/vincent/development/wizard101/experiments/guardreasoner/`

### Download Progress
- **Model**: yueliu1999/GuardReasoner-8B
- **Size**: ~16GB total
- **Currently**: ~13GB downloaded (3 files still downloading)
- **ETA**: Should complete in 10-30 minutes depending on internet

### What Happens Automatically
1. ✅ Download completes (~16GB)
2. ✅ Model loads into RAM (~18GB)
3. ✅ Evaluates 50 test samples (~5-10 min)
4. ✅ Saves results to `results_official_format.json`
5. ✅ Process exits

## When You Come Back

### Check If Complete
```bash
# Check if process still running
ps aux | grep "eval_official_format" | grep -v grep

# If empty → DONE!
# If shows process → Still running
```

### View Results
```bash
cd /Users/vincent/development/wizard101/experiments/guardreasoner

# Check if results file exists
ls -lh results_official_format.json

# View results
cat results_official_format.json | jq '.metrics'
```

### Expected Results File
```json
{
  "metrics": {
    "total": 50,
    "accuracy": 0.XX,
    "f1": 0.XXX,
    "precision": 0.XX,
    "recall": 0.XX
  }
}
```

### Check Download Progress (If Still Running)
```bash
# See model cache size
du -sh ~/.cache/huggingface/hub/models--yueliu1999--GuardReasoner-8B

# Check incomplete files
ls -lh ~/.cache/huggingface/hub/models--yueliu1999--GuardReasoner-8B/blobs/*.incomplete

# If no .incomplete files → Download done, evaluation running
```

## Timeline

### Current Status (07:33)
- Download: ~13GB / 16GB (81% complete)
- Estimated completion: 07:45 - 08:00

### Full Timeline
1. Download finish: 07:45 - 08:00 (~15-30 min)
2. Model load: 08:00 - 08:02 (~2 min)
3. Evaluation run: 08:02 - 08:12 (~10 min)
4. **COMPLETE**: ~08:15 (total ~45-50 min from 07:28 start)

## Quick Check Commands

```bash
# One-liner status check
cd /Users/vincent/development/wizard101/experiments/guardreasoner && \
echo "Download:" && du -sh ~/.cache/huggingface/hub/models--yueliu1999--GuardReasoner-8B && \
echo "Process:" && ps aux | grep eval_official_format | grep -v grep && \
echo "Results:" && ls -lh results_official_format.json 2>/dev/null || echo "Not ready yet"
```

## If Process Died (Unlikely)

```bash
# Restart evaluation
cd /Users/vincent/development/wizard101/experiments/guardreasoner
nohup python eval_official_format.py > eval_output.log 2>&1 &

# Monitor
tail -f eval_output.log
```

## What We're Verifying

**Paper Claim**: GuardReasoner-8B achieves **84% F1** on safety classification

**Our Test**:
- Official model: `yueliu1999/GuardReasoner-8B`
- Official prompt format from GitHub
- 50 test samples (25 harmful + 25 safe)

**Success**:
- F1 ≥ 0.80 → Paper verified! ✅
- F1 = 0.70-0.80 → Acceptable (dataset difference) ⚠️
- F1 < 0.70 → Investigate ❌

---

**Everything is running automatically. Just check back in an hour!**

**Results will be in**: `results_official_format.json`
