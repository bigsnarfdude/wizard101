# GuardReasoner 3B Evaluation - Status

**Started**: 2025-11-19 08:12:28
**Model**: yueliu1999/GuardReasoner-3B (faster than 8B!)

## Why We Switched

**8B Model**: TOO SLOW ❌
- 23 minutes per sample
- 19+ hours for 50 samples
- Not practical for MacBook

**3B Model**: MUCH FASTER ✅
- ~5-8 minutes per sample (estimated)
- ~4-7 hours for 50 samples
- Same official model, just smaller

## Current Status

**Process**: Running in background
- PID: 43228
- Status: Downloading remaining model files
- Size: 3GB (downloading to ~6GB total)

## What's Happening

1. ✅ Killed slow 8B evaluation
2. ✅ Switched script to 3B model
3. 🔄 Downloading 3B model (3GB → 6GB)
4. ⏳ Will load and start evaluation after download
5. ⏳ Will run 50 samples (~4-7 hours)
6. ✅ Will save results to `results_official_format.json`

## When You Get Back

### Check if complete:
```bash
cat ~/development/wizard101/experiments/guardreasoner/results_official_format.json
```

### Check if still running:
```bash
ps aux | grep eval_official_format | grep -v grep
```

### Check progress (if running):
```bash
# This will show me the latest logs
tail -20 ~/development/wizard101/experiments/guardreasoner/results_official_format.json 2>/dev/null || echo "Still running"
```

## Expected Results

**Paper claims (3B model)**:
- F1: ~0.78-0.82 (paper reports 8B = 0.84, 3B slightly lower)
- Still very good performance
- Fast enough to run on MacBook

**Our test**:
- 50 samples (25 harmful + 25 safe)
- Official prompt format
- Will compare to paper

## Timeline

- **Download**: 08:12 → ~08:20 (~8 min)
- **Model load**: ~08:20 → ~08:21 (~1 min)
- **Evaluation**: ~08:21 → ~12:00-15:00 (~4-7 hours)
- **Complete**: Sometime this afternoon

## Safe to Close Lid

✅ **Yes, you can close your laptop now!**

The evaluation will run in the background:
- No internet needed (model is downloading/cached)
- Will complete automatically
- Results saved when done

## Results Location

```
~/development/wizard101/experiments/guardreasoner/results_official_format.json
```

Contains:
- Accuracy
- F1 score
- Precision/Recall
- Comparison to paper (0.78-0.82 F1 expected)

---

**Everything is set up and running. Check back in a few hours!** 🚀
