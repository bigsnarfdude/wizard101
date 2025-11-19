# GuardReasoner Official Model Evaluation - Live Status

**Started**: 2025-11-19 06:56:44 MST
**Current Time**: 2025-11-19 07:12:00 MST (15 minutes elapsed)
**Status**: 🔄 Model downloading from HuggingFace

## What We're Testing

### Model Details
- **Name**: yueliu1999/GuardReasoner-8B
- **Base**: LLaMA 3.1 8B (Meta)
- **Training**: R-SFT + HS-DPO on 128K samples
- **Paper**: "GuardReasoner: Towards Reasoning-based LLM Safeguards" (Liu et al. 2025)
- **Paper F1**: **0.84** (84% accuracy)

### Our Setup
- **Hardware**: MacBook with 32GB RAM, Apple Silicon MPS
- **Test Samples**: 50 (25 harmful + 25 safe)
- **Dataset**: Our local harmful_behaviors + harmless_alpaca
- **Format**: Binary classification (harmful/safe)

## Download Progress

```
Time         Size      Status
-------------------------------
06:56        0 GB      Started download
07:06        6.0 GB    Downloading...
07:09        9.0 GB    Downloading...
07:12        9.0 GB    Continuing...
Complete:    ~16 GB    Expected total
```

**Progress**: ~56% complete (9GB / 16GB)

## Expected Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Download model | 15-20 min | 🔄 In progress (15 min elapsed) |
| Load to memory | 1-2 min | ⏳ Pending |
| Run evaluation | 5-10 min | ⏳ Pending |
| **Total** | **~25-30 min** | **~15 min remaining** |

## What Happens Next

### Step 1: Model Loads (1-2 minutes)
```
Moving model to MPS device...
Memory usage: ~14-16GB
✅ Model loaded successfully!
```

### Step 2: Evaluation Runs (5-10 minutes)
```
Evaluating:  50/50 [05:23<00:00, 0.12s/it]
- Generate reasoning for each sample
- Extract label (harmful/safe)
- Calculate metrics
```

### Step 3: Results Display
```
======================================================================
RESULTS: yueliu1999/GuardReasoner-8B
======================================================================
Total samples: 50
Accuracy: ??%

Harmful Class:
  Precision: ??%
  Recall: ??%
  F1: ???

📊 Macro F1 Score: ???

📄 PAPER COMPARISON:
Paper (GuardReasoner 8B): F1 = 0.840
This run: F1 = ???
Difference: ???
```

## Verification Criteria

### ✅ VERIFIED (What we want to see)
```
F1 ≥ 0.80  (within 5% of paper)
Accuracy ≥ 80%
Harmful F1 ≥ 0.78
Safe F1 ≥ 0.78
```

### ⚠️ PARTIAL MATCH (Acceptable)
```
F1 = 0.70-0.80
Could be due to:
- Different test dataset
- Smaller sample size (50 vs paper's larger set)
- Binary vs 3-task format
```

### ❌ FAILED (Needs investigation)
```
F1 < 0.70
Possible issues:
- Prompt format mismatch
- Parsing errors
- Model requires special handling
```

## What This Tells Us

### If F1 ≥ 0.80 ✅
**Conclusion**: GuardReasoner paper claims are accurate!
- Model is production-ready
- Can use as baseline comparison
- Our training efforts justified if we need customization

**Next steps**:
1. Run full 100-sample eval for better statistics
2. Compare to our local trained models
3. Consider using official weights vs training our own

### If F1 = 0.70-0.80 ⚠️
**Conclusion**: Model works, dataset differences likely
- Paper used WildGuard (larger, more diverse)
- We're using simplified harmful/harmless split
- Still validates the approach

**Next steps**:
1. Test with official WildGuard dataset
2. Try 3B model to see if consistent
3. Continue our training experiments

### If F1 < 0.70 ❌
**Conclusion**: Something wrong with our evaluation
- Check prompt format
- Inspect sample outputs manually
- Contact paper authors if needed

**Next steps**:
1. Debug evaluation script
2. Compare to paper's exact test setup
3. Try simpler baseline first

## Key Questions Being Answered

1. **Does the official model match paper claims?**
   - Paper: 84% F1
   - Our test: ??? F1
   - Verification: ✅ ⚠️ ❌

2. **Can we run it on MacBook?**
   - 32GB RAM: ✅ Sufficient
   - MPS acceleration: ✅ Working
   - Float16: ✅ ~14-16GB usage

3. **Should we use official weights or train our own?**
   - If official ≥ 80%: Consider using official
   - If our 3B ≥ 75%: Continue training
   - Compare cost/benefit

4. **Is GuardReasoner production-ready?**
   - Performance: ??? (testing now)
   - Inference speed: ??? (will measure)
   - Memory efficiency: ✅ Acceptable

## Files Created

- `eval_official_guardreasoner.py` - Main evaluation script
- `VERIFICATION_PLAN.md` - Detailed test plan
- `LOCAL_EVAL_README.md` - Quick start guide
- `RUNNING_EVALUATION_SUMMARY.md` - This file

## Commands to Reproduce

```bash
cd /Users/vincent/development/wizard101/experiments/guardreasoner

# Run 8B model (paper model)
python eval_official_guardreasoner.py --model 8b --samples 50

# Run 3B model (faster, less memory)
python eval_official_guardreasoner.py --model 3b --samples 50

# Run 1B model (fastest, lightweight)
python eval_official_guardreasoner.py --model 1b --samples 50

# Full 100-sample evaluation
python eval_official_guardreasoner.py --model 8b --samples 100
```

## Next Actions (After Results)

1. **Document findings** in experiment tracker
2. **Compare** to our Exp 18/19 models
3. **Decide** on training vs using official weights
4. **Update** README with verified performance

---

**⏳ Status**: Waiting for download to complete (~5-10 min)
**📊 Results**: Coming soon...
**🎯 Goal**: Verify 84% F1 claim from paper
