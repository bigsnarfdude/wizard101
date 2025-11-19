# GuardReasoner Verification Plan

## Goal
Verify that the official HuggingFace `yueliu1999/GuardReasoner-8B` model performs as claimed in the paper (84% F1 score).

## Official Models Available

| Model | HF Repository | Base Model | Parameters | Paper F1 |
|-------|---------------|------------|------------|----------|
| 1B | yueliu1999/GuardReasoner-1B | LLaMA 3.2 1B | 1B | ~0.75-0.80 |
| 3B | yueliu1999/GuardReasoner-3B | LLaMA 3.2 3B | 3B | ~0.78-0.82 |
| **8B** | **yueliu1999/GuardReasoner-8B** | **LLaMA 3.1 8B** | **8B** | **0.84** |

## Paper Claims (Liu et al. 2025)

**GuardReasoner-8B Performance:**
- **Overall F1**: 0.84 (macro average)
- **Prompt Harmful F1**: 0.87
- **Response Refusal F1**: 0.81
- **Response Harmful F1**: 0.82

**Comparisons:**
- Beats GPT-4o+CoT by **5.74% F1**
- Beats LLaMA Guard 3 8B by **20.84% F1**
- Achieves SOTA on safety classification

## Test Configuration

### Dataset
- **Source**: Our local test data
  - `harmful_behaviors_test.json` (100 harmful prompts)
  - `harmless_alpaca_test.json` (100 safe prompts)
- **Samples**: 50-100 (25-50 harmful + 25-50 safe)
- **Format**: Binary classification (harmful/safe)

### Evaluation Setup
- **Model**: `yueliu1999/GuardReasoner-8B` (official weights)
- **Precision**: float16 (~14-16GB RAM)
- **Device**: MPS (Apple Silicon) or CPU
- **Temperature**: 0.0 (greedy decoding for reproducibility)
- **Max tokens**: 512 (for detailed reasoning)

### Metrics
- **Accuracy**: Overall correctness
- **Precision**: When predicting "harmful", how often correct?
- **Recall**: Of all harmful content, how much detected?
- **F1 Score**: Harmonic mean of precision/recall
- **Macro F1**: Average of harmful F1 + safe F1

## Expected Results

### Best Case (Perfect Replication)
```
Macro F1: 0.84 ± 0.02
Accuracy: ~84%
Harmful F1: ~0.85
Safe F1: ~0.83
```

### Good Case (Close Match)
```
Macro F1: 0.80-0.83
Accuracy: ~80-83%
Harmful F1: ~0.81-0.84
Safe F1: ~0.79-0.82
```

### Acceptable Case (Dataset Differences)
```
Macro F1: 0.75-0.80
Accuracy: ~75-80%
Harmful F1: ~0.76-0.81
Safe F1: ~0.74-0.79
```

### Concerning (Something Wrong)
```
Macro F1: < 0.70
Accuracy: < 70%
→ Investigate: prompt format, parsing logic, dataset mismatch
```

## Potential Issues & Solutions

### Issue 1: Lower Performance Than Paper
**Possible causes:**
- Different test dataset (paper uses WildGuard, we use simplified)
- Prompt format mismatch
- Parsing errors in extracting labels
- Paper uses 3-task format, we use binary

**Solutions:**
1. Check sample outputs manually
2. Try different prompt templates
3. Use their exact test data (WildGuard test set)

### Issue 2: Model Not Loading
**Possible causes:**
- Insufficient RAM (need ~16GB for 8B model)
- Network issues during download
- MPS compatibility issues

**Solutions:**
1. Try 3B model instead (yueliu1999/GuardReasoner-3B)
2. Use CPU fallback
3. Reduce samples to 25-50

### Issue 3: "Unknown" Predictions
**Possible causes:**
- Response parsing failures
- Model outputs different format than expected
- Reasoning doesn't end with clear label

**Solutions:**
1. Inspect full responses
2. Update parsing logic
3. Check tokenizer special tokens

## Timeline

### Phase 1: Initial Test (Current)
- ✅ Download 8B model (~5-10 min)
- 🔄 Run 50 sample evaluation (~5-10 min)
- 📊 Get initial F1 score

### Phase 2: Verification (If Results Look Good)
- Run 100 sample evaluation (~10-15 min)
- Compare to paper claims
- Document findings

### Phase 3: Extended Testing (Optional)
- Test 3B model for comparison
- Try different prompt formats
- Use official WildGuard test set

### Phase 4: Documentation
- Update experiment tracker
- Write findings report
- Share results

## Success Criteria

✅ **VERIFIED** if:
- Macro F1 ≥ 0.80 (within 5% of paper)
- Results consistent across multiple runs
- Sample predictions look reasonable

⚠️ **NEEDS INVESTIGATION** if:
- Macro F1 between 0.70-0.80
- High variance between runs
- Many "unknown" predictions

❌ **FAILED** if:
- Macro F1 < 0.70
- Model clearly not working as expected
- Results far below paper claims

## Next Steps After Verification

### If Verified (F1 ≥ 0.80)
1. ✅ Confirm GuardReasoner is production-ready
2. Compare 8B vs our 3B implementation
3. Consider using official weights instead of training from scratch
4. Focus on deployment and integration

### If Partially Verified (F1 0.70-0.80)
1. Test with official WildGuard dataset
2. Compare 8B vs 3B vs 1B models
3. Investigate prompt engineering improvements
4. Continue with our training experiments

### If Not Verified (F1 < 0.70)
1. Check if model requires specific input format
2. Contact paper authors for clarification
3. Try 3B/1B models to isolate issue
4. Focus on our own training pipeline

## Current Status

**Date**: 2025-11-19
**Status**: 🔄 Running initial 50-sample evaluation
**Model**: yueliu1999/GuardReasoner-8B (downloading)
**ETA**: ~10-15 minutes for complete results

---

**Update Log:**
- 2025-11-19 06:56: Started 8B model evaluation
- 2025-11-19 06:58: Model downloading from HuggingFace
- 2025-11-19 07:00: Waiting for download completion (~16GB)
