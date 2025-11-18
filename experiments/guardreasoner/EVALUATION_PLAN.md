# Experiment 18: Evaluation Decision Plan

## Quick Summary
We trained GuardReasoner for **1 epoch** (paper used 5). Before training 4 more epochs, evaluate first.

## Option 2: Evaluate First ✅ SELECTED

### Step 1: Run Evaluation
```bash
# On nigel.birs.ca
cd ~/wizard101/experiments/guardreasoner
python evaluate_exp_18.py  # TODO: Create this script
```

**Test Dataset**: WildGuard test set (1,554 samples)
**Baseline**: Exp 12 = 57.5% accuracy

### Step 2: Decision Tree

```
Evaluation Results?
│
├─ ≥ 60% accuracy ✅
│  ├─ SUCCESS: 1 epoch is sufficient!
│  ├─ Action: Publish current model
│  ├─ Conclusion: Sample efficiency validated
│  └─ Next: Write paper on efficient training
│
└─ < 60% accuracy 🔄
   ├─ Need more training
   ├─ Action: Continue for 4 more epochs
   ├─ Expected time: 32 hours (4 × 8 hours)
   └─ Next: Re-evaluate after epoch 5
```

### Step 3: Evaluation Metrics

**Primary Metric**: Accuracy on WildGuard test
- Target: ≥ 60%
- Baseline: 57.5%
- Paper (Llama-3.1-8B): ~70-75% (estimated)

**Secondary Metrics**:
- Harmful F1 score
- Harmless F1 score
- False positive rate (safe content flagged as harmful)
- False negative rate (harmful content marked as safe)

**Qualitative Analysis**:
- Manual inspection of 50 reasoning traces
- Are they coherent?
- Are they relevant to the safety judgment?
- Do they help explain the decision?

## Hypothesis Testing

### Hypothesis 1: Sample Efficiency
**Claim**: With reasoning traces, 1 epoch on 11k samples is sufficient
**Evidence for**: Paper showed 500 samples × 50 epochs worked well
**Evidence against**: Paper still used 5 epochs on full dataset
**Test**: If accuracy ≥ 60%, hypothesis supported

### Hypothesis 2: Model Size Doesn't Matter
**Claim**: 3B model can match 8B model performance
**Evidence for**: Reasoning provides strong signal
**Evidence against**: Paper used 8B for a reason
**Test**: Compare our results to paper's reported metrics

### Hypothesis 3: Learning Rate Compensation
**Claim**: Higher LR (5e-5) compensates for fewer epochs
**Evidence for**: 1 epoch with aggressive learning matches 5 epochs gentle learning
**Evidence against**: Might cause instability or poor generalization
**Test**: Check if loss is stable and performance is good

## Next Steps Based on Results

### If ≥ 60% (Success Path)
1. ✅ Document training efficiency gains
2. ✅ Write model card highlighting 1-epoch training
3. ✅ Blog post: "Efficient Safety Tuning in 8 Hours"
4. ✅ Compare compute costs: 8h vs 40h
5. ✅ Release as production-ready model

### If < 60% (Continue Training Path)
1. 🔄 Start 4-epoch continuation training
2. 🔄 Evaluate after each epoch (epochs 2, 3, 4, 5)
3. 🔄 Plot training curve
4. 🔄 Find optimal stopping point
5. 🔄 Release best checkpoint

## Timeline

**Today (2025-11-18)**:
- [ ] Create evaluation script
- [ ] Run evaluation on 1-epoch model
- [ ] Analyze results
- [ ] Make decision

**If continuing training**:
- Day 1-2: Epoch 2 (8 hours)
- Day 2-3: Epoch 3 (8 hours)
- Day 3-4: Epoch 4 (8 hours)
- Day 4-5: Epoch 5 (8 hours)
- Day 5: Final evaluation

**If 1 epoch sufficient**:
- Today: Write blog post
- Tomorrow: Share on Twitter/HF
- Week: Submit to conference/workshop

## Cost-Benefit Analysis

### 1 Epoch Model
- **Time**: 8 hours ✅
- **Cost**: ~$5-10 GPU time ✅
- **Risk**: Might underperform 🤷
- **Reward**: Highly efficient if it works 🏆

### 5 Epoch Model
- **Time**: 40 hours ⏰
- **Cost**: ~$25-50 GPU time 💰
- **Risk**: Might overfit 🤷
- **Reward**: Matches paper exactly 📄

## Paper Claims to Validate

From GuardReasoner paper (arXiv:2505.20087):

✅ **Already validated**:
- [x] Training on reasoning traces works
- [x] Loss decreases smoothly
- [x] Training completes without errors

❓ **Need evaluation to validate**:
- [ ] Sample efficiency (5k samples sufficient)
- [ ] Overfitting resistance
- [ ] Reasoning quality maintained
- [ ] Performance matches or exceeds baselines

🔄 **Need 5 epochs to validate**:
- [ ] Training curve shape
- [ ] Optimal epoch count
- [ ] Overfitting resistance with 50 epochs

## Files to Create

### Evaluation Script
```python
# evaluate_exp_18.py
# Load model from exp_18_rsft_lora/
# Load WildGuard test set
# Run inference with reasoning
# Calculate metrics
# Save results to results/exp_18_evaluation.json
```

### Results Analysis Script
```python
# analyze_exp_18.py
# Load results JSON
# Compare to baseline (Exp 12: 57.5%)
# Generate plots
# Print decision recommendation
```

### Continuation Script (if needed)
```python
# continue_exp_18.py
# Resume from exp_18_rsft_lora/
# Train for 4 more epochs
# Save checkpoints after each epoch
# Evaluate after each epoch
```

## Success Criteria Summary

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Accuracy | 55% | 60% | 65% |
| Harmful F1 | 0.50 | 0.60 | 0.70 |
| Harmless F1 | 0.50 | 0.60 | 0.70 |
| FPR | <0.15 | <0.10 | <0.05 |
| FNR | <0.20 | <0.15 | <0.10 |

**Decision Rule**:
- All metrics ≥ Minimum + Accuracy ≥ Target = ✅ Ship it!
- Any metric < Minimum OR Accuracy < Target = 🔄 Train more

---

**Created**: 2025-11-18
**Experiment**: 18 (R-SFT 1 epoch)
**Status**: Ready for Phase 1 evaluation
**Next**: Create and run evaluation script
