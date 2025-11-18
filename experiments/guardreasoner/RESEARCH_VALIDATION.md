# GuardReasoner Research Validation Plan

## Paper Reference
**Title**: GuardReasoner: Reasoning-based Safety Content Moderation
**ArXiv**: 2505.20087v1
**PDF**: `/Users/vincent/development/wizard101/2505.20087v1.pdf`

## Replication Goal
Validate the GuardReasoner approach using Llama-3.2-3B-Instruct with reasoning-guided supervised fine-tuning (R-SFT).

## Training Configuration Comparison

| Parameter | Paper (GuardReasoner) | Our Experiment 18 | Status |
|-----------|----------------------|-------------------|--------|
| Base Model | Llama-3.1-8B-Instruct | Llama-3.2-3B-Instruct | ⚠️ Smaller model |
| Training Method | R-SFT | R-SFT | ✅ Same |
| Dataset Size | ~11k samples | 11,396 samples | ✅ Similar |
| Epochs | 5 | 1 | ⚠️ 1/5 complete |
| Batch Size | 32 | 128 (effective) | ✅ Similar |
| Learning Rate | 1e-6 | 5e-5 | ⚠️ Higher LR |
| Hardware | 8xA100 (full precision) | 1xGPU (4-bit) | ⚠️ Quantized |
| Training Time | 1-4 hours | 8.09 hours | Expected |
| Approach | LoRA | LoRA | ✅ Same |

## Key Paper Findings to Validate

### 1. Sample Efficiency
**Paper claim**: "5,000 samples sufficient for strong performance"
- Performance plateaus at 5k samples
- Full dataset shows no substantial improvement over 5k subset
- Our dataset: 11,396 samples (2.3x the plateau point)

**Our validation**:
- ✅ Using sufficient training data
- ❓ Unknown: Did 1 epoch provide enough signal with 11k samples?

### 2. Overfitting Resistance
**Paper claim**: "Robust to overfitting even with 50 epochs"
- 500 samples × 50 epochs = within 3% of full dataset model
- Reasoning traces provide regularization effect

**Our hypothesis**:
- With 11,396 samples and reasoning traces, 1 epoch might be sufficient
- Higher learning rate (5e-5 vs 1e-6) might compensate for fewer epochs
- Need evaluation to validate

### 3. Reasoning Quality
**Paper claim**: "Reasoning traces improve interpretability and accuracy"
- Models generate step-by-step safety analysis
- Reasoning enables better generalization

**Our validation plan**:
- Evaluate reasoning quality on test set
- Compare to non-reasoning baseline
- Check if reasoning is coherent and helpful

## Evaluation Strategy: Option 2 (Evaluate First)

### Phase 1: Evaluate 1-Epoch Model
**Test Dataset**: WildGuard test set (1,554 samples)
**Baseline**: Exp 12 performance (57.5% accuracy)

**Success Criteria**:
- ✅ **≥ 60% accuracy**: 1 epoch sufficient (validates sample efficiency hypothesis)
- 🔄 **< 60% accuracy**: Need more training (continue to 5 epochs)

**Metrics to Collect**:
1. Overall accuracy
2. Per-category F1 scores (harmful/harmless)
3. False positive rate
4. False negative rate
5. Reasoning coherence (manual inspection)

### Phase 2: Decision Point
Based on Phase 1 results:

**Scenario A: Strong Performance (≥60%)**
- Document as successful replication with modifications
- Publish 1-epoch model as-is
- Note: "Sample efficiency validated with smaller model and fewer epochs"

**Scenario B: Weak Performance (<60%)**
- Continue training for 4 more epochs
- Re-evaluate at each epoch
- Document training curve
- Final model: 5-epoch checkpoint matching paper

### Phase 3: Publication Validation
**Required for research publication**:
- Training curves (loss, accuracy per epoch)
- Comparison to paper's reported metrics
- Ablation studies:
  - With reasoning vs without reasoning
  - 1 epoch vs 5 epochs (if we train both)
  - 3B model vs 8B model (if resources permit)
- Error analysis on failure cases
- Reasoning trace quality analysis

## Current Status

### ✅ Completed
- [x] Dataset preparation (11,396 samples with reasoning traces)
- [x] 1-epoch R-SFT training completed
- [x] Model uploaded to HuggingFace
- [x] Training metrics logged (loss: 0.833, time: 8.09h)

### ⏳ In Progress
- [ ] Phase 1: WildGuard test evaluation
- [ ] Reasoning quality analysis

### 📋 Pending Decision
- [ ] Phase 2: Continue to 5 epochs? (depends on Phase 1 results)
- [ ] Ablation studies for publication
- [ ] Error analysis and failure case investigation

## Research Questions

### Primary Questions
1. **Does the 1-epoch model achieve reasonable performance?**
   - Target: ≥60% on WildGuard test
   - Baseline: 57.5% (Exp 12)

2. **Is reasoning quality maintained with fewer epochs?**
   - Manual inspection of reasoning traces
   - Coherence and relevance to safety judgment

3. **Does sample efficiency hold with smaller model?**
   - Paper used 8B model, we use 3B
   - Paper used 5 epochs, we use 1
   - Can we achieve similar results with 1/8th the training compute?

### Secondary Questions
4. **What's the optimal learning rate for 4-bit training?**
   - We used 5e-5 vs paper's 1e-6
   - Did higher LR compensate for fewer epochs?

5. **Does 4-bit quantization affect reasoning quality?**
   - Paper used full precision on A100s
   - We used 4-bit quantization for memory efficiency

6. **What's the training curve shape?**
   - If we continue to 5 epochs, does performance plateau?
   - When does overfitting start (if at all)?

## Next Actions

1. **Immediate** (today): Run WildGuard test evaluation
2. **Tomorrow**: Analyze results and make training decision
3. **This week**: Either publish 1-epoch model OR continue training
4. **Next week**: Begin ablation studies for publication

## Notes for Publication

### Contributions if Successful
- Validated GuardReasoner approach on smaller model (3B vs 8B)
- Demonstrated sample efficiency with 1 epoch (vs paper's 5)
- Memory-efficient training with 4-bit quantization
- Open-sourced LoRA adapter for community use

### Limitations to Acknowledge
- Smaller base model (3B vs 8B)
- Quantized training (4-bit vs full precision)
- Different hyperparameters (LR 5e-5 vs 1e-6)
- Fewer epochs if 1-epoch model is sufficient

### Alternative Outcomes
- If 1 epoch insufficient: Full 5-epoch replication
- If 3B insufficient: Scale to 8B model
- If quantization problematic: Try full precision training

## Model Artifacts

**Current Release**:
- HuggingFace: vincentoh/guardreasoner-llama3.2-3b-lora-1epoch
- Local: ~/wizard101/experiments/guardreasoner/models/exp_18_rsft_lora/
- Checkpoint: ~/Downloads/guardreasoner_models/exp_18_rsft_lora/

**Future Releases** (if needed):
- vincentoh/guardreasoner-llama3.2-3b-lora-5epoch (full training)
- vincentoh/guardreasoner-llama3.2-8b-lora (larger model)

---

**Last Updated**: 2025-11-18
**Experiment ID**: 18 (R-SFT 1 epoch)
**Status**: Awaiting Phase 1 evaluation
