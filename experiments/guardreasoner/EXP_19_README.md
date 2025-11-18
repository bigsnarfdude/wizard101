# Experiment 19: Full GuardReasoner Dataset Training

**Status**: Ready to run when GPU available
**Created**: 2025-11-18

---

## Overview

Train LLaMA-3.2-3B-Instruct with the **full GuardReasonerTrain dataset** (127K samples) to validate paper's methodology.

### Key Improvements Over Exp 18

| Metric | Exp 18 (Current) | Exp 19 (This) | Improvement |
|--------|------------------|---------------|-------------|
| **Dataset size** | 11,396 samples | 127,544 samples | **11× more** |
| **Tasks** | 1 (binary) | 3 (prompt/refusal/response) | **3× richer** |
| **Response input** | ❌ No | ✅ Yes | **Critical fix** |
| **Reasoning quality** | GPT | GPT-4o | **Higher quality** |
| **Expected accuracy** | 59% (1 epoch) | **75-80%** (3 epochs) | **+16-21%** |

---

## Quick Start

### Prerequisites
```bash
# Ensure GPU is free
nvidia-smi

# Check dependencies
pip install transformers datasets peft trl bitsandbytes accelerate
```

### Run Training
```bash
cd ~/development/wizard101/experiments/guardreasoner

# Start in screen session (will run for ~11 days)
screen -S exp19
python3 scripts/experiment_19_train_full_dataset.py

# Detach: Ctrl+A then D
# Reattach: screen -r exp19

# Monitor logs
tail -f guardreasoner-llama3.2-3b-lora-full/logs/*/events.out.tfevents.*
```

---

## Configuration

### Dataset
- **Source**: HuggingFace `yueliu1999/GuardReasonerTrain`
- **Splits used**: All 4 (WildGuard, Aegis, BeaverTails, ToxicChat)
- **Total samples**: 127,544 with reasoning traces
- **Format**: 3-task classification (prompt/refusal/response)

### Model
- **Base**: `meta-llama/Llama-3.2-3B-Instruct`
- **Quantization**: 4-bit (QLoRA)
- **LoRA config**: r=16, alpha=16, dropout=0.05
- **Target modules**: All attention + MLP layers

### Training
- **Epochs**: 3 (paper used 5, we tested with 3 in Exp 18)
- **Batch size**: 2 per device, 64 grad accum = **128 effective**
- **Learning rate**: 5e-5 (matched Exp 18)
- **Max seq length**: Auto-detected from data (likely 2048)
- **Optimizer**: Paged AdamW 32-bit
- **Precision**: BFloat16

---

## Timeline Estimates

### Training Time (128K samples, 3 epochs)
```
Hardware: Single GPU (24GB assumed)
Steps per epoch: ~1,000
Time per step: ~5.4 minutes

Epoch 1:  ~90 hours (~3.75 days)
Epoch 2:  ~90 hours (~3.75 days)
Epoch 3:  ~90 hours (~3.75 days)
────────────────────────────────────
Total:   ~270 hours (~11 days)
```

### Monitoring Checkpoints
- **Hour 1**: Verify training starts without errors
- **Hour 12**: Check first eval results (~500 steps)
- **Day 4**: Epoch 1 complete (should see improvement)
- **Day 8**: Epoch 2 complete (should plateau)
- **Day 11**: Epoch 3 complete (final model)

---

## Expected Results

### Performance Targets

**Minimum (Validate methodology works)**:
- Overall accuracy: ≥70%
- Prompt harmful F1: ≥0.75
- Response refusal F1: ≥0.70
- Response harmful F1: ≥0.70

**Target (Match constraints)**:
- Overall accuracy: **75-80%**
- Prompt harmful F1: 0.80-0.85
- Response refusal F1: 0.75-0.80
- Response harmful F1: 0.75-0.80

**Paper (8B model, for reference)**:
- Overall accuracy: ~84%
- All F1 scores: 0.80-0.87

### Gap Analysis
Expected gap: 4-9% below paper
- **Reason**: 3B model vs paper's 8B
- **Acceptable**: Validates methodology with smaller model

---

## Comparison to Paper

| Aspect | Paper | Exp 19 (Ours) | Match? |
|--------|-------|---------------|--------|
| **Dataset** | 127K samples | 127K samples | ✅ Same |
| **Reasoning traces** | GPT-4o | GPT-4o | ✅ Same |
| **3-task format** | Yes | Yes | ✅ Same |
| **Model size** | 8B | 3B | ⚠️ Smaller |
| **Training method** | Full FT | LoRA | ⚠️ Different |
| **Epochs** | 5 | 3 | ⚠️ Fewer |
| **Precision** | FP16/BF16 | 4-bit + BF16 | ⚠️ Lower |

**Conclusion**: Methodology matches, resource constraints differ.

---

## Files Generated

### During Training
```
guardreasoner-llama3.2-3b-lora-full/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights
├── training_info.json           # Experiment metadata
├── logs/                        # TensorBoard logs
│   └── events.out.tfevents.*
├── checkpoint-1000/             # Epoch 1
├── checkpoint-2000/             # Epoch 2
└── checkpoint-3000/             # Epoch 3 (final)
```

### After Training
```
results/
├── exp_19_training.log          # Full training log
├── exp_19_eval_results.json     # Evaluation metrics
└── exp_19_comparison.md         # Comparison to Exp 18
```

---

## Monitoring

### Check Progress
```bash
# View training log
tail -100 guardreasoner-llama3.2-3b-lora-full/logs/*/events.out.tfevents.*

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor loss
grep "loss" guardreasoner-llama3.2-3b-lora-full/logs/*.log | tail -20
```

### Warning Signs
- **Loss not decreasing**: Check learning rate (should be 5e-5)
- **OOM errors**: Reduce batch size (2→1) or seq length (2048→1536)
- **NaN loss**: Check for corrupted samples or LR too high

---

## After Training

### 1. Evaluate on Test Sets
```bash
# Run evaluation
python3 scripts/evaluate_exp_19.py

# Expected outputs:
# - Overall accuracy: 75-80%
# - Per-task F1 scores
# - Comparison to Exp 18 (59% baseline)
```

### 2. Compare Results
```bash
# Generate comparison report
python3 scripts/compare_exp_18_vs_19.py

# Should show:
# - Exp 18 (11K samples): 59%
# - Exp 19 (128K samples): 75-80%
# - Improvement: +16-21%
```

### 3. Next Steps (If successful)
- **Option A**: Train for 5 epochs (match paper)
- **Option B**: Implement HS-DPO (Stage 2)
- **Option C**: Scale to 8B model
- **Option D**: Evaluate on 13 benchmarks (match paper)

---

## Troubleshooting

### Issue: Dataset format error
**Error**: `KeyError: 'conversations'`
**Fix**: Script already handles this correctly with instruction/input/output format

### Issue: OOM (Out of Memory)
**Solutions**:
1. Reduce batch size: `per_device_train_batch_size=1`
2. Reduce seq length: `max_seq_length=1536`
3. Disable flash attention (comment out line)

### Issue: Slow training
**Check**:
1. GPU utilization: Should be >90%
2. Gradient accumulation: Should be 64 steps
3. Mixed precision: Should be BF16

### Issue: Loss plateaus early
**Solutions**:
1. Increase learning rate: 5e-5 → 1e-4
2. Train longer: 3 epochs → 5 epochs
3. Reduce weight decay: 0.01 → 0.001

---

## References

- **Paper**: Liu et al. "GuardReasoner" (arXiv:2501.18492)
- **Dataset**: https://huggingface.co/datasets/yueliu1999/GuardReasonerTrain
- **Model**: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- **Your Exp 18**: `EXP_18_RESULTS.md` (59% baseline)
- **Comparison**: `GUARDREASONER_COMPARISON.md` (full analysis)

---

## Status Updates

### 2025-11-18: Created
- ✅ Script written and saved
- ✅ Dataset downloaded (127K samples)
- ✅ Configuration validated
- ⏳ Waiting for GPU availability

### When Training Starts:
Update this section with:
- Start date/time
- GPU model
- Estimated completion date
- Any issues encountered

---

**Ready to run!** Just start the script when GPU is free. 🚀
