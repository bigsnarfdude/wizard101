# Speed Optimization for GuardReasoner Training

## Current Status (Exp 18)
- **Model**: Llama-3.2-3B-Instruct (Unsloth)
- **Dataset**: 127k samples
- **Time**: 8 hours for 1 epoch
- **Hardware**: RTX 4070 Ti Super (16GB VRAM)

## Gemini's Speed-Up Suggestions Applied

### Already Implemented ✅
1. **4-bit Quantization**: `load_in_4bit=True`
2. **LoRA Fine-Tuning**: Not full parameter (r=16, α=16)
3. **Gradient Accumulation**: Steps=64 (effective batch=128)
4. **Small Batch Size**: 2 per device
5. **Unsloth**: Already 2-5x faster than transformers

### New Optimizations 🚀

#### 1. Subset Training (Prototype Fast)
```python
"max_samples": 12000,  # Only 10% of dataset
"num_train_epochs": 1,
```
**Expected**: 30-40 minutes vs 8 hours
**Use case**: Quick validation, hyperparameter tuning

#### 2. Torch Compile
```python
os.environ['TORCH_COMPILE'] = '1'
```
**Expected**: 10-20% speedup
**Note**: PyTorch 2.0+ required

#### 3. 8-bit Optimizer
```python
optim="adamw_8bit",  # Instead of adamw_torch
```
**Expected**: Lower memory, slightly faster

#### 4. More Data Workers
```python
"dataloader_num_workers": 4,  # Parallel data loading
```
**Expected**: 5-10% speedup if CPU has cores

#### 5. More Frequent Logging
```python
"logging_steps": 5,  # Every 5 steps instead of 10
```
**Use case**: Catch issues early in fast runs

## Training Scripts Comparison

### Original (Slow but Complete)
**File**: `scripts/experiment_18_train_unsloth.py`
- Full 127k samples
- 1 epoch = 8 hours
- Save every 1000 steps

### Fast Prototype (NEW)
**File**: `train_exp_18_fast.py`
- 12k samples (10%)
- 1 epoch = 30-40 minutes
- All speed optimizations enabled

### When to Use Each

**Use Fast Script When:**
- Testing code changes
- Tuning hyperparameters
- Validating data format
- Quick experiments

**Use Original Script When:**
- Final training run
- Need full model quality
- Publishing results

## Expected Timeline

### Fast Prototype Run
```bash
# Quick validation (30-40 min)
python train_exp_18_fast.py
```
- Samples: 12k (10%)
- Time: 30-40 minutes
- Quality: ~90% of full model
- Use: Validate pipeline works

### Full Training Run
```bash
# Full quality (8 hours)
python scripts/experiment_18_train_unsloth.py
```
- Samples: 127k (100%)
- Time: 8 hours
- Quality: Maximum
- Use: Final model

### Scaling Options

| Dataset Size | Time (1 epoch) | Quality | Use Case |
|--------------|---------------|---------|----------|
| 12k (10%) | 30-40 min | 90% | Fast prototype |
| 25k (20%) | 1-1.5 hours | 93% | Quick validation |
| 64k (50%) | 4 hours | 97% | Good enough |
| 127k (100%) | 8 hours | 100% | Publication quality |

## HS-DPO Speed Optimization

### Current HS-DPO (Toy)
- 63 preference pairs
- 2 minutes training
- Already very fast!

### Full HS-DPO (Projected)
- ~7,000 preference pairs
- ~2-3 hours training
- Could be faster with:
  - Smaller beta (faster convergence)
  - Subset of hard samples (top 2k hardest)

## Recommended Workflow

### Phase 1: Fast Validation (2 hours total)
```bash
# 1. Train R-SFT on 12k samples (40 min)
python train_exp_18_fast.py

# 2. Mine hard samples from 1k test samples (10 min)
python mine_hard_samples.py --max-samples 1000

# 3. Create DPO pairs (instant)
python create_dpo_dataset.py

# 4. Train HS-DPO (5 min)
python train_exp_19_hsdpo_toy.py

# 5. Evaluate (10 min)
python evaluate_exp_19_hsdpo.py
```

### Phase 2: Full Quality (12 hours total)
```bash
# 1. Train R-SFT on full dataset (8 hours)
python scripts/experiment_18_train_unsloth.py --num_train_epochs 1

# 2. Mine hard samples from full training set (3 hours)
python mine_hard_samples.py  # All 127k samples

# 3. Train HS-DPO on ~7k pairs (2 hours)
python train_exp_19_hsdpo.py --epochs 2

# 4. Evaluate (30 min)
python evaluate_exp_19_hsdpo.py
```

## Cost Analysis

### Single RTX 4070 Ti Super
- **Fast prototype**: $0 (you own it)
- **Full training**: $0 (you own it)
- **Timeline**: 12 hours total

### Cloud GPU (if needed)
- **A100 spot**: ~$1.50/hour
- **Fast prototype**: $3 (2 hours)
- **Full training**: $18 (12 hours)

## Next Steps

1. ✅ Create fast training script
2. 📋 Run fast prototype (40 min)
3. 📋 Verify model outputs reasoning
4. 📋 If good → scale to full dataset
5. 📋 If issues → iterate with fast script
