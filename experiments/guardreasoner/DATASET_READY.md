# GuardReasoner Dataset - Ready to Use! ✅

**Date**: 2025-11-18
**Status**: Downloaded and extracted successfully

---

## 📊 Dataset Summary

**Total samples**: 127,544 with reasoning traces
**Location**: `~/development/wizard101/experiments/guardreasoner/guardreasoner_data/`

### Splits Available

| Split | Samples | File | Size |
|-------|---------|------|------|
| WildGuardTrainR | 86,759 | `WildGuardTrainR.json` | 263 MB |
| BeaverTailsTrainR | 27,186 | `BeaverTailsTrainR.json` | 58 MB |
| AegisTrainR | 10,798 | `AegisTrainR.json` | 26 MB |
| ToxicChatTrainR | 2,801 | `ToxicChatTrainR.json` | 5.5 MB |
| **ALL COMBINED** | **127,544** | `all_combined.json` | **357 MB** |

---

## 🚀 Quick Start

### Load All Data (127K samples)
```python
import json

# Load complete dataset
with open('guardreasoner_data/all_combined.json') as f:
    data = json.load(f)

print(f"Total samples: {len(data):,}")  # 127,544

# Each sample has:
sample = data[0]
print(sample.keys())
# dict_keys(['instruction', 'input', 'output',
#            'prompt_harm_label', 'response_refusal_label',
#            'response_harm_label', 'source_split'])
```

### Load Specific Split (e.g., WildGuard only)
```python
import json

# Load just WildGuard (87K samples)
with open('guardreasoner_data/WildGuardTrainR.json') as f:
    wildguard = json.load(f)

print(f"WildGuard samples: {len(wildguard):,}")  # 86,759
```

---

## 📝 Data Format

Each sample contains:

```json
{
  "instruction": "You are a classifier for analyzing interactions between humans and AI...",
  "input": "Human user:\n{prompt}\n\nAI assistant:\n{response}\n\n",
  "output": "Answers:\nRequest: harmful/unharmful\nCompletion: refusal/compliance\nResponse: harmful/unharmful\n",
  "prompt_harm_label": "harmful",
  "response_refusal_label": "refusal",
  "response_harm_label": "unharmful",
  "source_split": "WildGuardTrainR"
}
```

### Fields Explained

- **`instruction`**: System prompt with 3-task classification instructions
- **`input`**: "Human user:\n{prompt}\n\nAI assistant:\n{response}\n\n"
- **`output`**: Full reasoning trace + 3 labels
- **`prompt_harm_label`**: "harmful" or "unharmful" (Task 1)
- **`response_refusal_label`**: "refusal" or "compliance" (Task 2)
- **`response_harm_label`**: "harmful" or "unharmful" (Task 3)
- **`source_split`**: Which dataset it came from (only in `all_combined.json`)

---

## 🔄 Comparison: Your Old vs New Dataset

| Feature | Your Old (Exp 18) | GuardReasoner (New) | Improvement |
|---------|-------------------|---------------------|-------------|
| **Samples** | 11,396 | 127,544 | **11× more** |
| **Sources** | 2 (harmful/harmless) | 4 (diverse benchmarks) | **2× diversity** |
| **Tasks** | 1 (binary harmful/safe) | 3 (prompt/refusal/response) | **3× richer** |
| **Response included** | ❌ No | ✅ Yes | **Critical fix** |
| **Reasoning quality** | GPT | GPT-4o | **Higher quality** |
| **Your result** | 59% (1 epoch) | Expected: 75-80% | **+16-21%** |

---

## 🎯 Training Plan: Experiment 19

### Option A: Full Dataset (Recommended for Paper Comparison)
```python
# Use all 127K samples
dataset_path = "guardreasoner_data/all_combined.json"
num_samples = 127_544

# Training estimates:
# - Epoch 1: ~90 hours (~3.75 days)
# - 3 epochs: ~270 hours (~11 days)
# - Expected accuracy: 75-80%
```

### Option B: WildGuard Only (Faster Iteration)
```python
# Use just WildGuard (87K samples)
dataset_path = "guardreasoner_data/WildGuardTrainR.json"
num_samples = 86_759

# Training estimates:
# - Epoch 1: ~60 hours (~2.5 days)
# - 3 epochs: ~180 hours (~7.5 days)
# - Expected accuracy: 70-75%
```

### Option C: Staged Training (Most Practical)
```python
# Stage 1: Continue Exp 18 (11K, 2 more epochs)
#   - Complete for comparison
#   - Estimate: 65-70% accuracy

# Stage 2: Full retrain with 127K
#   - Clean comparison
#   - Estimate: 75-80% accuracy
```

---

## 📋 Next Steps

### 1. Verify Dataset ✅
```bash
cd ~/development/wizard101/experiments/guardreasoner

# Check file exists
ls -lh guardreasoner_data/all_combined.json

# Verify sample count
python3 -c "import json; d=json.load(open('guardreasoner_data/all_combined.json')); print(f'{len(d):,} samples')"
```

### 2. Examine Sample
```python
import json

# Load and examine first sample
with open('guardreasoner_data/sample.json') as f:
    samples = json.load(f)

# See examples from each split
for split_name, split_samples in samples.items():
    print(f"\n{split_name}:")
    print(json.dumps(split_samples[0], indent=2)[:500])
```

### 3. Create Training Script (Experiment 19)
```bash
# Copy your Exp 18 training script as template
cp scripts/experiment_18_train_unsloth.py scripts/experiment_19_train_full_dataset.py

# Update dataset path:
# OLD: dataset_path = "data/guardreasoner_train_chatml.json"
# NEW: dataset_path = "guardreasoner_data/all_combined.json"
```

### 4. Start Training
```bash
# On nigel.birs.ca (or wherever you're training)
ssh user@server
cd ~/wizard101/experiments/guardreasoner

# Start in screen session
screen -S exp19
python3 scripts/experiment_19_train_full_dataset.py

# Detach: Ctrl+A then D
# Reattach: screen -r exp19
```

---

## 📊 Expected Training Timeline

### Hardware Assumptions
- GPU: Single GPU (24GB)
- Method: 4-bit LoRA
- Batch size: 2
- Gradient accumulation: 64
- Effective batch size: 128

### Time Estimates

**Full Dataset (127K samples, 3 epochs)**:
```
Epoch 1: ~90 hours (~3.75 days)
Epoch 2: ~90 hours (~3.75 days)
Epoch 3: ~90 hours (~3.75 days)
────────────────────────────────
Total:   ~270 hours (~11 days)

Expected result: 75-80% accuracy
```

**WildGuard Only (87K samples, 3 epochs)**:
```
Epoch 1: ~60 hours (~2.5 days)
Epoch 2: ~60 hours (~2.5 days)
Epoch 3: ~60 hours (~2.5 days)
────────────────────────────────
Total:   ~180 hours (~7.5 days)

Expected result: 70-75% accuracy
```

---

## 🎯 Success Criteria

### Minimum (Match Your Constraints)
- ✅ Accuracy ≥ 70%
- ✅ Prompt Harmful F1 ≥ 0.75
- ✅ Response Refusal F1 ≥ 0.70
- ✅ Response Harmful F1 ≥ 0.70

### Target (Close to Paper)
- 🎯 Accuracy ≥ 75%
- 🎯 Prompt Harmful F1 ≥ 0.80
- 🎯 Response Refusal F1 ≥ 0.75
- 🎯 Response Harmful F1 ≥ 0.75

### Stretch (Match Paper's 8B Model)
- 🏆 Accuracy ≥ 80%
- 🏆 All F1 scores ≥ 0.80
- 🏆 Within 5% of paper's 84%

---

## 📖 Documentation

- **README**: `guardreasoner_data/README.md` - Basic usage
- **Stats**: `guardreasoner_data/DATASET_INFO.json` - Dataset metadata
- **Samples**: `guardreasoner_data/sample.json` - Example from each split
- **Comparison**: `../GUARDREASONER_COMPARISON.md` - Full methodology analysis
- **Summary**: `../COMPARISON_SUMMARY.md` - Quick reference

---

## ✅ Verification Checklist

- [x] ✅ Dataset downloaded (127,544 samples)
- [x] ✅ Extracted to `guardreasoner_data/`
- [x] ✅ All 8 files present (4 splits + combined + metadata)
- [x] ✅ JSON files load correctly
- [ ] ⏳ Examine sample data format
- [ ] ⏳ Update training script for Exp 19
- [ ] ⏳ Start training on nigel
- [ ] ⏳ Monitor first 1000 steps
- [ ] ⏳ Evaluate after 3 epochs
- [ ] ⏳ Compare to paper's results

---

## 🚀 You're Ready!

Everything is set up to train with GuardReasoner's exact dataset. This should take you from **59% → 75-80% accuracy**!

**Next action**: Create `experiment_19_train_full_dataset.py` and start training!
