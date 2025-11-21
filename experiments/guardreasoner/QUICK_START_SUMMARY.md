# GuardReasoner Quick Start Summary

## What We Have Now (2025-11-18)

### ✅ Completed: SFT Training (Experiment 18)
- **Model**: Llama-3.2-3B-Instruct with LoRA adapter
- **Training**: 1 epoch R-SFT on 11,396 samples (8 hours)
- **Location**: `~/wizard101/experiments/guardreasoner/models/exp_18_rsft_lora/`
- **HuggingFace**: vincentoh/guardreasoner-llama3.2-3b-lora-1epoch
- **Purpose**: Foundation model with reasoning traces

### 🔄 In Progress: Quick Evaluation
- **Script**: `evaluate_exp_18_quick.py`
- **Dataset**: 100 samples from combined_test.json
- **Runtime**: ~10 minutes (6 seconds/sample)
- **Status**: 25% complete (as of 14:18 UTC)
- **Checks**:
  - Model loads correctly ✅
  - Adapter applies successfully ✅
  - Generates reasoning traces ✅

### 📋 Next: RL Training Pipeline

## Dataset Information

### Training Data (`guardreasoner_train_chatml.json`)
- **Size**: 11,396 samples
- **Format**: ChatML with reasoning traces
- **Sources**:
  - Harmful Behaviors: Adversarial safety prompts
  - Harmless Alpaca: Benign instructions
- **Structure**:
```json
{
  "conversations": [
    {"from": "human", "value": "[prompt]"},
    {"from": "gpt", "value": "[reasoning]\nLabel: harmful/safe"}
  ]
}
```

### Test Data (`combined_test.json`)
- **Size**: 200 samples
- **Used for**: Quick evaluation (100 samples randomly sampled)
- **Format**:
```json
{
  "text": "prompt text here",
  "label": "harmless" or "harmful",
  "category": "harmless" or "harmful"
}
```

**Note**: Script normalizes `"harmless"` → `"safe"` for consistency

## Model Architecture

### How It Works
1. **Base**: Llama-3.2-3B-Instruct (4-bit quantized)
2. **Adapter**: LoRA with rank=16
   - Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
   - Alpha: 16, Dropout: 0
3. **Inference**: Base model + LoRA weights
4. **Output**: Reasoning trace + safety label

### Evaluation Process
```python
# 1. Load base model (4-bit)
base = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-3B-Instruct", load_in_4bit=True)

# 2. Load LoRA adapter
model = PeftModel.from_pretrained(base, "exp_18_rsft_lora")

# 3. Create prompt (ChatML format)
prompt = "<|im_start|>user\nAnalyze this content for safety:\n...<|im_end|>\n<|im_start|>assistant\n..."

# 4. Generate reasoning + label
output = model.generate(prompt, max_tokens=512, temperature=0.7)

# 5. Parse output
reasoning, label = parse_response(output)
```

## Why RL Training Next?

### The Paper's Finding (IMPORTANT)
**GRPO (Group Relative Policy Optimization) FAILED** ❌
- Tried on difficult samples starting from SFT model
- Result: -1.1% average F1 score
- Problem: "Difficulty improving safety guards distilled from strong teacher"

### Our Different Approach
**DPO on Reasoning Quality** (not just accuracy) ✅

**Key Innovation**: Optimize reasoning traces, not just labels

#### Traditional RL (Paper's Mistake)
```
Reward = 1 if correct_label else 0
Problem: No room to improve when SFT already matches teacher accuracy
```

#### Our RL Approach
```
Reward = 0.3 * label_correctness +
         0.3 * reasoning_coherence +
         0.2 * reasoning_relevance +
         0.2 * reasoning_specificity

Advantage: Can improve reasoning quality even when accuracy plateaus
```

### RL Pipeline (Toy → Full)

**Phase 1: Toy Experiment** (1-2 days)
```
1. Generate 1k samples × 4 completions = 4k generations
2. Score each: reasoning quality + label correctness
3. Create preference pairs: (good reasoning, bad reasoning)
4. DPO training: 1 epoch (~2 hours)
5. Evaluate: Better reasoning? Similar accuracy?
```

**Phase 2: Full RL** (3-4 days, if toy works)
```
1. Generate 10k samples × 4 completions = 40k generations
2. Score all with automated judge
3. Create 10k preference pairs
4. DPO training: 2-3 epochs (~24 hours)
5. Full evaluation: WildGuard test set (1,554 samples)
```

## Success Criteria

### Toy Experiment (Exp 19)
- ✅ Accuracy: Within ±2% of SFT baseline
- ✅ Reasoning quality: +10% improvement (automated score)
- ✅ Human preference: 60% prefer DPO reasoning

### Full Experiment (Exp 20)
- ✅ Accuracy: 59-61% on WildGuard (baseline: 57.5%)
- ✅ Reasoning quality: +10-15% improvement
- ✅ Human preference: 65% prefer DPO reasoning

## File Locations

### On nigel.birs.ca
```
~/wizard101/experiments/guardreasoner/
├── data/
│   ├── guardreasoner_train_chatml.json (11,396 samples)
│   └── combined_test.json (200 samples)
├── models/
│   └── exp_18_rsft_lora/ (1-epoch SFT model)
├── logs/
│   └── exp_18_quick_eval_v2.log (current evaluation)
├── results/
│   └── exp_18_quick_eval.json (will be created)
└── scripts/
    └── evaluate_exp_18_quick.py
```

### On MacBook (local)
```
~/development/wizard101/experiments/guardreasoner/
├── RL_TRAINING_PLAN.md (RL strategy document)
├── RESEARCH_VALIDATION.md (paper replication plan)
├── EVALUATION_PLAN.md (decision framework)
└── QUICK_START_SUMMARY.md (this file)
```

### On HuggingFace
```
vincentoh/guardreasoner-llama3.2-3b-lora-1epoch
├── adapter_model.safetensors (93MB - LoRA weights)
├── adapter_config.json
├── tokenizer files
└── README.md (model card)
```

## Next Steps

### Immediate (Today)
1. ✅ Wait for quick eval to finish (~10 min remaining)
2. ✅ Check if model generates coherent reasoning
3. ✅ Verify accuracy >50% (go/no-go for RL)

### Tomorrow
1. 📋 Create `create_rl_preferences.py` script
2. 📋 Generate toy preference dataset (1k pairs)
3. 📋 Start toy DPO training

### This Week
1. 📋 Evaluate toy DPO results
2. 📋 Decision: Scale to full RL or iterate?
3. 📋 If good: Create full preference dataset

## Key Files to Create

### For RL Training
- `create_rl_preferences.py` - Generate preference pairs from multiple completions
- `judge_reasoning_quality.py` - Automated scoring of reasoning traces
- `train_exp_19_dpo_toy.py` - Toy DPO experiment (1k pairs)
- `train_exp_20_dpo_full.py` - Full DPO experiment (10k pairs)

### For Evaluation
- `evaluate_exp_19_dpo.py` - Compare SFT vs DPO
- `human_eval_tool.py` - Interface for human preference study
- `analyze_reasoning_improvements.py` - What did RL improve?

## Commands Reference

### Monitor Evaluation
```bash
# Check progress
ssh user@server "tail -f ~/wizard101/experiments/guardreasoner/logs/exp_18_quick_eval_v2.log"

# Check results
ssh user@server "cat ~/wizard101/experiments/guardreasoner/results/exp_18_quick_eval.json"
```

### Upload Scripts
```bash
# From local machine
scp experiments/guardreasoner/script.py user@server:~/wizard101/experiments/guardreasoner/
```

### Run Training
```bash
# On nigel
cd ~/wizard101/experiments/guardreasoner
source venv/bin/activate
python script.py > logs/output.log 2>&1 &
```

## Research Questions

1. **Does 1-epoch SFT work?** (testing now)
2. **Can RL improve reasoning quality?** (next phase)
3. **Is DPO better than GRPO for this task?** (hypothesis: yes)
4. **What's the sweet spot: accuracy vs reasoning quality?**

---

**Last Updated**: 2025-11-18 14:19 UTC
**Status**: Evaluation running (25% complete)
**Next Milestone**: Evaluate reasoning quality when eval finishes
