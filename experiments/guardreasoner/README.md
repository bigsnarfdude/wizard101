# GuardReasoner-Inspired Safety Experiments

**Based on:** Liu et al. "GuardReasoner: Towards Reasoning-based LLM Safeguards" (arXiv:2501.18492)

**Goal:** Improve safety classification from 57% (baseline) → 75-80% (GuardReasoner level) using reasoning-based approaches

---

## Quick Start

### Current Status
- ✅ Planning complete (`PLAN.md`)
- ✅ Folder structure created
- ✅ Experiment tracker initialized (`EXPERIMENT_TRACKER.md`)
- 🔜 Ready to start Phase 1 (Experiments 20-25)

### What's Different from Original Experiments (01-17)?

**Original Approach (Serial Gauntlet):**
- Direct classification: "Does this violate policy X?"
- Result: 57% accuracy plateau across ALL configurations
- Problem: Over-flagging without contextual reasoning

**GuardReasoner Approach:**
- Step-by-step reasoning: Intent → Harm → Policy → Decision
- Expected: 75-80% accuracy with proper reasoning
- Advantage: Explainable + higher precision

---

## Folder Structure

```
guardreasoner/
├── PLAN.md                      # Complete experiment plan (45 experiments)
├── EXPERIMENT_TRACKER.md        # Live status tracking
├── README.md                    # This file
│
├── scripts/                     # Experiment scripts
│   ├── experiment_20_*.py
│   ├── experiment_21_*.py
│   └── ...
│
├── prompts/                     # Reasoning prompt templates
│   ├── single_step_reasoning.txt
│   ├── multistep_reasoning.txt
│   ├── cot_with_examples.txt
│   └── ...
│
├── data/                        # Generated datasets
│   ├── hard_samples_wildguard.json
│   ├── wildguard_reasoning_traces.json
│   └── rsft_training_data.json
│
├── results/                     # Experiment outputs
│   ├── exp_20.log
│   ├── exp_20_results.json
│   └── ...
│
└── models/                      # Fine-tuned models
    ├── gpt-oss-20b-reasoning-v1/
    └── ...
```

---

## Experiment Phases

### Phase 1: Zero-Training Baseline (Exp 20-25) 🔜 NEXT
**Timeline:** 1-2 days
**Goal:** Test reasoning prompts WITHOUT model fine-tuning
**Experiments:**
- Exp 20: Single-step reasoning
- Exp 21: Multi-step reasoning (GuardReasoner-style)
- Exp 22: Chain-of-thought with examples
- Exp 23: Safeguard + reasoning
- Exp 24: Reasoning + minimal policies
- Exp 25: Self-consistency ensemble

**Expected gain:** +10% over baseline (57% → 67%)

### Phase 2: Ensemble & Hard Sample Mining (Exp 26-30)
**Timeline:** 2-3 days
**Goal:** Identify decision boundary cases
**Key experiments:**
- Exp 26: 3-model ensemble voting
- Exp 27: Hard sample identification
- Exp 29: Ensemble + reasoning (best zero-training combo)

**Expected gain:** +11% over baseline (57% → 68%)

### Phase 3: Data Generation (Exp 31-35)
**Timeline:** 3-5 days
**Goal:** Generate reasoning traces for fine-tuning
**Key outputs:**
- 1554 WildGuard samples with GPT-4 reasoning traces
- 500-1000 synthetic hard samples
- ~2000-2500 training samples total

**Cost:** ~$50-100 for GPT-4 API

### Phase 4: Fine-Tuning (Exp 36-40)
**Timeline:** 1 week
**Goal:** R-SFT (reasoning-supervised fine-tuning)
**Key experiments:**
- Exp 36: R-SFT on gpt-oss:20b
- Exp 37: R-SFT on gpt-oss-safeguard
- Exp 40: Full pipeline evaluation

**Expected gain:** +20% over baseline (57% → 77%)
**Requirements:** GPU access (A100/H100)

### Phase 5: Analysis (Exp 41-45)
**Timeline:** 2-3 weeks (ongoing)
**Goal:** Deep dive into failure modes and optimization

---

## Success Metrics

### Tier 1: Zero-Training (Phase 1-2)
- ✅ **+10% gain** from reasoning prompts alone
- ✅ Identify top 20% hard samples
- ✅ Validate reasoning improves explainability

### Tier 2: Fine-Tuning (Phase 3-4)
- ✅ **75-80% accuracy** on WildGuard
- ✅ **+20% gain** over baseline
- ✅ Per-policy F1 > 70% for all policies

### Tier 3: Production-Ready (Phase 5)
- ✅ <100ms inference with >70% accuracy
- ✅ Generalizes to 3+ benchmarks
- ✅ Explainable reasoning (>80% coherent)

---

## Running Experiments

### Prerequisites
```bash
# Ensure you're in wizard101 directory
cd /Users/vincent/development/wizard101

# Ollama running on nigel (check connection)
ssh vincent@nigel.birs.ca "curl -s http://localhost:11434/api/tags | jq '.models[].name'"

# Verify WildGuard dataset exists
ls -lh experiments/wildguard_full_benchmark.json
```

### Run Experiment 20 (First experiment)
```bash
# On nigel (via screen)
ssh vincent@nigel.birs.ca
cd ~/wizard101/experiments/guardreasoner
screen -S guardreasoner
python3 scripts/experiment_20_single_step_reasoning.py > results/exp_20.log 2>&1

# Detach: Ctrl+A then D
# Reattach: screen -r guardreasoner
```

### Monitor Progress
```bash
# From local machine
ssh vincent@nigel.birs.ca "tail -f ~/wizard101/experiments/guardreasoner/results/exp_20.log"

# Check completion
ssh vincent@nigel.birs.ca "grep 'EXPERIMENT 20 RESULTS' ~/wizard101/experiments/guardreasoner/results/exp_20.log"
```

---

## Key Files to Read

1. **PLAN.md** - Complete methodology and all 45 experiments
2. **EXPERIMENT_TRACKER.md** - Live status and results
3. **prompts/** - See reasoning prompt templates
4. **results/** - Check experiment outputs

---

## Comparison to Original GuardReasoner

### Similarities
- ✅ Two-stage approach: Zero-training → Fine-tuning
- ✅ Multi-step reasoning framework
- ✅ Hard sample mining via ensemble disagreement
- ✅ R-SFT with reasoning traces
- ✅ DPO on hard samples (Phase 4)

### Differences
- ⚠️ **Models:** gpt-oss vs LLaMA (different base models)
- ⚠️ **Dataset:** WildGuard only vs 4 datasets (127K samples)
- ⚠️ **Reasoning generation:** GPT-4 vs GPT-4o
- ⚠️ **Scale:** 1554 samples vs 127K samples
- ⚠️ **Training:** LoRA vs full fine-tuning (resource constraint)

### Expected Performance
- **GuardReasoner (8B):** 84.09% F1 on their benchmarks
- **Our target (20B):** 75-80% on WildGuard
- **Why lower?** Smaller training dataset, different base model

---

## Next Steps

1. ✅ Wait for Exp 16-17 to complete (old experiments)
2. ⬜ Implement Experiment 20 script
3. ⬜ Create reasoning prompt templates
4. ⬜ Run Phase 1 experiments (20-25)
5. ⬜ Analyze results and decide on Phase 2

---

## Resources & References

### Papers
- **GuardReasoner:** Liu et al. arXiv:2501.18492 (2025)
- **WildGuard:** Allen AI - Adversarial safety benchmark

### Code
- **GuardReasoner repo:** github.com/yueliu1999/GuardReasoner
- **Our repo:** /Users/vincent/development/wizard101/

### Datasets
- **WildGuard full:** experiments/wildguard_full_benchmark.json (1554 samples)
- **Original experiments:** experiments/exp_01.log - exp_17.log

---

## Questions?

- Check **PLAN.md** for detailed methodology
- Check **EXPERIMENT_TRACKER.md** for live status
- Review **Phase 1 experiments** in `scripts/` folder

**Status:** Ready to implement Phase 1
**Next action:** Create Experiment 20 script
