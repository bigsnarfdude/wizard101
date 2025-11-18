# GuardReasoner Experiment Tracker

**Last Updated:** 2025-11-17
**Status:** Planning Phase → Ready for Phase 1

---

## Quick Status Dashboard

| Phase | Experiments | Status | Completion | Best Result |
|-------|-------------|--------|------------|-------------|
| **Phase 1** | 20-25 (Zero-training) | 🔜 Queued | 0/6 | TBD |
| **Phase 2** | 26-30 (Ensemble) | ⏸️ Pending | 0/5 | TBD |
| **Phase 3** | 31-35 (Data Gen) | ⏸️ Pending | 0/5 | TBD |
| **Phase 4** | 36-40 (Fine-tuning) | ⏸️ Pending | 0/5 | TBD |
| **Phase 5** | 41-45 (Analysis) | ⏸️ Pending | 0/5 | TBD |

**Baseline (Exp 12):** 57.5% accuracy on WildGuard
**Target:** 75-80% accuracy (GuardReasoner level)

---

## Phase 1: Zero-Training Baseline (Experiments 20-25)

### Experiment 20: Single-Step Reasoning Baseline
- **Status:** 🔜 Queued
- **Script:** `experiment_20_single_step_reasoning.py`
- **Prompt:** experiments/guardreasoner/prompts/single_step_reasoning.txt
- **Model:** gpt-oss:20b
- **Dataset:** WildGuard full (1554 samples)
- **Expected:** 59-62% (+2-5% over baseline)
- **Runtime:** ~4 hours
- **Started:** TBD
- **Completed:** TBD
- **Result:** TBD

### Experiment 21: Multi-Step Reasoning (GuardReasoner-style)
- **Status:** ⏸️ Pending
- **Script:** `experiment_21_multistep_reasoning.py`
- **Prompt:** experiments/guardreasoner/prompts/multistep_reasoning.txt
- **Model:** gpt-oss:20b
- **Dataset:** WildGuard full (1554 samples)
- **Expected:** 62-65% (+5-8% over baseline)
- **Runtime:** ~4 hours
- **Depends on:** Exp 20
- **Result:** TBD

### Experiment 22: Chain-of-Thought with Examples
- **Status:** ⏸️ Pending
- **Script:** `experiment_22_cot_examples.py`
- **Prompt:** experiments/guardreasoner/prompts/cot_with_examples.txt
- **Model:** gpt-oss:20b
- **Dataset:** WildGuard full (1554 samples)
- **Expected:** 64-67% (+7-10% over baseline)
- **Runtime:** ~4.5 hours (longer prompts)
- **Depends on:** Exp 21
- **Result:** TBD

### Experiment 23: Safeguard + Multi-Step Reasoning
- **Status:** ⏸️ Pending
- **Script:** `experiment_23_safeguard_reasoning.py`
- **Prompt:** Same as Exp 21
- **Model:** gpt-oss-safeguard:latest
- **Dataset:** WildGuard full (1554 samples)
- **Expected:** 62-67% (similar to Exp 21, +1-2% from safety tuning)
- **Runtime:** ~4 hours
- **Depends on:** Exp 21
- **Result:** TBD

### Experiment 24: Reasoning + Minimal Policies
- **Status:** ⏸️ Pending
- **Script:** `experiment_24_reasoning_minimal.py`
- **Prompt:** Reasoning + minimal policies (100-150 tokens)
- **Model:** gpt-oss:20b
- **Dataset:** WildGuard full (1554 samples)
- **Expected:** 62-65% (test if reasoning > policy verbosity)
- **Runtime:** ~3.5 hours
- **Depends on:** Exp 21
- **Result:** TBD

### Experiment 25: Reasoning + Self-Consistency
- **Status:** ⏸️ Pending
- **Script:** `experiment_25_self_consistency.py`
- **Prompt:** Exp 21 prompt with temperature=0.7
- **Model:** gpt-oss:20b (3 samples per prompt)
- **Dataset:** WildGuard full (1554 samples)
- **Expected:** 65-68% (+3-5% from ensemble)
- **Runtime:** ~12 hours (3x inference)
- **Depends on:** Exp 21
- **Result:** TBD

---

## Phase 2: Ensemble & Hard Sample Mining (Experiments 26-30)

### Experiment 26: 3-Model Ensemble Baseline
- **Status:** ⏸️ Pending
- **Models:** gpt-oss:20b, gpt-oss-safeguard, gpt-oss:20b (temp=0.8)
- **Expected:** 60-63% (+2-4% from diversity)
- **Result:** TBD

### Experiment 27: Hard Sample Identification
- **Status:** ⏸️ Pending
- **Method:** Disagreement mining across 3 models
- **Output:** hard_samples_wildguard.json (~310 samples)
- **Result:** TBD

### Experiment 28: Focused Hard Sample Evaluation
- **Status:** ⏸️ Pending
- **Dataset:** 310 hard samples from Exp 27
- **Expected:** 50-55% (lower on hard samples)
- **Result:** TBD

### Experiment 29: Ensemble + Reasoning
- **Status:** ⏸️ Pending
- **Method:** 3-model ensemble WITH reasoning prompts
- **Expected:** 64-67% (best zero-training result)
- **Result:** TBD

### Experiment 30: Weighted Ensemble
- **Status:** ⏸️ Pending
- **Method:** Confidence-weighted voting
- **Expected:** 65-68% (+1-2% over simple voting)
- **Result:** TBD

---

## Phase 3: Data Generation (Experiments 31-35)
**Status:** ⏸️ Pending Phase 1 completion

---

## Phase 4: Fine-Tuning (Experiments 36-40)
**Status:** ⏸️ Pending Phase 3 completion

---

## Phase 5: Analysis (Experiments 41-45)
**Status:** ⏸️ Pending Phase 4 completion

---

## Results Summary

### Best Results by Category

#### Zero-Training (no fine-tuning)
- **Best overall:** TBD (Exp 20-25)
- **Best reasoning:** TBD
- **Best ensemble:** TBD

#### With Fine-Tuning
- **Best R-SFT:** TBD (Exp 36-40)
- **Best overall:** TBD

### Accuracy Progression
```
Baseline (Exp 12):           57.5%
Best Phase 1 (reasoning):    TBD% (target: 67%)
Best Phase 2 (ensemble):     TBD% (target: 68%)
Best Phase 4 (R-SFT):        TBD% (target: 77%)
GuardReasoner benchmark:     84.09%
```

---

## Timeline

### Completed
- [x] 2025-11-17: Planning document created
- [x] 2025-11-17: Folder structure created
- [x] 2025-11-17: Experiment tracker initialized

### Upcoming
- [ ] 2025-11-18: Exp 16-17 complete (old experiments)
- [ ] 2025-11-18: Implement Exp 20 (single-step reasoning)
- [ ] 2025-11-19: Run Exp 20-22 (reasoning baselines)
- [ ] 2025-11-20: Run Exp 23-25 (reasoning variations)
- [ ] 2025-11-21: Analyze Phase 1 results
- [ ] 2025-11-22: Start Phase 2 (ensemble)

---

## Notes & Insights

### 2025-11-17: Planning
- Baseline experiments (01-17) show 57% plateau regardless of configuration
- GuardReasoner paper suggests reasoning traces key to breaking through
- Zero-training approach (Phase 1) most cost-effective starting point
- Need to validate prompt engineering before committing to fine-tuning

---

## Open Questions

1. **GPU access for fine-tuning?** (Phase 4)
2. **GPT-4 API budget?** (~$100 for reasoning trace generation)
3. **Priority: Speed vs Completeness?** (Can skip Phase 5 initially)

---

**Next Action:** Implement Experiment 20 script
**Blocking:** Experiments 16-17 completion (ETA: 7 hours)
