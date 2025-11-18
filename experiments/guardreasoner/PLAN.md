# GuardReasoner-Inspired Experiments Plan

**Based on:** "GuardReasoner: Towards Reasoning-based LLM Safeguards" (arXiv:2501.18492)
**Goal:** Validate reasoning-based safety classification on WildGuard benchmark
**Timeline:** Phased approach starting with zero-training experiments

---

## Background: Current Results (Experiments 01-17)

### Key Findings from Serial Gauntlet Approach
- **Accuracy plateau**: All configurations achieve ~56-57% on full WildGuard (1554 samples)
- **Policy length irrelevant**: Minimal/medium/verbose policies show no significant difference
- **Model choice irrelevant**: Baseline vs Safeguard models perform identically
- **Biased sampling issue**: 300-sample subset gave artificially low scores (23-36%)

### Root Cause Analysis
**Over-flagging problem** - Models flag content without contextual reasoning:
```
Example from exp_16.log:
[44] Predicted: {'hate_speech', 'harassment', 'illegal', 'violence', 'sexual_minors'}
     True:      SAFE ✗
```

Models see keywords → flag everything → low precision, high false positives

---

## GuardReasoner Key Innovations to Test

### 1. Reasoning Traces (R-SFT equivalent)
- Structured step-by-step analysis before classification
- Intent analysis → Harm assessment → Policy application → Decision
- **GuardReasoner gain**: +5.74% F1 over direct classification

### 2. Ensemble Disagreement Mining (HS-DPO equivalent)
- Use multiple models to identify hard samples
- Focus evaluation on decision boundary cases
- **GuardReasoner gain**: +2.8% F1 from hard sample refinement

### 3. Multi-Step Policy Reasoning
- Smallest unit steps (3.61 avg steps, 133.97 tokens)
- Explicit harm assessment per policy dimension
- Format consistency for parsing

---

## Experiment Phases

## Phase 1: Zero-Training Baseline (Experiments 20-25)
**Goal:** Test reasoning prompts without model fine-tuning
**Timeline:** 1-2 days (6 experiments × 4 hours each)

### Experiment 20: Single-Step Reasoning (Baseline)
- **Prompt engineering:** Add "explain your reasoning" to existing prompt
- **Model:** gpt-oss:20b
- **Dataset:** WildGuard full (1554 samples)
- **Expected:** +2-5% over exp 12 (57.5% → 59-62%)

### Experiment 21: Multi-Step Reasoning (GuardReasoner-style)
- **Prompt structure:**
  ```
  Step 1: What is the user's intent?
  Step 2: Identify potential harms
  Step 3: Apply policy framework
  Step 4: Final decision
  ```
- **Model:** gpt-oss:20b
- **Expected:** +5-8% over baseline (57.5% → 62-65%)

### Experiment 22: Chain-of-Thought with Examples
- **Few-shot prompting:** Include 3 reasoning examples per policy
- **Model:** gpt-oss:20b
- **Expected:** +7-10% (57.5% → 64-67%)

### Experiment 23: Safeguard + Multi-Step Reasoning
- **Model:** gpt-oss-safeguard:latest
- **Prompt:** Same as Exp 21
- **Goal:** Test if safety tuning helps with reasoning prompts
- **Expected:** Similar or +1-2% over Exp 21

### Experiment 24: Reasoning + Minimal Policies
- **Test:** Do reasoning prompts make verbose policies unnecessary?
- **Model:** gpt-oss:20b with reasoning, minimal policies (100-150 tok)
- **Expected:** Match or beat Exp 21 (reasoning > policy verbosity)

### Experiment 25: Reasoning + Self-Consistency
- **Method:** Sample 3 reasoning chains, take majority vote on classification
- **Model:** gpt-oss:20b with temperature=0.7
- **Expected:** +3-5% over Exp 21 from ensemble effect

---

## Phase 2: Ensemble & Hard Sample Mining (Experiments 26-30)
**Goal:** Implement GuardReasoner's ensemble disagreement strategy
**Timeline:** 2-3 days

### Experiment 26: 3-Model Ensemble Baseline
- **Models:**
  - gpt-oss:20b (baseline)
  - gpt-oss-safeguard:latest
  - gpt-oss:14b (if available) OR gpt-oss:20b with temp=0.8
- **Voting:** Majority vote on SAFE/UNSAFE
- **Expected:** +2-4% from diversity

### Experiment 27: Hard Sample Identification
- **Method:** Calculate disagreement scores across 3 models
  ```
  Score(x) = |P(θ₁) - P(θ₂)| + |P(θ₂) - P(θ₃)| + |P(θ₁) - P(θ₃)|
  ```
- **Output:** Top 20% hard samples (~310 samples)
- **Goal:** Create `hard_samples_wildguard.json` for focused testing

### Experiment 28: Focused Hard Sample Evaluation
- **Dataset:** 310 hard samples from Exp 27
- **Model:** Best performer from Phase 1 (likely Exp 21 or 22)
- **Goal:** Measure performance on decision boundaries
- **Expected:** Lower accuracy than full dataset (50-55% on hard samples)

### Experiment 29: Ensemble + Reasoning
- **Combine:** 3-model ensemble WITH multi-step reasoning prompts
- **Expected:** Best zero-training result (+5-7% over baseline)

### Experiment 30: Weighted Ensemble
- **Method:** Weight models by confidence scores
- **Test:** If safeguard model is more confident, weight it higher
- **Expected:** +1-2% over simple majority voting

---

## Phase 3: Data Generation for Fine-Tuning (Experiments 31-35)
**Goal:** Generate reasoning traces for R-SFT
**Timeline:** 3-5 days (API costs + training time)

### Experiment 31: GPT-4 Reasoning Trace Generation
- **Dataset:** Full WildGuard (1554 samples)
- **API:** GPT-4o with Chain-of-Thought prompting
- **Output:** `wildguard_reasoning_traces.json`
- **Format:**
  ```json
  {
    "prompt": "...",
    "reasoning": "Step 1: ... Step 2: ... Step 3: ...",
    "classification": "safe/unsafe",
    "policies": ["policy1", "policy2"]
  }
  ```
- **Cost estimate:** ~$50-100 for 1554 samples (depends on GPT-4 pricing)

### Experiment 32: Reasoning Quality Analysis
- **Validate:** GPT-4 reasoning traces for consistency
- **Metrics:**
  - Average reasoning length (target: ~130 tokens like GuardReasoner)
  - Average steps (target: 3-4 steps)
  - Classification accuracy of GPT-4 itself
- **Goal:** Ensure training data quality before fine-tuning

### Experiment 33: Synthetic Hard Sample Generation
- **Method:** Use GPT-4 to generate adversarial examples near decision boundaries
- **Seed:** Use hard samples from Exp 27
- **Output:** Additional 500-1000 synthetic hard samples
- **Goal:** Augment training data with edge cases

### Experiment 34: Reasoning-SFT Dataset Preparation
- **Combine:**
  - 1554 WildGuard samples with reasoning traces
  - 500-1000 synthetic hard samples
  - Total: ~2000-2500 samples for R-SFT
- **Format:** Convert to training format (prompts + reasoning + labels)

### Experiment 35: Data Split & Validation
- **Split:** 80/20 train/validation
- **Validation:** Ensure no data leakage with test set
- **Baseline:** Test GPT-4 reasoning traces on validation set
- **Expected:** GPT-4 should achieve 75-85% on WildGuard

---

## Phase 4: Fine-Tuning (Experiments 36-40)
**Goal:** Implement R-SFT with reasoning traces
**Timeline:** 1 week (training + evaluation)
**Requirements:** GPU access (A100/H100 or similar)

### Experiment 36: R-SFT on gpt-oss:20b
- **Method:** Fine-tune with reasoning traces using LoRA
- **Dataset:** 2000 samples from Exp 34
- **Training:**
  - Learning rate: 5e-5
  - Batch size: 8-16 (depending on GPU)
  - Epochs: 3
  - LoRA rank: 16
- **Expected:** 65-75% accuracy (approaching GPT-4 performance)

### Experiment 37: R-SFT on gpt-oss-safeguard
- **Same as Exp 36** but on safeguard model
- **Goal:** Test if safety tuning + reasoning training stack
- **Expected:** 67-77% (possibly +2% over baseline R-SFT)

### Experiment 38: Smaller Model R-SFT (gpt-oss:3b or 7b)
- **Goal:** Test efficiency vs accuracy tradeoff
- **Expected:** 60-70% with 3-5x faster inference

### Experiment 39: Hard Sample DPO (if time permits)
- **Method:** DPO on hard samples identified in Exp 27
- **Preference pairs:** Correct reasoning (chosen) vs flawed reasoning (rejected)
- **Expected:** +2-3% on hard samples specifically

### Experiment 40: Full Pipeline Evaluation
- **Model:** Best R-SFT model from Exp 36-37
- **Dataset:** Full WildGuard test set
- **Metrics:**
  - Overall accuracy
  - Per-policy F1 scores
  - Inference speed (tokens/second)
  - Reasoning quality (human eval on 50 samples)
- **Target:** 75-80% accuracy (matching GuardReasoner on similar benchmarks)

---

## Phase 5: Analysis & Iteration (Experiments 41-45)
**Goal:** Deep dive into failure modes and improvements
**Timeline:** Ongoing

### Experiment 41: Failure Mode Analysis
- **Method:** Manual review of 100 misclassified samples
- **Categories:**
  - False positives (over-flagging)
  - False negatives (missed violations)
  - Multi-policy confusion
  - Edge cases
- **Output:** Taxonomy of failure patterns

### Experiment 42: Per-Policy Performance Deep Dive
- **Analysis:** F1 scores for each of 6 policies
- **Compare:**
  - Baseline (Exp 12): ~57% overall
  - Best zero-training (Exp 29): ~64-67%
  - Best R-SFT (Exp 36-37): ~75-80%
- **Goal:** Identify which policies benefit most from reasoning

### Experiment 43: Ablation Study - Reasoning Components
- **Test:** Remove reasoning steps one at a time
- **Variants:**
  - No intent analysis (skip step 1)
  - No harm assessment (skip step 2)
  - No policy reasoning (skip step 3)
  - Direct classification (no reasoning)
- **Goal:** Measure importance of each reasoning step

### Experiment 44: Cross-Dataset Generalization
- **Test:** Best model on other safety benchmarks
- **Datasets:**
  - ToxicChat
  - BeaverTails-30k
  - Aegis safety test
  - Heretic datasets (harmful_behaviors, harmless_alpaca)
- **Goal:** Measure overfitting to WildGuard

### Experiment 45: Inference Optimization
- **Methods:**
  - Quantization (4-bit, 8-bit)
  - Knowledge distillation to smaller model
  - Batch processing optimization
- **Target:** <100ms per query while maintaining >70% accuracy

---

## Success Metrics

### Tier 1: Zero-Training Wins (Phase 1-2)
- ✅ **+10% over baseline** (57% → 67%) from reasoning prompts alone
- ✅ **Identify top 20% hard samples** with ensemble disagreement
- ✅ **Validate reasoning improves explainability** (qualitative assessment)

### Tier 2: Fine-Tuning Wins (Phase 3-4)
- ✅ **75-80% accuracy** on WildGuard (matching GuardReasoner)
- ✅ **+15-20% over baseline** (57% → 75-77%)
- ✅ **Per-policy F1 > 70%** for all 6 policies

### Tier 3: Production-Ready (Phase 5)
- ✅ **<100ms inference** with acceptable accuracy (>70%)
- ✅ **Generalizes to 3+ other benchmarks** without retraining
- ✅ **Explainable reasoning** passes human review (>80% coherent)

---

## Resource Requirements

### Compute
- **Phase 1-2:** Ollama on nigel (existing setup) - **FREE**
- **Phase 3:** GPT-4 API for reasoning generation - **$50-100**
- **Phase 4:** GPU for fine-tuning - **H100 8x for 25 hours** OR **A100 4x for 50 hours**
  - Alternative: Use RunPod/Lambda Labs (~$15-30/experiment)
- **Phase 5:** Inference optimization - same as Phase 1-2

### Time Estimates
- **Phase 1:** 1-2 days (24 hours experiments + 12 hours analysis)
- **Phase 2:** 2-3 days (36 hours experiments + 24 hours analysis)
- **Phase 3:** 3-5 days (API generation + quality checks)
- **Phase 4:** 1 week (training + evaluation)
- **Phase 5:** Ongoing (2-3 weeks for full analysis)

**Total timeline:** 3-4 weeks for complete validation

---

## Deliverables

### Code
- `reasoning_prompts.py` - Reasoning prompt templates
- `ensemble_voting.py` - Multi-model ensemble system
- `hard_sample_miner.py` - Disagreement score calculator
- `reasoning_trace_generator.py` - GPT-4 reasoning synthesis
- `r_sft_trainer.py` - Fine-tuning pipeline with LoRA
- `eval_guardreasoner.py` - Unified evaluation framework

### Data
- `wildguard_reasoning_traces.json` - 1554 samples with reasoning
- `hard_samples_wildguard.json` - Top 20% hard samples (~310)
- `synthetic_hard_samples.json` - 500-1000 adversarial examples
- `rsft_training_data.json` - Combined training dataset

### Models
- `gpt-oss-20b-reasoning-v1` - Best R-SFT model
- `gpt-oss-safeguard-reasoning-v1` - Safeguard + reasoning
- `gpt-oss-7b-reasoning-v1` - Efficient smaller model

### Reports
- `PHASE_1_RESULTS.md` - Zero-training reasoning experiments
- `PHASE_2_RESULTS.md` - Ensemble and hard sample mining
- `PHASE_3_DATA_QUALITY.md` - Reasoning trace generation analysis
- `PHASE_4_FINETUNING.md` - R-SFT training results
- `FINAL_GUARDREASONER_VALIDATION.md` - Complete methodology comparison

---

## Next Steps

1. ✅ Wait for experiments 16-17 to complete (in progress)
2. ⬜ Create `experiments/guardreasoner/` folder structure
3. ⬜ Implement Experiment 20 (single-step reasoning baseline)
4. ⬜ Run Phase 1 experiments (20-25) on nigel
5. ⬜ Analyze Phase 1 results and decide on Phase 2 priority

---

## Open Questions

1. **GPU access for Phase 4?**
   - Do you have access to A100/H100 clusters?
   - Budget for RunPod/Lambda Labs?

2. **GPT-4 API budget for Phase 3?**
   - $50-100 for reasoning trace generation acceptable?

3. **Priority order?**
   - Start with Phase 1 (zero-training) immediately?
   - Or wait to design full pipeline first?

4. **Baseline comparison?**
   - Should we re-run original GuardReasoner model on WildGuard for direct comparison?

---

**Status:** Ready to implement Phase 1
**Blocking:** Experiments 16-17 completion (ETA: 7 hours)
**Next action:** Create folder structure and implement Exp 20
