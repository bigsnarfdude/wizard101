# RL Training Plan for GuardReasoner

## Paper's RL Findings (Appendix F)

### ❌ GRPO Failed - Negative Results
**What they tried:**
- GRPO (Group Relative Policy Optimization) on difficult samples
- Starting from 5k-sample fine-tuned model
- Verifiable reward: accuracy of safety label
- Parameters tested:
  - Temperature: 0.6 to 1.4
  - Learning rate: 1e-5 to 1e-6
  - KL coefficient: 0 to 0.03
  - Rollout samples: 4 to 16
- Training time: 4-8 hours per experiment
- Hardware: 1-2 nodes of 8xA100

**Results:**
- ✅ +1% improvement on JBB-Response harmful-F1
- ❌ -1-2% drop on all other tasks
- ❌ Overall: -1.1% average F1 score

**Why it failed:**
> "This is a negative result that highlights the difficulty of improving with GRPO the performance of safety guard models distilled from a strong teacher."

**Key insight:** When you distill from a very strong teacher (DeepSeek-R1-671B), RL struggles to improve further.

## Our Strategy: Different Approach

### Core Problem
- SFT from strong teacher = high baseline
- Traditional RL (GRPO, DPO) = marginal or negative gains
- Need: RL that improves **reasoning quality** not just accuracy

### Proposed Approach: DPO on Reasoning Quality

Instead of optimizing for label accuracy (paper's mistake), optimize for **reasoning quality**.

#### Key Idea
Create preference pairs where:
- **Chosen**: Good reasoning → correct label
- **Rejected**: Bad reasoning → correct/incorrect label OR good reasoning → wrong label

#### Why This Works
1. **Reasoning Quality Signal**: Not just "did you get it right?" but "did you reason well?"
2. **Generalization**: Better reasoning → better OOD performance
3. **Interpretability**: Human-verifiable reasoning traces
4. **Avoids Teacher Collapse**: Not trying to match teacher accuracy, improving reasoning process

## Implementation Plan

### Phase 1: Quick Evaluation (Today)
Test if 1-epoch SFT model works at all.

```python
# experiments/guardreasoner/evaluate_exp_18_quick.py
"""
Quick test: 100 samples from WildGuard test
- Load exp_18_rsft_lora
- Generate reasoning + labels
- Calculate accuracy
- Manual check: Are reasoning traces coherent?
"""
```

**Success criteria**: >50% accuracy + coherent reasoning

### Phase 2: Toy RL Experiment (1-2 days)

#### Dataset Creation
```python
# experiments/guardreasoner/create_rl_preferences.py
"""
Create preference dataset for DPO:
1. Load 1k samples from WildGuard
2. Generate 4 completions per prompt (temperature=0.8)
3. Score each completion:
   - Reasoning coherence (automated via judge model)
   - Label correctness (ground truth)
   - Combined score = 0.7*reasoning + 0.3*correctness
4. Create pairs: (chosen=high_score, rejected=low_score)
5. Save to guardreasoner_dpo_toy.json
"""
```

**Dataset size**: 1,000 prompts × 4 completions = 4,000 generations → 1,000 preference pairs

#### Toy RL Training
```python
# experiments/guardreasoner/train_exp_19_dpo_toy.py
"""
DPO training on toy dataset:
- Base: exp_18_rsft_lora (1-epoch SFT)
- Dataset: 1k preference pairs
- Method: DPO (not GRPO)
- Epochs: 1
- Time: ~2 hours
"""
```

**Configuration:**
```python
dpo_config = {
    "base_model": "exp_18_rsft_lora",
    "dataset": "guardreasoner_dpo_toy.json",
    "beta": 0.1,  # DPO temperature
    "learning_rate": 5e-6,  # Lower than SFT
    "epochs": 1,
    "batch_size": 4,
    "gradient_accumulation": 8,
}
```

#### Evaluation
```python
# experiments/guardreasoner/evaluate_exp_19_dpo.py
"""
Compare models:
1. Baseline: exp_18_rsft_lora (SFT only)
2. DPO: exp_19_dpo_toy
Metrics:
- Accuracy
- Reasoning quality score (automated judge)
- Human eval: 50 samples manual inspection
"""
```

**Success criteria for toy experiment**:
- Accuracy: Similar to SFT baseline (±2%)
- Reasoning quality: +10% improvement
- Human eval: Prefer DPO reasoning in >60% cases

### Phase 3: Full RL Training (3-4 days)

If toy experiment works:

#### Full Dataset
- 10k prompts from WildGuard train set
- 4 completions each = 40k generations
- 10k preference pairs

#### Full DPO Training
- Base: exp_18_rsft_lora
- Dataset: 10k pairs
- Epochs: 2-3
- Time: ~16-24 hours

#### Comprehensive Evaluation
- Full WildGuard test set (1,554 samples)
- Compare: Baseline SFT vs DPO-tuned
- Metrics: Accuracy, reasoning quality, human preference

## Alternative RL Approaches (If DPO Fails)

### Option A: RLAIF (RL from AI Feedback)
- Use DeepSeek-R1 as judge
- Score reasoning quality
- Train with PPO instead of DPO

### Option B: Constitutional AI
- Define "constitution" for good reasoning:
  1. "Reasoning must address specific content in prompt"
  2. "Must identify relevant safety categories"
  3. "Must explain why content violates/doesn't violate policy"
- Use constitution-based reward

### Option C: Iterative Refinement
- Generate reasoning → judge scores it → regenerate if low → train on refined pairs

## Why Our Approach > Paper's GRPO

| Aspect | Paper (GRPO) | Our Approach (DPO) |
|--------|--------------|-------------------|
| **Objective** | Match teacher accuracy | Improve reasoning quality |
| **Signal** | Binary (correct/incorrect) | Continuous (reasoning score) |
| **Reward** | Label accuracy only | Reasoning + accuracy |
| **Problem** | Teacher ceiling effect | Room for reasoning improvement |
| **Method** | On-policy (GRPO) | Off-policy (DPO) - more stable |
| **Expected** | Marginal gains | Reasoning quality gains |

## Reasoning Quality Scoring

### Automated Judge (for dataset creation)
```python
def score_reasoning(prompt, reasoning, label, ground_truth):
    """
    Score reasoning trace quality (0-100)
    """
    # Component 1: Correctness (30%)
    correctness = 100 if label == ground_truth else 0

    # Component 2: Coherence (30%)
    # Use small judge model (Llama-3.2-3B)
    coherence = judge_coherence(reasoning)

    # Component 3: Relevance (20%)
    # Does reasoning address specific prompt content?
    relevance = judge_relevance(prompt, reasoning)

    # Component 4: Specificity (20%)
    # Avoids generic responses?
    specificity = judge_specificity(reasoning)

    total = (0.3 * correctness +
             0.3 * coherence +
             0.2 * relevance +
             0.2 * specificity)

    return total
```

### Human Evaluation Template
For manual inspection of 50 samples:
```
Prompt: [user prompt]

Reasoning A (SFT): [reasoning trace]
Label A: [safe/harmful]

Reasoning B (DPO): [reasoning trace]
Label B: [safe/harmful]

Ground Truth: [safe/harmful]

Questions:
1. Which reasoning is more coherent? (A/B/Tie)
2. Which reasoning is more specific? (A/B/Tie)
3. Which reasoning is more helpful? (A/B/Tie)
4. Which would you prefer in production? (A/B/Tie)
```

## Timeline

### Week 1 (Current)
- ✅ Day 1: SFT training complete (Exp 18)
- ✅ Day 1: Model uploaded to HuggingFace
- 🔄 Day 1-2: Quick evaluation (Phase 1)
- 📋 Day 2-3: Create toy preference dataset

### Week 2
- 📋 Day 1-2: Toy DPO training (Exp 19)
- 📋 Day 2: Evaluate toy DPO
- 📋 Day 3: Decision: Continue to full DPO?
- 📋 Day 4-5: Create full preference dataset (if yes)

### Week 3
- 📋 Day 1-3: Full DPO training (16-24 hours)
- 📋 Day 4: Full evaluation
- 📋 Day 5: Human preference study

## Success Metrics

### Toy Experiment (Exp 19)
| Metric | Baseline (SFT) | Target (DPO) | Stretch |
|--------|---------------|--------------|---------|
| Accuracy | ~55% | 53-57% (±2%) | 58%+ |
| Reasoning Quality | 60 | 66+ (+10%) | 70+ |
| Human Preference | - | 60% prefer DPO | 70% |

### Full Experiment (Exp 20)
| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Accuracy | 57.5% | 59-61% | 63%+ |
| Reasoning Quality | 65 | 72+ | 75+ |
| Human Preference | - | 65% prefer | 75% |

## Key Hypotheses to Test

1. **Reasoning Quality Hypothesis**: RL on reasoning quality improves model beyond accuracy-only RL
2. **Generalization Hypothesis**: Better reasoning → better OOD generalization
3. **Sample Efficiency Hypothesis**: DPO needs fewer samples than GRPO for improvements
4. **Stability Hypothesis**: Off-policy DPO more stable than on-policy GRPO

## Files to Create

### Evaluation Scripts
- `evaluate_exp_18_quick.py` - Quick 100-sample test
- `evaluate_exp_18_full.py` - Full WildGuard test eval

### RL Dataset Creation
- `create_rl_preferences.py` - Generate preference pairs
- `judge_reasoning_quality.py` - Automated scoring
- `inspect_preferences.py` - Manual review tool

### Training Scripts
- `train_exp_19_dpo_toy.py` - Toy DPO (1k pairs)
- `train_exp_20_dpo_full.py` - Full DPO (10k pairs)

### Analysis Scripts
- `compare_sft_vs_dpo.py` - Side-by-side comparison
- `human_eval_tool.py` - Interface for human preference study
- `analyze_reasoning_improvements.py` - What did DPO improve?

## Next Steps

1. **Immediate** (today):
   - Create `evaluate_exp_18_quick.py`
   - Run quick test (100 samples)
   - Verify model works

2. **Tomorrow**:
   - Create `create_rl_preferences.py`
   - Generate toy dataset (1k pairs)
   - Start toy DPO training

3. **This Week**:
   - Complete toy experiment
   - Evaluate results
   - Decide: Full DPO or iterate on approach?

---

**Created**: 2025-11-18
**Status**: Ready to implement Phase 1
**Key Innovation**: RL on reasoning quality, not just accuracy
