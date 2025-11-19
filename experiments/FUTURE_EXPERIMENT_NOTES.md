# Future Experiment Notes

**Date**: 2025-11-19
**Context**: Post Exp 18 epoch 3 training analysis

---

## Key Learning: Binary vs 3-Task Classification

### What We Missed

Our current Exp 18 uses **binary classification** (harmful/safe) instead of the paper's **3-task classification**.

**Impact**:
- Cannot evaluate on response-based benchmarks
- Missing ~15% accuracy ceiling
- Not production-ready for full safety pipeline

---

## Benchmarks We Cannot Evaluate

| Benchmark | Requires | Our Support |
|-----------|----------|-------------|
| ToxicChat | Prompt harm | ✅ |
| HarmBench Prompt | Prompt harm | ✅ |
| OpenAI Moderation | Prompt harm | ✅ |
| **HarmBench Response** | Response harm | ❌ |
| **SafeRLHF** | Response harm | ❌ |
| **BeaverTails** | Response harm | ❌ |
| **XSTest Refusal** | Refusal detection | ❌ |
| **WildGuard Full** | All 3 tasks | ❌ (partial only) |

**Result**: Can only evaluate on ~6 of 13 paper benchmarks

---

## Why 3-Task Matters

### 1. Richer Training Signal
- 3 labels per sample vs 1
- More gradient signal → better generalization

### 2. Full Conversation Evaluation
- Binary: Only sees user prompt
- 3-Task: Sees prompt + AI response

### 3. Edge Case Coverage
```
Safe prompt + Harmful response = Binary says "safe" ❌
Safe prompt + Harmful response = 3-Task catches it ✅
```

### 4. Production Cascade Design
- Binary: Need multiple models for full pipeline
- 3-Task: Single model covers entire safety check

### 5. Accuracy Ceiling
- Binary: ~70% ceiling
- 3-Task: ~84% (paper result)

---

## Action Items for Next Experiment

### Experiment 19+ Requirements

1. **Use GuardReasonerTrain dataset**
   ```python
   from datasets import load_dataset
   ds = load_dataset("yueliu1999/GuardReasonerTrain")
   ```

2. **Implement 3-task output format**
   ```
   Input: prompt + response
   Output:
   - prompt_harm_label: harmful/unharmful
   - response_refusal_label: refusal/compliance
   - response_harm_label: harmful/unharmful
   ```

3. **Update evaluation scripts** for all 3 tasks

4. **Add benchmark coverage**:
   - HarmBench Response
   - SafeRLHF
   - BeaverTails
   - XSTest Refusal
   - Full WildGuard

---

## Exp 18 Value

Despite using wrong data format, Exp 18 still valuable:

- ✅ Validates training pipeline works
- ✅ Confirms LoRA approach is viable
- ✅ Establishes baseline for binary classification
- ✅ Provides ablation comparison (binary vs 3-task)

**Expected Exp 18 result**: 65-70% accuracy (binary)
**Expected Exp 19+ result**: 75-80% accuracy (3-task)

---

## Dataset Comparison

| Aspect | Our Exp 18 | GuardReasonerTrain |
|--------|------------|-------------------|
| Samples | 11K | 128K |
| Tasks | 1 (binary) | 3 (prompt/refusal/response) |
| Input | Prompt only | Prompt + Response |
| Sources | 2 | 4 |
| Reasoning | Yes | Yes (GPT-4o generated) |

---

## Recommended Next Steps

1. **Complete Exp 18** (in progress, ~1h remaining)
2. **Evaluate Exp 18** on available benchmarks
3. **Download GuardReasonerTrain** dataset
4. **Start Exp 19** with 3-task format
5. **Evaluate on all 13 benchmarks**
6. **Compare binary vs 3-task** results

---

## Code Changes Needed

### Training Format
```python
# Current (binary)
{
    "prompt": "user request",
    "label": "harmful"
}

# Needed (3-task)
{
    "prompt": "user request",
    "response": "ai response",
    "prompt_harm_label": "harmful",
    "response_refusal_label": "refusal",
    "response_harm_label": "unharmful"
}
```

### Evaluation Changes
```python
# Current
accuracy = (correct_harmful + correct_safe) / total

# Needed
task1_f1 = compute_f1(prompt_harm_preds, prompt_harm_labels)
task2_f1 = compute_f1(refusal_preds, refusal_labels)
task3_f1 = compute_f1(response_harm_preds, response_harm_labels)
overall_f1 = (task1_f1 + task2_f1 + task3_f1) / 3
```

---

## References

- Paper: GuardReasoner (Liu et al., 2025) - arXiv:2501.18492
- Dataset: huggingface.co/datasets/yueliu1999/GuardReasonerTrain
- Our comparison: experiments/GUARDREASONER_COMPARISON.md

---

**Bottom Line**: Exp 18 is a valid baseline but we need to switch to 3-task for proper paper replication and full benchmark coverage.
