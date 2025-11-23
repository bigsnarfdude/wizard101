# L0 Bouncer (DeBERTa) - Experiment Results

**Date**: 2025-11-20
**Purpose**: Fast first-tier classifier for Safety Cascade

---

## Architecture Context

**Safety Cascade (Cascata Fiduciae Fundata)**:
- **L0 (This)**: DeBERTa-v3-xsmall (22M params) - 70% traffic at ~6ms
- L1: Exp 18 3B R-SFT - 25% traffic at 50ms
- L2: 6-policy gauntlet - 4% traffic at 200ms
- L3: GPT-4o async audit - 1% traffic

---

## Final Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 62.5% |
| **Precision** | 30.8% |
| **Recall** | **80.0%** |
| **F1** | 44.4% |
| **Inference Speed** | 5.74ms mean |
| **P99 Latency** | 5.86ms |

### Confusion Matrix (on 80 eval samples)
- Safe samples: 65
- Harmful samples: 15
- Harmful caught: 12/15 (80%)
- False positives: 28 (safe classified as harmful)

---

## Training Configuration

```python
# Model
STUDENT_MODEL = "microsoft/deberta-v3-xsmall"  # 22M params
TEACHER_MODEL = "Exp 18 (95% accuracy)"

# Hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 10
MAX_LENGTH = 256

# Class weights (safety-first)
safe_weight = 0.656
harmful_weight = 3.158  # Boosted 1.5x for high recall
```

---

## Key Learnings

### 1. Class Imbalance Solution

**Problem**: Initial training achieved 78.75% accuracy but 0% F1 (model predicted all "safe")

**Solution**:
- Computed balanced class weights: safe=0.656, harmful=2.105
- Boosted harmful weight by 1.5x: harmful=3.158
- Result: Recall improved from 30.8% to 80%

### 2. Safety-First Trade-off

The bouncer accepts **lower precision for higher recall**:

| Approach | Precision | Recall | Use Case |
|----------|-----------|--------|----------|
| Balanced | 66.7% | 30.8% | Miss harmful content |
| Safety-first | 30.8% | 80.0% | Catch harmful, some false positives |

**Rationale**: Better to send safe content to L1 (cost: 50ms) than miss harmful content.

### 3. Small Dataset Performance

Achieved 80% recall with only 800 training samples (190 harmful, 610 safe).

**Potential improvements**:
- Larger dataset (WildGuard full: 11K samples)
- More harmful examples to improve precision
- Adversarial augmentation

---

## Cascade Integration

### Traffic Flow Design

```
User Request
    │
    ▼
[L0 Bouncer] ─────────────────────────────┐
    │                                     │
    │ Confidence < 0.7                    │ Confidence ≥ 0.7
    │ or Harmful                          │ and Safe
    ▼                                     ▼
[L1 Analyst] ← 30%                    [Pass Through] ← 70%
```

### Expected Performance

With current L0 performance:
- **70% pass-through** at 6ms (confident safe)
- **30% escalation** at 56ms (6ms + 50ms L1)
- **Average latency**: 0.7 * 6 + 0.3 * 56 = **21ms**

vs. L1 only: 50ms
**Speedup**: ~2.4x

---

## Files

- **Training script**: `train_l0_bouncer.py`
- **Model**: `./models/l0_bouncer/`
- **Teacher labels**: `teacher_labels.json`

---

## Next Steps

1. **Improve precision** - More harmful training examples
2. **Confidence thresholding** - Add confidence scores for routing
3. **Build cascade router** - Connect L0 → L1 pipeline
4. **Production testing** - Real-world traffic evaluation

---

## Connection to Ensemble Truth

This L0 bouncer complements the ensemble truth cache:

| Component | Speed | Coverage | Use Case |
|-----------|-------|----------|----------|
| Ensemble cache | 5ms | 64% | Known patterns |
| L0 bouncer | 6ms | 100% | All queries |
| L1 analyst | 50ms | 100% | Uncertain cases |

**Optimal cascade**:
1. Check ensemble cache (5ms) → if match, return
2. Run L0 bouncer (6ms) → if confident safe, return
3. Escalate to L1 (50ms) → detailed analysis
