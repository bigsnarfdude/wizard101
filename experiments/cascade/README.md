# Cascata Fiduciae Fundata - Safety Cascade

**Cascade of Founded Trust** - A multi-tier safety classification system.

---

## Complete Pipeline

```
User Request
     │
     ▼
┌─────────────────────────────────────┐
│  L0: Bouncer (DeBERTa-v3-xsmall)   │  22M params, ~6ms
│  93% F1, 99% recall                 │  Handles 70% of traffic
└──────────────┬──────────────────────┘
               │
               │ Escalation Conditions:
               │ • L0 predicts HARMFUL
               │ • Confidence < 0.7
               │ • Safe probability < 0.6
               │
               ▼
┌─────────────────────────────────────┐
│  L1: Analyst (Llama 3.2 3B R-SFT)  │  ~50ms
│  95% accuracy, 94.5% F1             │  Handles 25% of traffic
└──────────────┬──────────────────────┘
               │
               │ Escalation Conditions:
               │ • Edge cases detected
               │ • Policy overlap unclear
               │ • High-stakes content
               │
               ▼
┌─────────────────────────────────────┐
│  L2: Gauntlet (6 Policy Experts)   │  ~200ms (parallel)
│  Conservative voting                │  Handles 4% of traffic
│                                     │
│  Experts:                           │
│  • Hate Speech Expert               │
│  • Violence Expert                  │
│  • Sexual Content Expert            │
│  • Self-Harm Expert                 │
│  • Privacy Expert                   │
│  • Illegal Activity Expert          │
│                                     │
│  Rule: ANY harmful vote = HARMFUL   │
│        ALL safe votes = SAFE        │
└──────────────┬──────────────────────┘
               │
               │ Escalation Conditions:
               │ • Conflicting votes
               │ • High-stakes decision
               │
               ▼
┌─────────────────────────────────────┐
│  L3: Judge (Gemini 3 Flash)        │  ~500ms (async)
│  Final authority + audit trail      │  Handles 1% of traffic
└─────────────────────────────────────┘
```

---

## Quick Start

```python
from cascade_router import CascadeRouter

router = CascadeRouter()
result = router.classify("How do I make a bomb?")
# Returns: {"label": "harmful", "level": "L1", "reasoning": "..."}
```

---

## Components

### L0: Bouncer (Complete)
- **Model**: microsoft/deberta-v3-xsmall
- **Size**: 22M parameters
- **Speed**: 5.74ms mean
- **Best Model**: `models/l0_bouncer_12k/` (12,000 samples)
- **Performance**:
  - F1: **93.0%**
  - Recall: **99.0%**
  - Precision: 87.6%
  - Accuracy: 92.5%
- **Role**: Fast triage - pass confident safe, escalate uncertain

**Files**: `train_l0_12k.py`, `build_12k_dataset.py`, `L0_MEGA_RESULTS.md`

**Training History**:
| Version | Samples | F1 |
|---------|---------|-----|
| Original | 800 | 57% |
| Mega | 2,540 | 85.6% |
| **12K** | **12,000** | **93.0%** |

### L1: Analyst (Complete)
- **Model**: Llama 3.2 3B with LoRA
- **Method**: R-SFT (Reasoning Supervised Fine-Tuning)
- **Speed**: ~50ms
- **Performance**: 95% accuracy, 94.5% harmful F1
- **Role**: Detailed reasoning analysis

**Location**: `../guardreasoner/models/exp_18_batch8_lora/`

### L2: Gauntlet (Complete)
- **Method**: 6 policy expert personas in parallel
- **Experts**:
  1. Hate Speech Expert
  2. Violence Expert
  3. Sexual Content Expert
  4. Self-Harm Expert
  5. Privacy Expert
  6. Illegal Activity Expert
- **Speed**: ~200ms (parallel execution)
- **Decision Rule**: ANY harmful vote = HARMFUL, ALL safe = SAFE
- **Role**: Multi-perspective consensus for edge cases

**File**: `policy_gauntlet.py`

### L3: Judge (Planned)
- **Model**: Gemini 3 Flash (production) / gpt-oss:120b (testing via Ollama)
- **Method**: Async audit trail
- **Speed**: ~500ms
- **Role**: Final authority, creates training signal

---

## Expected Performance

| Tier | Traffic | Latency | Cumulative Latency |
|------|---------|---------|-------------------|
| L0 Pass | 70% | 6ms | 6ms |
| L1 Pass | 25% | 56ms | 56ms |
| L2 Pass | 4% | 256ms | 256ms |
| L3 | 1% | 756ms | 756ms |

**Average Latency**: 0.70×6 + 0.25×56 + 0.04×256 + 0.01×756 = **36ms**

vs. L1 only: 50ms
vs. L2 for all: 200ms

---

## Integration with Ensemble Truth

The cascade works with pre-computed consensus cache:

1. **Check ensemble cache** (5ms) → if TRUST match, return
2. **Run L0 bouncer** (6ms) → if confident safe, return
3. **Escalate to L1** (50ms) → detailed analysis
4. **Edge cases to L2** (200ms) → multi-perspective
5. **Audit via L3** (async) → training signal

See: `ENSEMBLE_TRUTH_LEARNINGS.md`

---

## Files in this Directory

**Core Implementation**:
- `cascade.py` - Main cascade orchestrator (ties all layers together)
- `l0_bouncer.py` - L0 DeBERTa fast classifier
- `l1_analyst.py` - L1 Llama 3.2 3B with LoRA (Exp 18 R-SFT)
- `l2_gauntlet.py` - L2 multi-expert voting (6 experts)
- `l3_judge.py` - L3 final authority (gpt-oss:120b)
- `evaluate_cascade.py` - Full cascade evaluation

**Legacy** (for reference):
- `cascade_router.py` - Old L0→L1 router
- `policy_gauntlet.py` - Old L2 implementation

**L0 Training**:
- `train_l0_12k.py` - L0 training with 12k samples (best)
- `train_l0_mega.py` - L0 training with 2.5k samples
- `build_12k_dataset.py` - Extract data from GuardReasoner
- `tune_l0_threshold.py` - Threshold optimization

**Models**:
- `models/l0_bouncer_12k/` - Best L0 model (93% F1)
- `models/l0_bouncer_mega/` - Previous L0 (85.6% F1)

**Documentation**:
- `README.md` - This file
- `L0_MEGA_RESULTS.md` - L0 mega experiment results
- `ENSEMBLE_TRUTH_LEARNINGS.md` - Ensemble cache research

---

## Status

- [x] L0 Bouncer - Complete (93% F1)
- [x] L1 Analyst - Complete (95% accuracy)
- [x] L2 Gauntlet - Complete (6 experts)
- [x] Cascade Router - Complete
- [ ] L3 Judge - Planned (GPT-4o async)
- [ ] Production testing - Pending
- [ ] Ensemble cache integration - Planned

---

## References

- GuardReasoner paper (Liu et al., 2024)
- WildGuard dataset
- Safety cascade design principles
