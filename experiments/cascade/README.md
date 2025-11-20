# Cascata Fiduciae Fundata - Safety Cascade

**Cascade of Founded Trust** - A multi-tier safety classification system.

---

## Architecture

```
User Request
     │
     ▼
┌─────────────┐
│  L0 Bouncer │  DeBERTa-v3-xsmall (22M params)
│   ~6ms      │  80% recall, handles 70% traffic
└─────┬───────┘
      │
      │ Uncertain/Harmful
      ▼
┌─────────────┐
│ L1 Analyst  │  Llama 3.2 3B R-SFT (Exp 18)
│   ~50ms     │  95% accuracy, handles 25% traffic
└─────┬───────┘
      │
      │ Edge cases
      ▼
┌─────────────┐
│ L2 Gauntlet │  6 Policy Personas (parallel)
│  ~200ms     │  Multi-perspective, handles 4% traffic
└─────┬───────┘
      │
      │ High-stakes
      ▼
┌─────────────┐
│  L3 Judge   │  GPT-4o (async audit)
│  ~2000ms    │  Final authority, handles 1% traffic
└─────────────┘
```

---

## Components

### L0: Bouncer (Complete)
- **Model**: microsoft/deberta-v3-xsmall
- **Size**: 22M parameters
- **Speed**: 5.74ms mean
- **Performance**: 80% recall, 30.8% precision
- **Role**: Fast triage - pass confident safe, escalate uncertain

**Files**: `train_l0_bouncer.py`, `L0_BOUNCER_RESULTS.md`

### L1: Analyst (Complete)
- **Model**: Llama 3.2 3B with LoRA
- **Method**: R-SFT (Reasoning Supervised Fine-Tuning)
- **Speed**: ~50ms
- **Performance**: 95% accuracy, 94.5% harmful F1
- **Role**: Detailed reasoning analysis

**Location**: `../guardreasoner/models/exp_18_batch8_lora/`

### L2: Gauntlet (Planned)
- **Method**: 6 policy personas in parallel
- **Personas**: Legal, Ethics, Security, Privacy, Safety, Business
- **Speed**: ~200ms (parallel execution)
- **Role**: Multi-perspective consensus for edge cases

### L3: Judge (Planned)
- **Model**: GPT-4o
- **Method**: Async audit trail
- **Speed**: ~2000ms
- **Role**: Final authority, creates training signal

---

## Expected Performance

| Tier | Traffic | Latency | Cumulative Latency |
|------|---------|---------|-------------------|
| L0 Pass | 70% | 6ms | 6ms |
| L1 Pass | 25% | 56ms | 56ms |
| L2 Pass | 4% | 256ms | 256ms |
| L3 | 1% | 2256ms | 2256ms |

**Average Latency**: 0.70×6 + 0.25×56 + 0.04×256 + 0.01×2256 = **51ms**

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

- `README.md` - This file
- `train_l0_bouncer.py` - L0 training script
- `L0_BOUNCER_RESULTS.md` - L0 experiment results
- `ENSEMBLE_TRUTH_LEARNINGS.md` - Ensemble cache research

---

## Next Steps

1. **Build cascade router** - Connect L0 → L1 with confidence thresholding
2. **Implement L2 gauntlet** - Policy persona prompts
3. **Add async L3 audit** - GPT-4o logging
4. **Production testing** - Real traffic evaluation
5. **Confidence calibration** - Tune routing thresholds

---

## References

- GuardReasoner paper (Liu et al., 2024)
- WildGuard dataset
- Safety cascade design principles
