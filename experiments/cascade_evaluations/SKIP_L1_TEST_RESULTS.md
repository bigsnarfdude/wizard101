# Skip L1 Test Results - November 21, 2025

## Question

Can we skip GuardReasoner-8B (L1) and go direct to gpt-oss-safeguard (L2) for speed?

## Answer

**No. L1 is critical.**

---

## Test Setup

Ran gpt-oss-safeguard:20b directly on the 61 samples that normally go to L1 (uncertain samples where L0 confidence < 0.9).

---

## Results

| Layer | Accuracy | Value Added | Time |
|-------|----------|-------------|------|
| L0 (baseline) | 59.0% | - | - |
| **L1 (GuardReasoner-8B)** | **88.5%** | **+29.5%** | 501s |
| L2 (gpt-oss-safeguard) | 59.0% | +0.0% | 10.7s |

### Key Metrics

- **L2 fixed**: 0 samples
- **L2 broke**: 0 samples
- **L2 value added**: 0%

L2 on uncertain samples performs the same as L0 - essentially random guessing.

---

## Why L1 Matters

The 61 samples that reach L1 are **hard cases** where L0 wasn't confident:

1. **Ambiguous phrasing** - "murder of crows", "kill a process"
2. **Context-dependent** - Questions that could be harmful or educational
3. **Subtle harm** - Discrimination phrased as questions
4. **Edge cases** - Privacy violations, self-harm references

These require **reasoning**, not just classification. GuardReasoner-8B's step-by-step analysis:
- Identifies what's being requested
- Considers legitimate vs harmful uses
- Evaluates context clues
- Makes informed judgment

A simple classifier (L2) can't do this - it just pattern matches.

---

## Speedup Fantasy vs Reality

| Approach | Time | Accuracy | Works? |
|----------|------|----------|--------|
| Skip L1, use L2 | 11s | 59.0% | ❌ |
| Keep L1 | 501s | 88.5% | ✓ |

The 45x speedup comes at 30% accuracy loss - unacceptable for safety.

---

## Conclusion

**GuardReasoner-8B earns its 500s.**

The cascade architecture is validated:
- **L0** catches easy cases fast (94.2%)
- **L1** reasons through hard cases (5.8%)
- **L2** breaks ties when L0/L1 disagree (2.3%)

Each layer serves a distinct purpose. No shortcuts.

---

## Files

- `skip_l1_test_results.json` - Full test results
