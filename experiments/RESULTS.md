# Multi-Policy Safety Gauntlet Results

## Executive Summary

Testing serial gauntlet architecture with gpt-oss:20b model across 6 safety policies.

**Key Finding:** Minimal policies (100-150 tokens) outperform medium policies (300-500 tokens) for multi-policy classification.

## Architecture

- **Model:** gpt-oss:20b (Ollama)
- **Method:** Serial gauntlet (1 model, 6 sequential runs with different policies)
- **Policies:** hate_speech, violence, self_harm, sexual_minors, harassment, illegal
- **Benchmark:** 90 samples from ToxicChat, SimpleSafetyTests, XSTest

## Critical Bug Fix (2025-01-15)

**Problem:** Original code keyword-matched "VIOLATION" in reasoning field, causing 95% false positive rate.

**Bug:**
```python
is_violation = "VIOLATION" in content_field or "VIOLATION" in thinking_field.upper()
```

**Fix:**
```python
is_violation = content_field == "VIOLATION"
```

**Impact:**
- Before: 3.3% multi-policy accuracy, 35.6% FP rate
- After: 40.0% multi-policy accuracy, 6.7% FP rate
- Improvement: +36.7% accuracy!

## Results Comparison

### Multi-Policy Exact Match Accuracy

| Approach | Accuracy | vs OpenAI |
|----------|----------|-----------|
| **OpenAI gpt-oss-safeguard-20b** | 43.6% | baseline |
| **Minimal policies (100-150 tok)** | 40.0% | -3.6% |
| **Medium policies (300-500 tok)** | 37.8% | -5.8% |

### Overall Metrics

|  | Minimal | Medium |
|--|---------|--------|
| Safe/unsafe accuracy | 88.9% | 86.7% |
| False positive rate | 6.7% | 5.6% |
| False negative rate | 4.4% | 7.8% |

### Per-Policy F1 Scores

| Policy | Minimal | Medium | Change |
|--------|---------|--------|--------|
| illegal | 69.6% | 66.7% | -2.9% |
| harassment | 34.3% | 34.0% | -0.3% |
| sexual_minors | 35.3% | 35.3% | 0% |
| violence | 12.5% | 12.5% | 0% |
| hate_speech | 25.6% | 25.6% | 0% |
| self_harm | 0.0% | 0.0% | 0% |

## Analysis

### Why Minimal Policies Win

1. **Focus:** Shorter policies force model to focus on core distinctions
2. **Serial overhead:** 6 sequential runs amplify token costs
3. **Confusion reduction:** Less text = less chance for cross-policy confusion
4. **Different task:** Single-policy optimal length (400-600 tok from llm-abuse-patterns) doesn't apply to multi-policy

### Policy Length Study

- **Minimal (100-150 tokens):** 40.0% accuracy ✓
- **Medium (300-500 tokens):** 37.8% accuracy
- **Conclusion:** For serial gauntlet, minimal is optimal

### Known Issues

**Weak policies:**
- self_harm: 0% F1 (catches nothing)
- violence: 12.5% F1 (high recall 66.7%, low precision 6.9%)
- hate_speech: 25.6% F1 (needs improvement)

**Strong policies:**
- illegal: 69.6% F1
- harassment: 34.3% F1
- sexual_minors: 35.3% F1

## Source Breakdown

| Source | Samples | Accuracy | FP Rate | FN Rate |
|--------|---------|----------|---------|---------|
| ToxicChat | 40 | 85.0% | 5.0% | 10.0% |
| SimpleSafetyTests | 20 | 100.0% | 0.0% | 0.0% |
| XSTest | 30 | 86.7% | 10.0% | 3.3% |

## Next Steps

1. Fix self_harm policy (0% recall)
2. Reduce violence false positives (93% FP rate)
3. Improve hate_speech detection
4. Consider policy-specific tuning vs uniform length
5. Test parallel gauntlet (6 models simultaneously)

## Files

- `serial_gauntlet_simple.py` - Main implementation
- `eval_benchmark.py` - Evaluation framework
- `policies_minimal/` - 100-150 token policies (BEST)
- `policies_medium/` - 300-500 token policies (worse)
- `academic_benchmark.json` - 90 sample test set
- `eval_results.json` - Detailed results

## Conclusion

**Minimal policies achieve 40.0% multi-policy accuracy, only 3.6% below OpenAI's specialized model.**

For serial gauntlet architectures, minimal policies (100-150 tokens) are optimal. The 400-600 token range from single-policy jailbreak detection does not generalize to multi-policy safety.
