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

## Future Experiments

### 1. Tiered Caching System (Performance Optimization)

**Motivation:** Most violations are repetitive. Running full LLM checks on known bad content wastes compute.

**Architecture:**
```
L1: Hash Lookup (μs)
  ↓ miss
L2: Pattern/Embedding Match (ms)
  ↓ miss  
L3: Full LLM Check (seconds)
  ↓
Cache result for future hits
```

**Implementation:**
```python
violation_cache = {
    "exact": {},              # Exact string matches - O(1)
    "embeddings": FAISS,      # Semantic similarity - O(log n)
    "patterns": regex_db      # Keyword patterns - O(k)
}

def check_with_cache(content):
    # L1: Exact match
    if content in violation_cache["exact"]:
        return violation_cache["exact"][content]
    
    # L2: Embedding similarity (cosine > 0.95)
    similar = violation_cache["embeddings"].search(content, threshold=0.95)
    if similar:
        return similar.violations
    
    # L3: Full LLM gauntlet (cache miss)
    result = run_serial_gauntlet(content)
    
    # Cache for future
    violation_cache["exact"][content] = result
    violation_cache["embeddings"].add(content, result)
    
    return result
```

**Expected Performance:**
- 90%+ cache hit rate in production (repeated violations)
- L1 hits: <1ms (vs 12s LLM)
- L2 hits: <100ms (vs 12s LLM)
- 100-1000x speedup for common violations

**Tradeoffs:**
- Memory: Cache storage (millions of entries)
- Freshness: Stale cache if policies change
- Edge cases: Near-miss variations might slip through

**Evaluation:**
- Measure cache hit rate on real traffic
- Compare accuracy: cached vs fresh LLM decisions
- Benchmark latency distribution

### 2. Policy-Specific Length Optimization

**Motivation:** Current study used uniform lengths (all minimal or all medium). Each policy might have different optimal lengths.

**Hypothesis:**
- **illegal**: Complex (needs examples) → 400-600 tokens ✓
- **violence**: Simple (clear patterns) → 100-150 tokens ✓
- **hate_speech**: Context-heavy → 300-500 tokens ?
- **self_harm**: Currently broken (0% F1) → needs redesign
- **sexual_minors**: Working (35% F1) → keep minimal
- **harassment**: Context-heavy → 300-500 tokens ?

**Experiment:**
Test all 64 combinations of minimal/medium per policy (2^6), measure multi-policy accuracy for each.

**Expected Finding:**
Mixed approach beats uniform:
- Illegal (medium): 400 tokens
- Harassment (medium): 350 tokens  
- Hate_speech (medium): 300 tokens
- Violence (minimal): 120 tokens
- Sexual_minors (minimal): 100 tokens
- Self_harm: redesigned policy

Target: 45%+ multi-policy accuracy (beat OpenAI's 43.6%)

### 3. Parallel Gauntlet (Latency Optimization)

**Motivation:** Serial takes 12s per content. Production needs <2s response time.

**Implementation:**
- Load 6 model instances simultaneously
- Run all policy checks in parallel
- Aggregate results

**Requirements:**
- GPU: 6 × 20GB = 120GB VRAM
- Frameworks: asyncio, multiprocessing, or Ray

**Expected:**
- Latency: 12s → 2s (6x improvement)
- Accuracy: Same as serial (40%)
- Throughput: Higher (can process multiple contents simultaneously)

**Note:** Pure engineering optimization, no research value for accuracy experiments.

### 4. Self-Harm Policy Redesign

**Problem:** Current policy has 0% F1 (catches nothing)

**Root Cause Analysis Needed:**
- Is policy text unclear?
- Are test samples too subtle?
- Does model refuse to engage with suicide content?

**Approaches to test:**
1. More explicit examples in policy
2. Clinical framing ("mental health assessment")
3. Dedicated model fine-tuned for crisis detection
4. Ensemble: LLM + keyword patterns

**Success metric:** >30% F1 (match other policies)

### 5. Ensemble Methods

**Motivation:** Combine multiple approaches for higher accuracy.

**Architecture:**
```python
def ensemble_check(content):
    # Vote from multiple systems
    llm_result = run_serial_gauntlet(content)
    keyword_result = pattern_matcher.check(content)
    embedding_result = classifier.predict(content)
    
    # Weighted vote
    return weighted_majority(
        llm_result,      # weight: 0.6
        keyword_result,  # weight: 0.2
        embedding_result # weight: 0.2
    )
```

**Components:**
- LLM gauntlet (current system - 40% accuracy)
- Keyword patterns (fast, high precision, low recall)
- Embedding classifier (BERT fine-tuned on safety data)

**Target:** 50%+ multi-policy accuracy (ensemble > individual)

### 6. Active Learning Pipeline

**Motivation:** Learn from mistakes to improve policies.

**Pipeline:**
```
1. Run gauntlet on new content
2. Collect failures (wrong predictions)
3. Analyze failure patterns
4. Generate new policy examples
5. Re-evaluate
6. Iterate
```

**Implementation:**
- Track all wrong predictions
- Cluster similar failures (embeddings)
- Manual review of cluster representatives
- Update policy files with new examples
- Re-run evaluation

**Expected:** Continuous improvement over time as policy examples expand.

