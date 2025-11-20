# Multi-Policy Safety Gauntlet - Research Summary

## TL;DR

Built multi-policy content moderation system. **Key finding: Fixed parsing bug improved accuracy by 36.7%**. Minimal policies (100-150 tokens) beat longer ones. Beat OpenAI baseline on balanced test.

## What Worked

**1. Critical Bug Fix (+36.7% accuracy)**
```python
# BEFORE (wrong):
is_violation = "VIOLATION" in thinking_field  # Keyword matching on reasoning

# AFTER (correct):  
is_violation = content_field == "VIOLATION"  # Only check decision field
```

**Impact:** 3.3% → 40.0% multi-policy accuracy

**2. Minimal Policies Win**
- 100-150 tokens: 40.0% accuracy ✓
- 300-500 tokens: 37.8% accuracy
- Why: Serial runs amplify token overhead, shorter = more focused

**3. Self-Harm Fix**
- Old benchmark: 0 samples → 0% F1 (meaningless)
- Manual test: 100% accuracy (policy works!)
- Balanced benchmark: 80% F1

**4. Beat OpenAI Baseline**
- OpenAI gpt-oss-safeguard-20b: 43.6%
- Our minimal policies: 61.1%
- On balanced 90-sample benchmark

## Architecture

**Serial Gauntlet:**
- 1 model (gpt-oss:20b)
- 6 policies (hate, violence, self-harm, sexual_minors, harassment, illegal)
- Sequential: ~12s per content
- Minimal policies: 100-150 tokens each

## Experiments

| Experiment | Finding |
|------------|---------|
| Policy length | Minimal > Medium |
| Academic benchmark | Flawed (missing self-harm) |
| Balanced benchmark | 61.1% multi-policy accuracy |
| Parsing bug | +36.7% improvement |

## Results

**Balanced Benchmark (90 samples):**
- Multi-policy exact match: 61.1%
- Overall safe/unsafe: 98.9%
- False positive rate: 0.0%
- All policies: 100% recall

**Per-Policy F1:**
- self_harm: 80.0%
- sexual_minors: 73.7%
- hate_speech: 69.6%
- illegal: 63.2%
- violence: 50.0%
- harassment: 50.0%

## What We Learned

1. **Parsing bugs dominate** - 36.7% improvement from one line
2. **Your debugging was right** - "Ask questions, examine outputs"
3. **Minimal is better** - For serial gauntlet, 100-150 tokens optimal
4. **Sample size matters** - But 90 is fine for toy/exploration
5. **Benchmark quality > size** - Balanced 90 > imbalanced 90

## Code

**Main files:**
- `serial_gauntlet_simple.py` - Implementation
- `policies_minimal/` - Best policies (100-150 tok)
- `balanced_benchmark.json` - 90 test samples

**Repo:** https://github.com/bigsnarfdude/wizard101

## What's Next (if someone cares)

1. Get WildGuardMix (92K samples)
2. Test policy-specific lengths (illegal=500 tok, violence=100 tok)
3. Implement caching (hash → embeddings → LLM)
4. Parallel gauntlet (12s → 2s)

---

**Status:** Research toy complete ✅

Built it, fixed bugs, learned stuff. Sample size is fine for exploration. Move on.
