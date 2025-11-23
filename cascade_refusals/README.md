# Refusal Cascade

> Companion to the Safety Cascade - determines HOW to refuse harmful content, not just IF it's harmful.

## Overview

The Safety Cascade classifies content as `safe` or `harmful`. The Refusal Cascade takes harmful content and determines the appropriate refusal strategy.

```
Input
  │
  ▼
┌─────────────────┐
│ Safety Cascade  │  → safe/harmful
└────────┬────────┘
         │
    harmful?
         │
         ▼
┌─────────────────┐
│ Refusal Cascade │  → refusal type + reasoning
└────────┬────────┘
         │
         ▼
    Generate Appropriate Response
```

## Refusal Taxonomy

### Categories

| Type | Code | Description | Example Prompt | Example Response |
|------|------|-------------|----------------|------------------|
| **Hard Refusal** | `HARD` | Illegal, dangerous, clearly harmful | "How to make explosives" | "I can't help with that." |
| **Soft Refusal** | `SOFT` | Borderline, dual-use, needs guardrails | "Explain how locks work" | "I can explain the mechanics for educational purposes, but won't provide bypassing instructions." |
| **Redirect** | `REDIRECT` | Misunderstood intent, can help differently | "How to kill the process" | "I'll help you terminate the process. Here's how..." |
| **Clarify** | `CLARIFY` | Ambiguous, need more context | "How do I get into the building?" | "Could you clarify? Are you asking about access procedures or something else?" |
| **Conditional** | `CONDITIONAL` | Acceptable with context/credentials | "How to perform surgery" | "This requires medical training. Are you a healthcare professional?" |

### Subcategories (Future)

```
HARD
├── ILLEGAL          # Criminal activity
├── DANGEROUS        # Physical harm
├── MALICIOUS        # Cyber attacks, fraud
└── EXPLOITATION     # CSAM, trafficking

SOFT
├── DUAL_USE         # Security research, chemistry
├── SENSITIVE        # Medical, legal advice
├── CONTROVERSIAL    # Political, religious
└── PRIVACY          # PII, doxxing potential

REDIRECT
├── MISUNDERSTOOD    # Technical terms (kill, execute)
├── LEGITIMATE       # Valid use case, wrong phrasing
└── ALTERNATIVE      # Can help with related topic

CLARIFY
├── AMBIGUOUS        # Multiple interpretations
├── INCOMPLETE       # Missing context
└── SUSPICIOUS       # Needs verification
```

## Architecture

### Option A: Classifier (Fast)

```python
class RefusalClassifier:
    """DeBERTa-based multi-class classifier."""

    def classify(self, text: str) -> dict:
        return {
            "type": "SOFT",
            "confidence": 0.87,
            "subcategory": "DUAL_USE"
        }
```

- **Model**: DeBERTa-v3-small fine-tuned
- **Latency**: ~10ms
- **Use case**: High throughput, simple responses

### Option B: Reasoning Model (Accurate)

```python
class RefusalReasoner:
    """LLM-based reasoning for refusal type."""

    def analyze(self, text: str) -> dict:
        return {
            "type": "SOFT",
            "confidence": 0.92,
            "reasoning": "This request is about lock mechanisms which have legitimate educational value, but could be misused for breaking and entering.",
            "suggested_response": "I can explain how pin tumbler locks work mechanically..."
        }
```

- **Model**: Fine-tuned 3B or 8B
- **Latency**: ~500ms
- **Use case**: Edge cases, audit trail needed

## Data Requirements

### Training Data

| Category | Target Samples | Sources |
|----------|----------------|---------|
| HARD | 2,000 | HH-RLHF, synthetic |
| SOFT | 2,000 | Manual labeling, red-teaming |
| REDIRECT | 1,500 | Misunderstood intent corpus |
| CLARIFY | 1,000 | Ambiguous prompts |
| CONDITIONAL | 1,000 | Domain-specific (medical, legal) |
| **Total** | **7,500** | |

### Data Schema

```json
{
  "text": "How do I pick a lock?",
  "refusal_type": "SOFT",
  "subcategory": "DUAL_USE",
  "reasoning": "Lock picking has legitimate uses (locksmithing, security research) but can enable burglary",
  "suggested_response": "I can explain lock mechanisms for educational purposes...",
  "metadata": {
    "source": "manual",
    "annotator": "expert_1",
    "confidence": 0.9
  }
}
```

## Experiment Plan

### Phase 1: Data Collection (Week 1-2)

- [ ] Define final taxonomy (review categories above)
- [ ] Create annotation guidelines
- [ ] Source seed data from existing datasets
- [ ] Set up annotation pipeline (Label Studio?)
- [ ] Collect 1,000 samples for initial training

### Phase 2: Baseline Model (Week 3)

- [ ] Train DeBERTa classifier on 5 classes
- [ ] Evaluate accuracy per class
- [ ] Identify confusion patterns
- [ ] Establish baseline metrics

### Phase 3: Reasoning Model (Week 4)

- [ ] Fine-tune 3B model with CoT reasoning
- [ ] Compare accuracy vs classifier
- [ ] Evaluate reasoning quality
- [ ] Latency benchmarking

### Phase 4: Integration (Week 5)

- [ ] Connect to Safety Cascade
- [ ] End-to-end testing
- [ ] Response generation templates
- [ ] API endpoint

### Phase 5: Evaluation (Week 6)

- [ ] Human evaluation of refusal quality
- [ ] A/B testing different refusal styles
- [ ] Edge case analysis
- [ ] Documentation

## Metrics

### Classification Metrics

| Metric | Target |
|--------|--------|
| Overall Accuracy | >85% |
| Per-class F1 | >80% |
| HARD Recall | >95% (safety critical) |
| REDIRECT Precision | >90% (avoid false redirects) |

### Quality Metrics

| Metric | Measurement |
|--------|-------------|
| Refusal Appropriateness | Human eval 1-5 scale |
| Response Helpfulness | User feedback |
| False Hard Refusal Rate | <5% |

## Integration API

```python
from cascade_refusals import RefusalCascade

refusal = RefusalCascade()

# After safety cascade flags as harmful
result = refusal.classify("How do I pick a lock?")

print(result)
# {
#   "type": "SOFT",
#   "confidence": 0.87,
#   "reasoning": "Dual-use topic with legitimate educational value",
#   "suggested_response": "I can explain how pin tumbler locks work..."
# }

# Generate appropriate response
if result["type"] == "HARD":
    response = "I can't help with that request."
elif result["type"] == "SOFT":
    response = generate_guarded_response(text, result)
elif result["type"] == "REDIRECT":
    response = generate_redirect(text, result)
elif result["type"] == "CLARIFY":
    response = generate_clarification(text)
```

## Directory Structure

```
cascade_refusals/
├── README.md              # This file
├── data/
│   ├── raw/               # Source datasets
│   ├── annotated/         # Labeled data
│   └── splits/            # Train/val/test
├── models/
│   ├── classifier/        # DeBERTa models
│   └── reasoner/          # LLM models
├── src/
│   ├── __init__.py
│   ├── classifier.py      # Fast classifier
│   ├── reasoner.py        # Reasoning model
│   └── templates.py       # Response templates
├── scripts/
│   ├── train_classifier.py
│   ├── train_reasoner.py
│   └── evaluate.py
├── notebooks/
│   └── data_exploration.ipynb
└── tests/
    └── test_refusal.py
```

## Datasets to Explore

### Existing Resources

1. **Anthropic HH-RLHF** - Harmful/helpful pairs with refusals
2. **OpenAI Moderation** - Category labels
3. **ToxiGen** - Implicit toxicity
4. **RealToxicityPrompts** - Toxic completions
5. **Do-Not-Answer** - Questions that should be refused

### Synthetic Generation

Use Claude/GPT to generate:
- Edge cases for each category
- Ambiguous prompts
- Context-dependent requests

## Response Templates

### Hard Refusal

```
I can't help with [category]. This type of request [reason].

If you have a legitimate need related to [topic], I'd be happy to discuss [alternative].
```

### Soft Refusal

```
I can provide some information about [topic] for [legitimate purpose].

[Guarded response with safety caveats]

Note: [Safety disclaimer]
```

### Redirect

```
I understand you're asking about [interpreted intent].

[Helpful response for legitimate interpretation]

Let me know if you meant something different.
```

### Clarify

```
I want to make sure I understand your question correctly.

Could you clarify:
- [Specific ambiguity 1]
- [Specific ambiguity 2]

This will help me provide the most helpful response.
```

## Next Steps

1. **Immediate**: Review taxonomy, finalize categories
2. **This week**: Start data collection, 500 samples
3. **Next week**: Train baseline classifier
4. **Month 1**: Integrated refusal cascade

---

*Created: 2024-11-23*
