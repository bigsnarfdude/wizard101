# Project: cascade_dlp - Outbound Data Loss Prevention for LLM Systems

## Context
We're a Tier 1 AI safety team building a multi-stage outbound DLP cascade. Our inbound pipeline already has:
- 95% fast classification (Tier 1 classifier)
- Semantic refusal response router

Now we need to catch sensitive data going OUT. Current approach is keyword matching - we need something smarter.

## Architecture Goal
Build a prototype cascade_dlp system inspired by CaMeL (Google DeepMind) that handles BOTH:
1. **Adversarial exfiltration** - prompt injection attacks trying to extract data
2. **Accidental leakage** - model regurgitating memorized training data, PII, secrets

## Core Components to Prototype

### 1. Data Provenance Tracker (`provenance.py`)
Track metadata for every value flowing through the system:
```python
@dataclass
class DataProvenance:
    sources: Set[str]  # Where did this data come from?
    allowed_readers: Set[str]  # Who can see this?
    sensitivity_level: str  # public, internal, confidential, restricted
    data_types: Set[str]  # pii, credential, code, text, etc.
```

### 2. Output Detectors (`detectors/`)
Modular detectors that can be composed in cascade:

- `pii_detector.py` - Names, emails, SSNs, phone numbers, addresses
  - Use NER approach (spaCy/transformers) + regex fallback
  - Reference: PIILO dataset patterns, BigCode StarPII categories

- `secret_detector.py` - API keys, passwords, tokens, private keys
  - Reference: SecretBench categories (15,084 verified secrets across 8 types)
  - Patterns: AWS keys, GitHub tokens, JWT, private keys, connection strings

- `memorization_detector.py` - Training data regurgitation
  - Perplexity-based anomaly detection
  - Compare output against reference model
  - Flag unusually confident/verbatim sequences

- `exfiltration_detector.py` - Adversarial extraction attempts
  - Detect encoding tricks (base64, hex, leetspeak)
  - URL/endpoint extraction in responses
  - Unusual data flow patterns

### 3. Policy Engine (`policies.py`)
Security policies as composable functions:
```python
def check_output_policy(
    output: str,
    provenance: DataProvenance,
    detections: List[Detection],
    context: RequestContext
) -> PolicyResult:
    # Returns: Allowed, Denied(reason), or RequiresReview
```

### 4. Cascade Orchestrator (`cascade.py`)
Multi-stage filtering with early exit:
```
Stage 1: Fast regex/bloom filter (microseconds)
    ↓ flagged
Stage 2: NER-based PII/secret detection (milliseconds)
    ↓ flagged
Stage 3: Semantic analysis - is this actually sensitive in context? (10s of ms)
    ↓ flagged
Stage 4: Provenance check - should this data flow to this recipient? (ms)
    ↓ denied
Block or require human review
```

### 5. Evaluation Framework (`eval/`)
Build test datasets combining:

**Adversarial tests:**
- Prompt injection attempts to extract system prompt
- Encoding-based exfiltration (base64, rot13, etc.)
- Multi-turn extraction attacks
- Indirect injection via tool outputs

**Accidental leakage tests:**
- Canary strings inserted at known frequencies
- Synthetic PII in various formats
- Fake credentials that should never appear
- Code snippets that shouldn't be reproducible

**False positive tests:**
- Legitimate outputs that look like PII (fictional names, example.com emails)
- Code examples with placeholder credentials
- Documentation discussing security concepts

## File Structure
```
cascade_dlp/
├── src/
│   ├── provenance.py
│   ├── cascade.py
│   ├── policies.py
│   └── detectors/
│       ├── base.py
│       ├── pii_detector.py
│       ├── secret_detector.py
│       ├── memorization_detector.py
│       └── exfiltration_detector.py
├── eval/
│   ├── datasets/
│   │   ├── adversarial/
│   │   ├── accidental/
│   │   └── false_positives/
│   ├── benchmarks.py
│   └── metrics.py
├── tests/
└── README.md
```

## Key Metrics to Track
- **Detection Rate** (recall) - % of actual leaks caught
- **False Positive Rate** - % of legitimate outputs incorrectly blocked
- **Latency** - p50, p95, p99 per stage
- **Attack Success Rate** - for adversarial test suite

## Start With
1. Scaffold the project structure
2. Implement `DataProvenance` and basic `Detection` types
3. Build `secret_detector.py` first (most concrete patterns from SecretBench)
4. Create initial test dataset with synthetic secrets
5. Wire up basic cascade with timing instrumentation

## References
- CaMeL paper: arxiv.org/abs/2503.18813 (capability-based security for LLM agents)
- SecretBench: github.com/setu1421/SecretBench (15k verified secrets)
- PIILO: PII detection dataset (22k annotated samples)
- Lakera PINT: Prompt injection benchmark (4,314 samples)

## Constraints
- Python 3.11+
- Keep dependencies minimal for now (just spacy, transformers, pydantic)
- Design for easy A/B testing of detector configurations
- All detectors must return confidence scores, not just binary

---

## Implementation Status

### Completed
- [x] L0 baseline: Presidio (96.2% F1, 4.8ms)
- [x] Research landscape documented
- [x] Architecture decision: 120B for L1 (breach cost justifies compute)
- [x] Purview enterprise features documented

### Next Steps
- [ ] Scaffold project structure
- [ ] Implement DataProvenance and Detection types
- [ ] Build secret_detector.py with SecretBench patterns
- [ ] Create synthetic test dataset
- [ ] Wire up cascade with timing

---

*Spec created: 2025-11-23*
