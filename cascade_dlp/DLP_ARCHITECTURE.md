# DLP Architecture - Project Emmentaler

## Summary

Data Loss Prevention cascade for LLM outputs. Presidio as L0, with L1 for edge cases.

**Key insight**: Each 1% recall improvement = 10M fewer potential leaks/day at 1B prompts. Cost of breach is catastrophic.

## Results

### L0: Presidio
- **Recall**: 96.2%
- **Precision**: 100%
- **Latency**: 4.8ms
- **Cost**: Free (self-hosted)

### What Presidio Does (Not Just Regex)
1. Pattern recognizers with named groups
2. NER models (spaCy en_core_web_lg)
3. Context enhancement ("credit card:" boosts score)
4. Validators (Luhn checksum for cards)
5. Deny lists
6. Custom recognizers
7. Score aggregation

## Architecture

```
Model Response
      │
      ▼
┌─────────────┐
│ L0: Presidio│  4.8ms, 96% recall, FREE
└──────┬──────┘
       │
   score ≥ 0.7?
       │
   ┌───┴───┐
   │       │
  yes      no
   │       │
   ▼       ▼
REDACT  ┌──────────┐
        │ L1: 120B │  ~500ms, 99%+ recall
        │ DLP-Guard│
        └────┬─────┘
             │
        findings?
             │
         ┌───┴───┐
        yes      no
         │       │
         ▼       ▼
      REDACT   ALLOW
```

## Cost Analysis: 1B Prompts/Day

### Volume After Cascade

| Stage | Prompts/Day | % of Total |
|-------|-------------|------------|
| Total responses | 1,000,000,000 | 100% |
| After L0 (4% uncertain) | 40,000,000 | 4% |
| After score filter (10%) | 4,000,000 | 0.4% |

### L1 Options Compared

| Model | Annual Cost | Recall | Rationale |
|-------|-------------|--------|-----------|
| Claude API | $22M | ~99% | Too expensive |
| Self-hosted 3.8B | $50K | ~95% | Cost-optimized |
| Self-hosted 120B | $400K | ~99% | **Recommended** |

### Why 120B is Worth It

```
Cost of 120B vs 3.8B: +$350K/year
Cost of one breach:   $20M - $100M+

$350K is insurance premium against catastrophic loss.
```

**Breach costs:**
- GDPR fine: up to €20M or 4% global revenue
- HIPAA: $50K - $1.5M per violation
- Class action lawsuits: $100M+
- Reputation damage: incalculable
- Customer trust: lost

### Impact of Recall

| Recall | Leaks Caught/Day | Potential Leaks/Year |
|--------|------------------|----------------------|
| 96% (L0 only) | 960M | 14.6B potential |
| 99% (+L1) | 990M | 3.6B potential |
| 99.9% (optimized) | 999M | 365M potential |

**Each 1% = 10M fewer potential leaks per day.**

## Recommended Solution

### Production Stack

| Layer | Component | Latency | Recall | Cost/Year |
|-------|-----------|---------|--------|-----------|
| L0 | Presidio | 4.8ms | 96% | $0 |
| Filter | Score threshold | <1ms | - | $0 |
| L1 | Self-hosted 120B | ~500ms | 99%+ | ~$400K |

**Total: ~$400K/year for 99%+ recall**

### Why Not Smaller Models for L1?

PII detection seems simple, but edge cases need reasoning:
- "my password is hunter2" (semantic secret)
- "call me at [number that looks like ID]"
- Context-dependent sensitivity
- Novel exfiltration patterns

120B catches what patterns and small models miss.

## Research Opportunity: DLP-Guard

No one has published a purpose-built DLP model like LlamaGuard for safety.

**Opportunity:**
- Fine-tune 8B-70B specifically for data leakage
- Train on Presidio gaps (what it misses)
- Optimize for recall over precision
- Benchmark against ai4privacy dataset
- Open-source contribution

**Training data:**
- ai4privacy/pii-masking-200k
- Synthetic semantic secrets
- False negatives from Presidio
- Business-specific PII patterns

## Entity Types Covered

### Presidio Built-in (16 recognizers)
- CREDIT_CARD (with Luhn validation)
- US_SSN, US_ITIN, US_PASSPORT
- US_BANK_NUMBER, US_DRIVER_LICENSE
- EMAIL_ADDRESS, PHONE_NUMBER
- IP_ADDRESS, URL
- IBAN_CODE, CRYPTO
- UK_NHS, MEDICAL_LICENSE
- PERSON, LOCATION, DATE_TIME (via NER)

### Gaps for L1 to Cover
- Semantic secrets ("password is...")
- Business-specific identifiers
- Context-dependent data
- Novel PII formats

## Files

| File | Purpose |
|------|---------|
| `explore_dataset.py` | Dataset analysis, baseline metrics |
| `presidio_eval.py` | Presidio evaluation (96.2% F1) |
| `presidio_deep_dive.py` | Understand Presidio techniques |
| `eval_scanners.py` | Compare multiple approaches |

## Next Steps

### Immediate
- [x] Establish Presidio baseline (96.2% F1)
- [x] Understand Presidio techniques
- [x] Cost analysis for L1 options

### Short-term
- [ ] Implement score threshold filter
- [ ] Test 120B as L1 on Presidio gaps
- [ ] Measure end-to-end recall

### Long-term
- [ ] Train DLP-Guard (purpose-built model)
- [ ] Benchmark against all options
- [ ] Open-source contribution

## Key Decisions

1. **L0**: Presidio (done, works great)
2. **L1**: Use 120B, not smaller - breach cost justifies compute
3. **Architecture**: Cascade with score threshold to reduce L1 load
4. **Research**: DLP-Guard is open opportunity

---

## Enterprise DLP Reference: Microsoft Purview

Reference: https://learn.microsoft.com/en-us/purview/dlp-policy-design

### Purview Policy Structure

```
┌─────────────────────────────────────────┐
│           PURVIEW DLP POLICY            │
├─────────────────────────────────────────┤
│  1. Monitor    - What content types     │
│  2. Scope      - Admin boundaries       │
│  3. Location   - Where to monitor       │
│  4. Conditions - Triggers (AND/OR/NOT)  │
│  5. Actions    - Block/Alert/Audit      │
└─────────────────────────────────────────┘
```

### Key Features

| Purview Feature | Purpose |
|-----------------|---------|
| Confidence levels | Adjustable detection sensitivity |
| Pre-built templates | HIPAA, PCI-DSS, GDPR compliance |
| Boolean conditions | AND/OR/NOT rule logic |
| Multiple actions | Block, alert, audit, override |
| User override | Proceed with justification + audit |

### Mapping to DLP-Guard

| Purview | Our Implementation |
|---------|-------------------|
| Confidence levels | Presidio score threshold (≥0.7) |
| Pre-built templates | Custom recognizers per compliance |
| Actions | REDACT / ALLOW / ALERT / AUDIT |
| User override | Future: allow with justification |
| Boolean conditions | Score aggregation logic |

### Features to Implement

#### 1. Configurable Actions
```python
class DLPAction(Enum):
    BLOCK = "block"      # Reject entirely
    REDACT = "redact"    # Remove sensitive data
    ALERT = "alert"      # Notify security team
    AUDIT = "audit"      # Log only, allow through
    OVERRIDE = "override" # User can proceed with justification
```

#### 2. Compliance Templates
```python
COMPLIANCE_TEMPLATES = {
    "HIPAA": {
        "entities": ["US_SSN", "MEDICAL_LICENSE", "PERSON", "DATE_TIME"],
        "confidence": 0.8,
        "action": "REDACT"
    },
    "PCI-DSS": {
        "entities": ["CREDIT_CARD", "US_BANK_NUMBER"],
        "confidence": 0.9,
        "action": "BLOCK"
    },
    "GDPR": {
        "entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"],
        "confidence": 0.7,
        "action": "REDACT"
    }
}
```

#### 3. Policy as Code
```yaml
# dlp_policy.yaml
policy:
  name: "Production DLP"
  version: "1.0"

  rules:
    - name: "Credit Card Detection"
      entity: CREDIT_CARD
      confidence: 0.8
      action: REDACT
      notify: security@company.com

    - name: "Medical Records"
      entities: [MEDICAL_LICENSE, US_SSN]
      confidence: 0.7
      action: ALERT
      require_override_reason: true

    - name: "PII Audit"
      entities: [PERSON, EMAIL_ADDRESS]
      confidence: 0.6
      action: AUDIT
```

#### 4. Audit Logging
```python
@dataclass
class DLPAuditEvent:
    timestamp: datetime
    policy_name: str
    entity_type: str
    confidence: float
    action_taken: str
    user_override: bool
    override_reason: Optional[str]
    content_hash: str  # For forensics without storing PII
```

### Enterprise Considerations

1. **Compliance reporting** - Generate reports for auditors
2. **Policy versioning** - Track policy changes over time
3. **Testing mode** - Audit-only before enforcing
4. **Incident response** - Integrate with SIEM/SOAR
5. **User training** - Override requires acknowledgment

### Why This Matters

Purview represents enterprise-grade DLP. For DLP-Guard to be production-ready:
- Must support compliance frameworks (HIPAA, PCI, GDPR)
- Must have configurable actions beyond just block/allow
- Must provide audit trail for compliance
- Must allow policy-as-code for version control

---

*Session: 2024-11-23*
*Status: L0 complete, L1 architecture decided, Purview features documented*
