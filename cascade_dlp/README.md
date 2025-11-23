# Cascade DLP

> Data Loss Prevention for LLM outputs - detecting sensitive data exfiltration in model responses.

## Overview

```
User Prompt                    Model Response
    │                               │
    ▼                               ▼
┌─────────────┐              ┌─────────────┐
│   Inbound   │              │  Outbound   │
│   Cascade   │              │    DLP      │
│  (refusals) │              │  (egress)   │
└─────────────┘              └─────────────┘
    │                               │
    ▼                               ▼
 Block harmful               Block sensitive
   requests                  data leakage
```

The inbound cascade prevents harmful prompts. The outbound DLP prevents sensitive data from leaking in responses.

## Threat Model

### Insider Risk
- Model trained on sensitive internal data
- Prompt injection extracts private information
- Unintentional disclosure of proprietary knowledge

### Data Exfiltration Categories

| Category | Examples | Risk Level |
|----------|----------|------------|
| **PII** | SSN, phone, email, address | HIGH |
| **Credentials** | API keys, passwords, tokens | CRITICAL |
| **Financial** | Credit cards, bank accounts | HIGH |
| **Medical** | PHI, diagnosis, prescriptions | HIGH |
| **Internal** | Employee names, internal docs | MEDIUM |
| **Code** | Proprietary source, algorithms | MEDIUM |

## Detection Approaches

### Option A: Regex/Pattern Matching (Fast)

```python
class PatternScanner:
    """Fast regex-based PII detection."""

    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "api_key": r"\b(sk-|pk-|api_)[a-zA-Z0-9]{20,}\b",
    }

    def scan(self, text: str) -> list[dict]:
        findings = []
        for category, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                findings.append({"category": category, "matches": matches})
        return findings
```

- **Latency**: <1ms
- **Accuracy**: High precision, low recall (misses novel patterns)

### Option B: NER Model (Balanced)

```python
class NERScanner:
    """Named Entity Recognition for PII."""

    def __init__(self):
        self.model = load_model("dslim/bert-base-NER")

    def scan(self, text: str) -> list[dict]:
        entities = self.model(text)
        return [e for e in entities if e["label"] in PII_LABELS]
```

- **Latency**: ~50ms
- **Accuracy**: Better recall, handles context

### Option C: LLM Classifier (Thorough)

```python
class LLMScanner:
    """LLM-based sensitive data detection."""

    def scan(self, text: str) -> dict:
        prompt = f"""Analyze this text for sensitive data:

{text}

Categories to check:
- PII (names, SSN, phone, email, address)
- Credentials (API keys, passwords, tokens)
- Financial (credit cards, bank accounts)
- Medical (PHI, diagnosis)
- Internal (employee info, proprietary)

Return JSON with findings."""

        return self.model.generate(prompt)
```

- **Latency**: ~500ms
- **Accuracy**: Best recall, understands context

## Cascade Architecture

```
Model Response
      │
      ▼
┌─────────────┐
│   Pattern   │  <1ms
│   Scanner   │
└──────┬──────┘
       │
  findings?
       │
   ┌───┴───┐
   │       │
  yes      no
   │       │
   ▼       ▼
┌──────┐ ┌──────┐
│ BLOCK│ │ NER  │  ~50ms
└──────┘ │Scan  │
         └──┬───┘
            │
       findings?
            │
        ┌───┴───┐
        │       │
       yes      no
        │       │
        ▼       ▼
    ┌──────┐  ALLOW
    │ BLOCK│
    └──────┘
```

## Response Actions

| Finding | Action | User Message |
|---------|--------|--------------|
| PII detected | BLOCK | "I've removed some personal information from my response for privacy protection." |
| Credentials | BLOCK | "I've redacted sensitive credentials. Please never share API keys or passwords." |
| Uncertain | REDACT | Replace with `[REDACTED]` |
| Clean | ALLOW | Pass through |

## Datasets

### Existing Resources

1. **PII Detection Datasets**
   - ai4privacy/pii-masking-200k
   - presidio test data

2. **Secret Detection**
   - GitHub secret scanning patterns
   - TruffleHog patterns

3. **Medical (PHI)**
   - i2b2 de-identification
   - MIMIC-III notes

### Synthetic Generation

Generate test cases with:
- Embedded PII in benign responses
- Edge cases (partial SSN, formatted numbers)
- False positives (phone numbers in code)

## Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| PII Recall | >99% | Critical - miss = breach |
| Precision | >95% | Minimize false blocks |
| Latency p99 | <100ms | Production viable |
| Credential Recall | 100% | Zero tolerance |

## Integration

```python
from cascade_dlp import DLPScanner

scanner = DLPScanner()

# After model generates response
response = model.generate(prompt)

# Scan for sensitive data
result = scanner.scan(response)

if result["findings"]:
    if result["severity"] == "CRITICAL":
        response = "I can't share that information."
    else:
        response = scanner.redact(response)

return response
```

## Directory Structure

```
cascade_dlp/
├── README.md              # This file
├── data/
│   ├── patterns/          # Regex patterns
│   ├── test_cases/        # Evaluation data
│   └── false_positives/   # Known FPs
├── models/
│   ├── ner/               # NER models
│   └── classifier/        # LLM classifier
├── src/
│   ├── __init__.py
│   ├── pattern_scanner.py
│   ├── ner_scanner.py
│   ├── llm_scanner.py
│   └── redactor.py
└── tests/
    └── test_dlp.py
```

## Compliance Considerations

- **GDPR** - PII detection and redaction
- **HIPAA** - PHI protection
- **PCI-DSS** - Credit card masking
- **SOC 2** - Audit logging of detections

## Next Steps

1. **Immediate**: Define pattern library for common PII
2. **Week 1**: Implement pattern scanner baseline
3. **Week 2**: Add NER model layer
4. **Week 3**: Evaluation on test dataset
5. **Month 1**: Production integration

---

*Created: 2024-11-23*
