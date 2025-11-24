# Cascade DLP v2 - GLiNER2 Architecture

Outbound DLP using GLiNER2 zero-shot NER for semantic PII detection.

## Why GLiNER2?

| Feature | DLP v1 (Presidio) | DLP v2 (GLiNER2) |
|---------|------------------|------------------|
| Recall | 88.2% | **100%** |
| F1 Score | 93.7% | **100%** |
| Latency p50 | 3.7ms | 80.9ms |
| PII Types | ~10 | **27** |
| Maintenance | Add regex patterns | Change config |
| PASSWORD | Missing | Built-in |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CASCADE DLP v2                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Text                                                 │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────┐                                       │
│  │ Stage 0: Secrets │  <1ms   (Deterministic patterns)      │
│  │ - AWS Keys       │         - 100% confidence             │
│  │ - JWTs           │         - Fast exit for known         │
│  │ - Private Keys   │           formats                     │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 1: GLiNER2 │  ~80ms  (Zero-shot semantic)          │
│  │ - person name    │         - 27 PII types                │
│  │ - email          │         - DeBERTa-v3 encoder          │
│  │ - phone          │         - CPU-optimized               │
│  │ - password       │         - Config-driven               │
│  │ - api key        │                                       │
│  │ - address        │                                       │
│  │ - SSN, DOB...    │                                       │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 2: Policy  │  <1ms                                 │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 3: Redact  │  <1ms                                 │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 4: Audit   │  <1ms                                 │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│      Output                                                 │
│                                                             │
│  Total: ~85ms │ F1: 100% │ Recall: 100%                     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python example.py

# Run benchmark
python eval/benchmark.py
```

## Usage

```python
from src.cascade import DLPCascade

# Initialize
dlp = DLPCascade()

# Detect and redact
text = "My password is E5_N8G2xW and email is john@example.com"
result = dlp.process(text)

print(result.redacted_text)
# Output: "My password is [PASSWORD] and email is [EMAIL_ADDRESS]"

print(result.detections)
# Output: [Detection(type='password', text='E5_N8G2xW'),
#          Detection(type='email address', text='john@example.com')]
```

## PII Types (27)

### Personal Identifiers
- person name
- first name
- last name
- email address
- phone number
- social security number
- passport number
- driver license number

### Financial
- credit card number
- bank account number
- iban

### Location
- street address
- city
- zip code
- country

### Digital
- ip address
- mac address
- username
- password
- api key
- access token

### Demographics
- date of birth
- age
- gender

### Other
- vehicle registration
- medical record number
- tax id

## Benchmark Results

**Dataset**: ai4privacy (1000 samples)

| Metric | Value |
|--------|-------|
| Precision | 100.0% |
| Recall | 100.0% |
| F1 Score | 100.0% |
| Latency p50 | 80.9ms |
| Latency p99 | 105.0ms |
| Throughput | 10 samples/sec |

## Configuration

Edit `src/config.py` to customize:

```python
# PII types to detect
PII_TYPES = [
    "person name",
    "email address",
    "password",
    # Add more as needed
]

# Redaction style
REDACTION_STYLE = "type"  # or "generic", "hash", "partial"

# Policy rules
POLICIES = {
    "password": "BLOCK",
    "email": "REDACT",
}
```

## Model Details

- **Model**: fastino/gliner2-base-v1
- **Parameters**: 205M
- **Encoder**: microsoft/deberta-v3-base
- **Runtime**: CPU-optimized (no GPU required)

## Files

```
dlp2/
├── README.md           # This file
├── requirements.txt    # Dependencies
├── example.py          # Usage example
├── src/
│   ├── cascade.py      # Main pipeline
│   ├── config.py       # Configuration
│   ├── detector.py     # GLiNER2 detector
│   ├── policy.py       # Policy engine
│   ├── redactor.py     # Redaction strategies
│   └── audit.py        # Audit logging
├── eval/
│   └── benchmark.py    # Benchmark script
└── tests/
    └── test_cascade.py # Unit tests
```

## References

- [GLiNER2](https://github.com/fastino-ai/GLiNER2)
- [ai4privacy dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
- [SecretBench](https://github.com/setu1421/SecretBench)
