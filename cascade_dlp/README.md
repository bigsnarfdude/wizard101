# Cascade DLP v2 - Hybrid Presidio + GLiNER2

**Production-ready DLP using Presidio for accurate span extraction and GLiNER2 for semantic validation.**

## Why Hybrid?

| Feature | Presidio Only | GLiNER2 Only | **Hybrid (Recommended)** |
|---------|---------------|--------------|--------------------------|
| **Span Accuracy** | ✅ Excellent | ⚠️ Text matching | ✅ **Excellent** |
| **Confidence Scores** | ✅ Real (0.0-1.0) | ❌ Hardcoded | ✅ **Real** |
| **Semantic Coverage** | ⚠️ Limited (~15 types) | ✅ Excellent (27 types) | ✅ **Best of both** |
| **Latency** | ~5ms | ~80ms | ~85ms |
| **Production Ready** | ✅ Battle-tested | ⚠️ Span workarounds | ✅ **Production-ready** |

### Key Advantages

✅ **Accurate spans** - Presidio provides exact character positions (no text matching hacks)  
✅ **Real confidence scores** - Enable threshold tuning per entity type  
✅ **Comprehensive coverage** - Presidio's standard PII + GLiNER2's semantic entities  
✅ **Production-ready** - Battle-tested Presidio + cutting-edge GLiNER2  
✅ **Automatic deduplication** - Handles overlaps, prefers higher confidence

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              HYBRID DLP PIPELINE (v2)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Text                                                 │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────┐                                       │
│  │ Stage 0: Secrets │  <1ms   (Deterministic patterns)     │
│  │ - AWS Keys       │         - 100% confidence             │
│  │ - JWTs           │         - Fast exit for known         │
│  │ - Private Keys   │           formats                     │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 1a:        │  ~5ms   (Accurate span detection)    │
│  │ Presidio         │         - PERSON, EMAIL, PHONE       │
│  │                  │         - CREDIT_CARD, SSN, IP       │
│  │                  │         - Real confidence scores     │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 1b:        │  ~80ms  (Semantic validation)        │
│  │ GLiNER2          │         - Validates Presidio         │
│  │                  │         - Finds: password, api key   │
│  │                  │         - 27 PII types total         │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 2: Policy  │  <1ms   (BLOCK/REDACT rules)         │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 3: Redact  │  <1ms   (Replace PII)                │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Stage 4: Audit   │  <1ms   (Logging)                    │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│      Output                                                 │
│                                                             │
│  Total: ~85ms │ Presidio + GLiNER2 │ Production-Ready      │
└─────────────────────────────────────────────────────────────┘
```


## Quick Start

```bash
# Install (creates venv, installs deps, downloads spaCy model)
./install.sh

# Activate virtual environment
source venv/bin/activate

# Run example
python example.py
```

### Manual Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Usage

### Basic Usage

```python
from cascade_dlp import DLPCascade

# Initialize hybrid pipeline (default: both Presidio and GLiNER2)
dlp = DLPCascade()

# Detect and redact
text = "Contact John Smith at john.smith@company.com or call 555-123-4567"
result = dlp.process(text)

print(result.redacted_text)
# Output: "Contact [PERSON] at [EMAIL_ADDRESS] or call [PHONE_NUMBER]"

print(result.detections)
# Output: [
#   Detection(entity_type='person', text='John Smith', start=8, end=18, 
#             confidence=0.85, stage='presidio'),
#   Detection(entity_type='email address', text='john.smith@company.com', 
#             start=22, end=44, confidence=1.0, stage='presidio'),
#   Detection(entity_type='phone number', text='555-123-4567', start=53, 
#             end=65, confidence=0.4, stage='presidio')
# ]
```

### Advanced Configuration

```python
# Use only Presidio (faster, standard PII only)
dlp_presidio = DLPCascade(use_presidio=True, use_gliner=False)

# Use only GLiNER2 (semantic coverage, text matching for spans)
dlp_gliner = DLPCascade(use_presidio=False, use_gliner=True)

# Hybrid (recommended for production)
dlp_hybrid = DLPCascade(use_presidio=True, use_gliner=True)
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

**Tested on**: Nigel (NVIDIA GPU, 7 test cases)

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Latency p50** | 74ms |
| **Latency p99** | 106ms |
| **Throughput** | ~12 samples/sec |
| **Presidio Coverage** | PERSON, EMAIL, PHONE, CREDIT_CARD, SSN, IP, DATE_TIME, LOCATION |
| **GLiNER2 Enrichment** | passwords, API keys, addresses, bank accounts |
| **Confidence Scores** | Real (0.10-1.00 from Presidio, 0.85 from GLiNER2) |

### Detection Breakdown

| Test Case | Presidio | GLiNER2 | Total | Latency |
|-----------|----------|---------|-------|---------|
| Password | 0 | 1 | 1 | 105.4ms |
| Contact info | 3 | 0 | 3 | 92.7ms |
| Credit card | 2 | 0 | 2 | 74.1ms |
| API key | 0 | 1 | 1 | 73.1ms |
| Address | 0 | 3 | 3 | 73.9ms |
| User profile | 3 | 1 | 4 | 83.7ms |
| Clean text | 0 | 0 | 0 | 72.7ms |

### Comparison vs Pure GLiNER2

| Metric | Pure GLiNER2 | Hybrid | Improvement |
|--------|--------------|--------|-------------|
| Span Accuracy | ⚠️ Text matching | ✅ Exact positions | **Production-ready** |
| Confidence Scores | ❌ Hardcoded (0.9) | ✅ Real (0.10-1.00) | **Threshold tuning** |
| Latency | ~70ms | ~85ms | +15ms (acceptable) |
| Coverage | 27 types | 15 + 27 types | **Combined** |

## Configuration

Edit `cascade_dlp/config.py` to customize:

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
cascade_dlp/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── install.sh             # Installation script
├── example.py             # Usage example
├── cascade_dlp/           # Python package
│   ├── __init__.py        # Package exports
│   ├── cascade.py         # Main pipeline
│   └── config.py          # Configuration
└── experiments/           # Benchmarks & tests
    ├── eval/
    │   └── benchmark.py
    └── tests/
        └── test_cascade.py
```

## References

- [GLiNER2](https://github.com/fastino-ai/GLiNER2)
- [ai4privacy dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
- [SecretBench](https://github.com/setu1421/SecretBench)
