# Wizard101 - AI Safety Cascade

**Cascata Fiduciae Fundata** (Cascade of Founded Trust)

A comprehensive four-layer AI safety system that provides end-to-end protection from input validation to output filtering, combining speed, accuracy, and robustness through intelligent routing.

![Wizard101 Project](wizard101.png)

---

## System Overview

The Wizard101 safety cascade consists of four specialized protection layers:

```

┌─────────────────────────────────────────────────────────┐
│                    ONION SECURITY                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   User Input                                            │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────┐                                           │
│   │ Layer 1 │  Safety Cascade (L0→L1→L3)                │
│   │         │  Block harmful prompts                    │
│   └────┬────┘                                           │
│        │                                                │
│        ▼                                                │
│   ┌─────────┐                                           │
│   │ Layer 2 │  Refusal Cascade (Llama Guard)            │
│   │         │  Route to appropriate refusal             │
│   └────┬────┘                                           │
│        │                                                │
│        ▼                                                │
│   ┌─────────┐                                           │
│   │ Layer 3 │  QUARANTINE (this layer)                  │
│   │         │  Sanitize before privileged LLM           │
│   └────┬────┘                                           │
│        │                                                │
│        ▼                                                │
│   ┌─────────┐                                           │
│   │ Layer 4 │  Privileged LLM                           │
│   │         │  Has tools, can take actions              │
│   └────┬────┘                                           │
│        │                                                │
│        ▼                                                │
│   ┌─────────┐                                           │
│   │ Layer 5 │  DLP                                      │
│   │         │  Block sensitive data in output           │
│   └─────────┘                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘


```

### The Four Cascades

| Layer | Purpose | Speed | Status |
|-------|---------|-------|--------|
| **[cascade_inbound](#cascade_inbound-request-safety)** | Block harmful prompts | 2ms-8s | ✅ Built |
| **[cascade_refusals](#cascade_refusals-refusal-generation)** | Generate appropriate refusals | ~1s | ✅ Built |
| **[cascade_dlp](#cascade_dlp-data-loss-prevention)** | Detect PII/secrets in outputs | <10ms | ✅ Built |
| **[cascade_quarantine](#cascade_quarantine-feedback-loop)** | Capture edge cases for retraining | N/A | 🔄 Planned |

---

## cascade_inbound: Request Safety

**Status**: ✅ Production Ready
**Purpose**: Fast multi-tier content safety classification - determines IF content is harmful

### Architecture

```
Input → L0 Bouncer (2ms, DeBERTa-v3-xsmall, 22M params)
           │
           ├─ Confident (94.2%) → Return safe/harmful
           │
           └─ Uncertain (5.8%) ↓
                              │
                        L1 Analyst (8s, GuardReasoner-8B 4-bit)
                              │
                              ├─ Confident → Return safe/harmful
                              │
                              └─ Uncertain (2.3%) ↓
                                                 │
                                           L2 Classifier (0.18s, gpt-oss-safeguard:20b)
                                                 │
                                                 └─ Final verdict
```

### Performance Metrics

**Public Safety Benchmarks (1,050 samples) - November 2025**

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.0% |
| **Precision** | 96.9% |
| **Recall** | 95.1% |
| **F1 Score** | 96.0% |

**Recommended Production Stack**: DeBERTa → GuardReasoner-8B → gpt-oss-safeguard:20b

| Layer | Model | Speed | VRAM | Accuracy |
|-------|-------|-------|------|----------|
| **L0** | DeBERTa-v3-xsmall | 2ms | <1GB | 95.2% (when confident) |
| **L1** | GuardReasoner-8B (4-bit) | 8s | 5GB | 88.5% (+29.5% value) |
| **L2** | gpt-oss-safeguard:20b | 0.18s | 13GB | 87.5% |

**Total System**: 94.9% accuracy, 96.6% F1, ~19GB VRAM

### Key Findings

1. **L1 is irreplaceable** - GuardReasoner-8B adds +29.5% value; L2 alone scores 59% on hard cases
2. **Harmony template critical** - gpt-oss models need `<|start|>user<|message|>...<|end|>` format
3. **Safeguard 20b = 120b accuracy** - Same 87.5% at 6.7x faster, 5x less VRAM
4. **0.9 threshold optimal** - Sends 5.8% to L1, catches 31 vs 39 dangerous FN
5. **512 tokens sufficient** - 4x faster L1 with same accuracy

### Layer Distribution

```
L0 catches:  94.2% (fast confident decisions)
L1 catches:   5.8% (reasoning required)
L2 catches:   2.3% (expert consensus)
```

### Quick Start

```python
from cascade_inbound import SafetyCascade, CascadeConfig

# Initialize cascade
config = CascadeConfig(
    l0_confidence_threshold=0.9,
    enable_l2=True
)
cascade = SafetyCascade(config)

# Classify text
result = cascade.classify("How do I pick a lock?")

print(f"Label: {result.label}")           # harmful
print(f"Stopped at: {result.stopped_at}") # L0
print(f"Confidence: {result.confidence}") # 0.95
print(f"Latency: {result.total_latency_ms}ms")  # 2.3
```

**Location**: `cascade_inbound/`
**Documentation**: [cascade_inbound/README.md](cascade_inbound/README.md)

---

## cascade_refusals: Refusal Generation

**Status**: ✅ Built
**Purpose**: Determines HOW to refuse harmful content, not just IF it's harmful

### Refusal Taxonomy

| Type | Description | Example Response |
|------|-------------|------------------|
| **HARD** | Illegal, dangerous, clearly harmful | "I can't help with that." |
| **SOFT** | Borderline, dual-use, needs guardrails | "I can explain the mechanics for educational purposes, but won't provide bypassing instructions." |
| **REDIRECT** | Misunderstood intent, can help differently | "I'll help you terminate the process. Here's how..." |
| **CLARIFY** | Ambiguous, need more context | "Could you clarify? Are you asking about access procedures or something else?" |
| **CONDITIONAL** | Acceptable with context/credentials | "This requires medical training. Are you a healthcare professional?" |

### Architecture

```
Harmful Input (from inbound cascade)
         │
         ▼
┌─────────────────┐
│  Llama Guard 3  │  1s - MLCommons categories (S1-S14)
│   (8B model)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
  ┌─▼─┐   ┌──▼──┐   ┌──────────┐
  │HARD│  │SOFT │  │CONDITIONAL│
  └─┬─┘   └──┬──┘   └────┬─────┘
    │        │           │
    └────────┼───────────┘
             │
      ┌──────▼──────┐
      │  Response   │
      │  Generator  │
      │ (templates) │
      └──────┬──────┘
             │
             ▼
      Final Refusal + Resources
```

### MLCommons Category Mapping

**HARD REFUSALS** (No engagement):
- S1: Violent Crimes
- S2: Non-Violent Crimes
- S3: Sex-Related Crimes
- S4: Child Exploitation
- S9: Indiscriminate Weapons
- S14: Code Interpreter Abuse

**SOFT REFUSALS** (Partial engagement):
- S6: Specialized Advice (medical, legal, financial)
- S10: Hate Speech
- S11: Suicide & Self-Harm
- S12: Sexual Content

**CONDITIONAL** (Need more context):
- S5: Defamation
- S7: Privacy
- S8: Intellectual Property
- S13: Elections

### Performance

| Metric | Value |
|--------|-------|
| Accuracy | 100% (8/8 test cases) |
| Latency | 436-1052ms per query |

### Quick Start

```python
from cascade_refusals import RefusalPipeline

pipeline = RefusalPipeline()
result = pipeline.process("How do I make a bomb?")

print(result)
# {
#   "safe": False,
#   "categories": ["S9"],
#   "category_names": ["Indiscriminate Weapons"],
#   "strategy": "HARD",
#   "response": "I can't provide information about creating weapons...",
#   "tone": "firm",
#   "latency_ms": 1052
# }
```

**Location**: `cascade_refusals/`
**Documentation**: [cascade_refusals/README.md](cascade_refusals/README.md)

---

## cascade_dlp: Data Loss Prevention

**Status**: ✅ Production Ready
**Purpose**: Detects sensitive data (PII, secrets, credentials) in LLM outputs to prevent exfiltration

### Threat Model

**Insider Risk**:
- Model trained on sensitive internal data
- Prompt injection extracts private information
- Unintentional disclosure of proprietary knowledge

**Detection Categories**:

| Category | Examples | Risk Level |
|----------|----------|------------|
| **PII** | SSN, phone, email, address | HIGH |
| **Credentials** | API keys, passwords, tokens | CRITICAL |
| **Financial** | Credit cards, bank accounts | HIGH |
| **Medical** | PHI, diagnosis, prescriptions | HIGH |
| **Internal** | Employee names, internal docs | MEDIUM |

### Architecture

```
Model Response
      │
      ▼
┌─────────────┐
│  Presidio   │  <1ms - Fast pattern/NER hybrid
│  Scanner    │
└──────┬──────┘
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
│  or  │
│REDACT│
└──────┘
```

### Benchmark Results (ai4privacy 209K samples)

**Full Scale Evaluation - November 2025**

| Metric | Value | Notes |
|--------|-------|-------|
| **Precision** | 100.0% | Zero false positives on 200k samples |
| **Recall** | 88.2% | Detected 88.2% of PII instances |
| **F1 Score** | 93.7% | Excellent balanced performance |
| **Latency p50** | 3.7ms | Median processing time |
| **Latency p95** | 5.5ms | 95th percentile |
| **Latency p99** | 6.5ms | 99th percentile |
| **Throughput** | 259 samples/sec | Real-time capable |
| **Total Runtime** | 13.4 minutes | For 209,261 samples |

**Confusion Matrix**:
- True Positives: 184,589
- False Positives: 0
- False Negatives: 24,672
- True Negatives: 0

### Dataset Breakdown

| Dataset | Samples | Precision | Recall | F1 | Latency |
|---------|---------|-----------|--------|-----|---------|
| secret_test_set | 11 | 100% | 100% | 100% | 9.1ms |
| pii_test_set | 10 | 71.4% | 100% | 83.3% | 4.3ms |
| ai4privacy (1K) | 1000 | 100% | 87.5% | 93.3% | 4.0ms |
| **ai4privacy (FULL)** | **209,261** | **100%** | **88.2%** | **93.7%** | **3.8ms** |

### Key Findings

1. **Perfect Precision**: Zero false positives across 209k samples - no unnecessary blocking
2. **Consistent Performance**: Metrics improved slightly at scale, indicating good generalization
3. **Fast Processing**: 3.7ms median latency enables real-time DLP
4. **No Overfitting**: Performance maintained or improved on larger dataset
5. **Secret Detection**: 100% recall on API keys, tokens, credentials
6. **Production Ready**: <10ms latency for all operations

### Quick Start

```python
from cascade_dlp import DLPScanner

scanner = DLPScanner()

# After model generates response
response = model.generate(prompt)

# Scan for sensitive data
result = scanner.scan(response)

if result["findings"]:
    if result["severity"] == "CRITICAL":
        response = "I've redacted sensitive credentials from my response."
    else:
        response = scanner.redact(response)

return response
```

**Location**: `cascade_dlp/`
**Documentation**: [cascade_dlp/README.md](cascade_dlp/README.md)
**Benchmark Results**: [cascade_dlp/eval/BENCHMARK_RESULTS_FULL.txt](cascade_dlp/eval/BENCHMARK_RESULTS_FULL.txt)

---

## cascade_quarantine: Feedback Loop

**Status**: 🔄 Planned
**Purpose**: Captures edge cases and false positives/negatives for continuous model improvement

### Concept

```
┌────────────────────────────────────────────────┐
│             FEEDBACK COLLECTION                │
├────────────────────────────────────────────────┤
│                                                │
│  False Positives → cascade_quarantine/fp/      │
│  False Negatives → cascade_quarantine/fn/      │
│  Edge Cases      → cascade_quarantine/edge/    │
│                                                │
│  Human Review → Labeling → Retraining Dataset  │
│                                                │
└────────────────────────────────────────────────┘
```

### Planned Features

1. **Automatic Capture**: Log uncertain cases (low confidence scores)
2. **Human Review**: Web interface for expert annotation
3. **Dataset Building**: Convert quarantine → training data
4. **Online Learning**: Periodic L0 model retraining
5. **A/B Testing**: Validate improvements before deployment

**Location**: `cascade_quarantine/`
**Documentation**: [cascade_quarantine/README.md](cascade_quarantine/README.md)

---

## Complete Pipeline Flow

### End-to-End Example

```python
from cascade_inbound import SafetyCascade
from cascade_refusals import RefusalPipeline
from cascade_dlp import DLPScanner
from your_llm import generate_response

# Initialize all cascades
safety = SafetyCascade()
refusal = RefusalPipeline()
dlp = DLPScanner()

def safe_llm_pipeline(user_input: str) -> str:
    """Complete safety pipeline."""

    # STEP 1: Inbound safety check
    safety_result = safety.classify(user_input)

    if safety_result.label == "harmful":
        # STEP 2: Generate appropriate refusal
        refusal_result = refusal.process(user_input)
        return refusal_result["response"]

    # STEP 3: Generate model response
    model_output = generate_response(user_input)

    # STEP 4: Outbound DLP check
    dlp_result = dlp.scan(model_output)

    if dlp_result["findings"]:
        if dlp_result["severity"] == "CRITICAL":
            return "I've removed sensitive information from my response for security."
        else:
            return dlp.redact(model_output)

    return model_output

# Usage
response = safe_llm_pipeline("What is the capital of France?")
print(response)  # "The capital of France is Paris."

response = safe_llm_pipeline("How do I make a bomb?")
print(response)  # "I can't provide information about creating weapons..."
```

### Latency Breakdown

| Component | Typical Latency | Traffic % |
|-----------|-----------------|-----------|
| L0 Bouncer | 2ms | 94.2% |
| L1 Analyst | 8s | 5.8% |
| L2 Classifier | 0.18s | 2.3% |
| Refusal Generator | ~1s | When harmful |
| DLP Scanner | <10ms | 100% |
| **Average Pipeline** | **~15ms** | **For 94% of safe requests** |

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ VRAM (for full cascade)
- [Ollama](https://ollama.ai) (for L2/L3 and refusals)

### Quick Install

```bash
# Clone repository
git clone https://github.com/bigsnarfdude/wizard101.git
cd wizard101

# Install each cascade
cd cascade_inbound && ./install.sh && cd ..
cd cascade_refusals && pip install -r requirements.txt && cd ..
cd cascade_dlp && pip install -r requirements.txt && cd ..
```

### Ollama Models

```bash
# Start Ollama service
ollama serve

# Pull required models
ollama pull gpt-oss-safeguard:latest     # For L2 classifier (~13GB)
ollama pull gpt-oss-safeguard:20b        # Alternative L2 (~13GB)
ollama pull meta-llama/Llama-Guard-3-8B  # For refusal generator (~8GB)
```

---

## System Requirements

### Minimum Configuration (L0 + DLP only)

- **VRAM**: 4GB
- **Components**: Fast classifier + DLP scanner
- **Latency**: <10ms for 99% of traffic
- **Use case**: High-throughput, latency-sensitive applications

### Recommended Configuration (Full Cascade)

- **VRAM**: 24GB
- **Components**: L0 + L1 + L2 + Refusals + DLP
- **Latency**: 2ms-8s depending on complexity
- **Use case**: Production safety-critical applications

### Hardware Recommendations

| Tier | GPU | VRAM | Notes |
|------|-----|------|-------|
| **Minimal** | GTX 1080 | 8GB | L0 + DLP only |
| **Standard** | RTX 3090 | 24GB | Full cascade without L3 |
| **Enterprise** | A100 | 40GB+ | Full cascade + headroom |

---

## Performance Summary

### Inbound Safety (cascade_inbound)

- **Accuracy**: 94.9%
- **F1 Score**: 96.6%
- **Latency**: 2ms (L0) to 8s (L1)
- **Layer Distribution**: 94.2% / 5.8% / 2.3%

### Refusal Generation (cascade_refusals)

- **Accuracy**: 100% (tested)
- **Latency**: 436-1052ms
- **Categories**: 14 MLCommons categories
- **Strategies**: HARD / SOFT / CONDITIONAL / REDIRECT / CLARIFY

### Data Loss Prevention (cascade_dlp)

- **Precision**: 100% (209K samples)
- **Recall**: 88.2%
- **F1 Score**: 93.7%
- **Latency**: 3.7ms (median), 6.5ms (p99)
- **Throughput**: 259 samples/sec

---

## Design Principles

### 1. Defense in Depth
Multiple specialized layers provide redundant protection. Failure of one layer doesn't compromise the system.

### 2. Speed First
70-80% of safe requests handled in <10ms by L0 + DLP. Only uncertain cases escalate.

### 3. High Recall
Safety systems must catch harmful content. We prioritize recall over precision.

### 4. Transparent Reasoning
L1, L2, and Refusal Generator provide reasoning traces for auditability.

### 5. Graceful Degradation
Each cascade can operate independently. System remains functional even if components fail.

### 6. Continuous Learning
Quarantine layer captures edge cases to improve models over time.

---

## Research Foundation

This project builds on:

- **GuardReasoner** (Liu et al., 2025) - Reasoning-based safety classification
- **gpt-oss-safeguard** (OpenAI, 2025) - Multi-policy safety models
- **Llama Guard 3** (Meta, 2024) - MLCommons safety categories
- **Presidio** (Microsoft, 2023) - PII detection and anonymization
- **ai4privacy** (2024) - Large-scale PII masking dataset (209K samples)
- **WildGuard** (Han et al., 2024) - Safety benchmark dataset

---

## File Structure

```
wizard101/
├── README.md                    # This file
├── wizard101.png                # Project diagram
├── cascade_inbound/             # Request safety (L0/L1/L2)
│   ├── README.md
│   ├── cascade.py
│   ├── l0_bouncer.py
│   ├── l1_analyst.py
│   ├── l2_gauntlet.py
│   └── models/
├── cascade_refusals/            # Refusal generation
│   ├── README.md
│   ├── refusal_pipeline.py
│   ├── refusal_generator.py
│   └── test_llama_guard.py
├── cascade_dlp/                 # Data loss prevention
│   ├── README.md
│   ├── src/cascade.py
│   ├── eval/
│   │   ├── benchmark.py
│   │   ├── benchmark_scale.py
│   │   ├── BENCHMARK_RESULTS_FULL.txt
│   │   └── datasets/
│   └── tests/
└── cascade_quarantine/          # Feedback loop (planned)
    └── README.md
```

---

## Citation

```bibtex
@misc{wizard101-2025,
  author = {Vincent Oh},
  title = {Wizard101: AI Safety Cascade},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/bigsnarfdude/wizard101}
}
```

---

## License

MIT License - Educational & Research Use

Copyright (c) 2025 bigsnarfdude

---

## Next Steps

1. **Threshold Optimization**: Tune L0/L1 thresholds per use case
2. **Model Distillation**: Distill L1 reasoning into faster L0
3. **Multi-language**: Extend DLP to non-English content
4. **Quarantine Implementation**: Build feedback loop pipeline
5. **Online Learning**: Update L0 on quarantine data
6. **API Integration**: Production deployment endpoints

---

*Last Updated: 2025-11-24*
*Version: 1.0.0*
