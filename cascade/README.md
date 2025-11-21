# Safety Cascade

**Cascata Fiduciae Fundata** (Cascade of Founded Trust)

A multi-tier content safety classification system that balances speed, accuracy, and resource usage through intelligent routing.

## System Requirements

### Hardware

| Tier | Minimum VRAM | Recommended VRAM | Notes |
|------|--------------|------------------|-------|
| L0 only | 4GB | 8GB | DeBERTa classifier |
| L0 + L1 | **16GB** | 24GB | Adds GuardReasoner 8B |
| Full cascade | **24GB** | 48GB | Includes L2/L3 with Ollama |

### Software

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- [Ollama](https://ollama.ai) (for L2/L3 tiers)

## Quick Start

### Installation

```bash
git clone https://github.com/bigsnarfdude/wizard101.git
cd wizard101/cascade

# Run installation script
chmod +x install.sh
./install.sh
```

### Basic Usage

```python
from cascade import SafetyCascade, CascadeConfig

# Initialize with L0 + L1 (no Ollama required)
config = CascadeConfig(enable_l2=False, enable_l3=False)
cascade = SafetyCascade(config)

# Classify text
result = cascade.classify("What is the capital of France?")
print(f"Label: {result.label}")
print(f"Stopped at: {result.stopped_at}")
print(f"Latency: {result.total_latency_ms:.1f}ms")
```

### Full Cascade (requires Ollama)

```python
from cascade import SafetyCascade

# Full 4-tier cascade
cascade = SafetyCascade()

result = cascade.classify("How do I improve my cooking skills?")
print(f"Label: {result.label}")
print(f"Journey: {' → '.join(l['level'] for l in result.layers)}")
```

## Architecture

```
Input → L0 Bouncer (6ms, 22M params)
           │
           ├─ Confident (70%) → Return result
           │
           └─ Uncertain (30%) ↓
                              │
                        L1 Analyst (100ms, 8B params)
                              │
                              ├─ Confident (75%) → Return result
                              │
                              └─ Uncertain (25%) ↓
                                                 │
                                           L2 Gauntlet (200ms, 6 experts)
                                                 │
                                                 ├─ Consensus (96%) → Return result
                                                 │
                                                 └─ Split (4%) ↓
                                                              │
                                                        L3 Judge (async, 120B)
                                                              │
                                                              └─ Final verdict
```

### Tier Details

| Tier | Model | Size | Latency | Purpose |
|------|-------|------|---------|---------|
| **L0 Bouncer** | DeBERTa-v3-xsmall | 22M | ~6ms | Fast binary filter |
| **L1 Analyst** | GuardReasoner-8B | 8B | ~100ms | Reasoning-based analysis |
| **L2 Gauntlet** | gpt-oss:20b × 6 | 20B | ~200ms | Multi-expert voting |
| **L3 Judge** | gpt-oss:120b | 120B | ~2s | Final authority |

## Performance

### L0 Bouncer (Production Model)

| Metric | Value |
|--------|-------|
| F1 Score | 95.2% |
| Recall | 97% |
| Precision | 93.5% |
| Accuracy | 95.2% |
| Mean Latency | 5.74ms |
| P99 Latency | 5.86ms |

### End-to-End Cascade

| Scenario | Typical Latency | Traffic % |
|----------|-----------------|-----------|
| Clear safe/harmful | 6ms | 70% |
| Needs reasoning | 100-150ms | 25% |
| Expert review | 200-300ms | 4% |
| Final judge | 2-3s | <1% |

## Configuration

```python
from cascade import CascadeConfig

config = CascadeConfig(
    # L0 thresholds
    l0_confidence_threshold=0.7,  # Below this → escalate
    l0_safe_threshold=0.6,        # Safe probability threshold

    # L1 thresholds
    l1_confidence_threshold=0.7,  # Below this → escalate to L2

    # Enable/disable tiers
    enable_l2=True,  # Requires Ollama
    enable_l3=True,  # Requires Ollama with 120B model
)
```

## Model Downloads

Models are available on HuggingFace:

### L0 Bouncer Variants

| Model | Samples | F1 | HuggingFace |
|-------|---------|----|----|
| **l0-bouncer-full** (recommended) | 124K | 95.2% | [vincentoh/deberta-v3-xsmall-l0-bouncer-full](https://huggingface.co/vincentoh/deberta-v3-xsmall-l0-bouncer-full) |
| l0-bouncer-12k | 12K | 93% | [vincentoh/deberta-v3-xsmall-l0-bouncer](https://huggingface.co/vincentoh/deberta-v3-xsmall-l0-bouncer) |
| l0-bouncer-mega | 2.5K | 85.6% | [vincentoh/deberta-v3-xsmall-l0-bouncer-mega](https://huggingface.co/vincentoh/deberta-v3-xsmall-l0-bouncer-mega) |

### L1 Analyst

- Model: [yueliu1999/GuardReasoner-8B](https://huggingface.co/yueliu1999/GuardReasoner-8B)
- Paper: "GuardReasoner: Towards Reasoning-based LLM Safeguards" (arXiv:2501.18492)
- Expected F1: ~84% on safety benchmarks

## Ollama Setup

L2 and L3 require [Ollama](https://ollama.ai) running locally:

```bash

# Start Ollama service
ollama serve

# Pull models
ollama pull gpt-oss:20b   # For L2 Gauntlet (~12GB)
ollama pull gpt-oss:120b  # For L3 Judge (~70GB, optional)
```

## API Reference

### SafetyCascade

```python
class SafetyCascade:
    def __init__(self, config: CascadeConfig = None):
        """Initialize cascade with optional configuration."""

    def classify(self, text: str) -> CascadeResult:
        """
        Classify text through the cascade.

        Returns:
            CascadeResult with label, confidence, and journey details
        """
```

### CascadeResult

```python
@dataclass
class CascadeResult:
    label: str              # "safe" or "harmful"
    stopped_at: str         # "L0", "L1", "L2", or "L3"
    confidence: float       # 0.0 - 1.0
    total_latency_ms: float # End-to-end latency
    layers: list            # Journey through tiers
    reasoning: str          # Explanation (from L1+)
    audit_id: str           # Audit trail ID (from L3)
```

### Individual Tiers

```python
from cascade import L0Bouncer, L1Analyst, L2Gauntlet, L3Judge

# L0: Fast classifier
l0 = L0Bouncer(model_path="./models/l0_bouncer")
result = l0.classify("text")  # Returns dict with label, confidence, probs

# L1: Reasoning analyst
l1 = L1Analyst(model_id="yueliu1999/GuardReasoner-8B")
result = l1.analyze("text")  # Returns dict with label, reasoning, confidence

# L2: Expert voting
l2 = L2Gauntlet(ollama_url="http://localhost:11434/api/generate")
result = l2.analyze("text")  # Returns dict with label, votes, consensus

# L3: Final judge
l3 = L3Judge()
result = l3.judge("text", context={})  # Returns dict with label, reasoning, audit_id
```

## Training

### L0 Bouncer Training

```bash
# Download GuardReasoner dataset and train
python train_l0_full.py
```

Training config:
- Base model: `microsoft/deberta-v3-xsmall`
- Learning rate: 2e-5
- Batch size: 32 (with gradient accumulation)
- Epochs: 3
- Dataset: GuardReasoner (124K samples)

## Design Philosophy

1. **Safety-first**: High recall (catch harmful content) over precision (avoid false positives)
2. **Efficient routing**: 70% of traffic handled at L0 in <10ms
3. **Graceful escalation**: Uncertain cases get more compute, not less accuracy
4. **Audit trail**: L3 creates training signal for continuous improvement

## Directory Structure

```
cascade/
├── __init__.py           # Package exports
├── cascade.py            # Main SafetyCascade class
├── l0_bouncer.py         # L0: DeBERTa classifier
├── l1_analyst.py         # L1: GuardReasoner-8B
├── l2_gauntlet.py        # L2: Expert voting
├── l3_judge.py           # L3: Final authority
├── requirements.txt      # Python dependencies
├── install.sh            # Installation script
├── example.py            # Usage examples
├── README.md             # This file
└── models/               # Downloaded models
    └── l0_bouncer/
```

## Citation

```bibtex
@misc{safety-cascade-2024,
  author = {Vincent Oh},
  title = {Safety Cascade: Multi-tier Content Safety Classification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/bigsnarfdude/wizard101}
}
```

## License

MIT License
