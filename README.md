# Wizard101 - Safety Cascade

**Cascata Fiduciae Fundata** (Cascade of Founded Trust)

A multi-tier AI safety guardrail system that combines speed, accuracy, and robustness through intelligent routing.

![Wizard101 Project](wizard101.png)

---

## Recommended Production Stack

> **TL;DR: DeBERTa → GuardReasoner-8B → gpt-oss-safeguard:20b with Harmony template**

### Results:     94.9% 

### Accuracy:     96.6% F1



| Layer | Model | Speed | VRAM | Accuracy |
|-------|-------|-------|------|----------|
| **L0** | DeBERTa-v3-xsmall | 2ms | <1GB | 95.2% (when confident) |
| **L1** | GuardReasoner-8B (4-bit) | 8s | 5GB | 88.5% (+29.5% value) |
| **L2** | gpt-oss-safeguard:20b | 0.18s | 13GB | 87.5% |

### Key Findings

1. **L1 is irreplaceable** - GuardReasoner-8B adds +29.5% value; L2 alone scores 59% on hard cases
2. **Harmony template critical** - gpt-oss models need `<|start|>user<|message|>...<|end|>` format
3. **Safeguard 20b = 120b accuracy** - Same 87.5% at 6.7x faster, 5x less VRAM
4. **0.9 threshold optimal** - Sends 5.8% to L1, catches 31 vs 39 dangerous FN
5. **512 tokens sufficient** - 4x faster L1 with same accuracy

> **Why L1 matters**: Tested skipping L1 → L2 scored 59% (same as L0). Hard cases need reasoning, not classification. No shortcuts.

### Quick Deploy

```python
# L2 with Harmony template
prompt = f'''<|start|>user<|message|>Classify this text as either "harmful" or "safe". Respond with only one word.

Text: "{text}"<|end|>
<|start|>assistant<|channel|>final<|message|>'''

response = ollama.generate(model="gpt-oss-safeguard:latest", prompt=prompt)
```

### Resource Comparison

| Config | Accuracy | L2 Speed | Total VRAM |
|--------|----------|----------|------------|
| **Recommended** (safeguard 20b) | 94.9% | 0.18s | ~19GB |
| Heavy (120b) | 94.9% | 1.2s | ~71GB |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SAFETY CASCADE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input  ──►  L0 Bouncer  ──►  L1 Analyst  ──►  L2 Classifier│
│               (2ms)           (8s)             (0.18s)       │
│               DeBERTa         GuardReasoner    gpt-oss-      │
│               124K trained    8B 4-bit         safeguard     │
│                                                              │
│   Routing: confidence < 0.9 escalates, disagreement → L2    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layer Details

| Layer | Model | Speed | Purpose | Catches |
|-------|-------|-------|---------|---------|
| **L0 Bouncer** | DeBERTa-v3-xsmall | 2ms | Fast filter | 94.2% |
| **L1 Analyst** | GuardReasoner-8B (4-bit) | 8s | Reasoning | 5.8% |
| **L2 Classifier** | gpt-oss-safeguard:20b | 0.18s | Tiebreaker | 2.3% |

---

## Benchmark Results

### Public Safety Benchmarks (1,050 samples) - November 2025

**HarmBench + XSTest + SimpleSafetyTests**

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.0% |
| **Precision** | 96.9% |
| **Recall** | 95.1% |
| **F1 Score** | 96.0% |

| Benchmark | Samples | Accuracy | F1 |
|-----------|---------|----------|-----|
| HarmBench | 500 | 99.6% | 99.8% |
| SimpleSafetyTests | 100 | 96.0% | 98.0% |
| XSTest (over-refusal) | 450 | 87.3% | 85.4% |

**Key Findings**:
- Official GuardReasoner prompt format critical for L1 performance (+29.7% accuracy improvement)
- Speed optimization: Reducing max_new_tokens from 2048→512 improved L1 inference 4x faster while maintaining accuracy

### GuardReasoner Test Set (1,000 samples)

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.2% |
| **Precision** | 97.4% |
| **Recall** | 96.6% |
| **F1 Score** | 97.0% |
| **False Negatives** | 16 |

### Heretic Adversarial Jailbreaks (1,000 samples)

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.3% |
| **Precision** | 93.4% |
| **Recall** | 99.6% |
| **F1 Score** | 96.4% |
| **False Negatives** | 2 |

**Key Finding**: 99.6% recall on adversarial jailbreaks - catches 498/500 harmful prompts.

### Layer Distribution (Heretic 1k)

```
L0 catches:  98.6% (confident decisions)
L1 catches:   1.1% (reasoning required)
L2 catches:   0.3% (expert consensus)
```

**Note**: L0 trained on 124K samples catches nearly all adversarial prompts.

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA GPU with 8GB+ VRAM (for L1)
- Ollama with `gpt-oss:20b` model (for L2)

### Installation

```bash
# Clone repository
git clone https://github.com/bigsnarfdude/wizard101.git
cd wizard101/experiments/cascade

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets scikit-learn peft bitsandbytes

# Download L0 model (or train your own)
# Model: DeBERTa-v3-xsmall trained on 124K samples
```

### Running the Cascade

```python
from cascade import SafetyCascade, CascadeConfig

# Initialize cascade
config = CascadeConfig(
    l0_confidence_threshold=0.7,
    l1_confidence_threshold=0.7,
    enable_l2=True,
    enable_l3=False,
)
cascade = SafetyCascade(config)

# Classify text
result = cascade.classify("How do I pick a lock?")

print(f"Label: {result.label}")           # harmful
print(f"Stopped at: {result.stopped_at}") # L0
print(f"Confidence: {result.confidence}") # 0.95
print(f"Latency: {result.total_latency_ms}ms")  # 2.3
```

### Batch Evaluation (Memory Efficient)

```bash
# Evaluate on 1000 samples with batch mode
python evaluate_cascade_batch.py --samples 1000

# Evaluate on Heretic adversarial dataset
python evaluate_heretic.py --batch
```

---

## Training L0 Bouncer

The L0 model is trained on the GuardReasoner dataset (124K samples).

```bash
# Full training (124K samples, ~35 minutes on A100)
python train_l0_full.py

# Results:
# - Accuracy: 95.2%
# - Precision: 96.5%
# - Recall: 95.0%
# - F1: 95.8%
```

**Training Data**: `../guardreasoner/guardreasoner_data/all_combined.json`

---

## File Structure

```
experiments/cascade/
├── cascade.py              # Main cascade orchestrator
├── l0_bouncer.py           # L0: Fast DeBERTa classifier
├── l1_analyst.py           # L1: Llama 3.2 3B with LoRA
├── l2_gauntlet.py          # L2: 6-expert voting panel
├── l3_judge.py             # L3: Final authority (Claude/GPT-4)
├── train_l0_full.py        # Train L0 on 124K samples
├── evaluate_cascade.py     # Standard evaluation
├── evaluate_cascade_batch.py  # Memory-efficient batch evaluation
├── evaluate_heretic.py     # Adversarial jailbreak evaluation
└── README.md               # This file
```

---

## Configuration

### Cascade Thresholds

```python
@dataclass
class CascadeConfig:
    # L0: Stop if confidence >= threshold
    l0_confidence_threshold: float = 0.7

    # L1: Stop if confidence >= threshold
    l1_confidence_threshold: float = 0.7

    # Enable/disable layers
    enable_l2: bool = True
    enable_l3: bool = True
```

**Tuning Tips**:
- Higher thresholds = more escalation = better accuracy, slower
- Lower thresholds = less escalation = faster, may miss edge cases
- For safety-critical: use 0.8+ thresholds
- For speed-critical: use 0.6 thresholds

---

## Layer Implementation Details

### L0 Bouncer (`l0_bouncer.py`)

**Architecture**: DeBERTa-v3-xsmall (22M params)
**Training**: 124K samples, 3 epochs, batch size 32
**Speed**: ~2ms per classification

```python
from l0_bouncer import L0Bouncer

l0 = L0Bouncer()
result = l0.classify("What is the capital of France?")
# {'label': 'safe', 'confidence': 0.98, 'safe_prob': 0.98}
```

### L1 Analyst (`l1_analyst.py`)

**Architecture**: GuardReasoner-8B (4-bit quantized with bitsandbytes)
**Training**: Official GuardReasoner prompt format
**Speed**: ~8s per analysis (with 512 max tokens)

```python
from l1_analyst import L1Analyst

l1 = L1Analyst()
result = l1.analyze("How to pick a lock")
# {'label': 'harmful', 'confidence': 0.9, 'reasoning': '...step-by-step...'}
```

### L2 Classifier (`l2_gauntlet.py`)

**Architecture**: gpt-oss:120b direct classification (no CoT)
**Method**: Direct classification without chain-of-thought
**Speed**: ~500ms per analysis

```python
from l2_gauntlet import L2Gauntlet

l2 = L2Gauntlet()
result = l2.analyze("edge case prompt")
# {'label': 'harmful', 'confidence': 0.95}
```

**Note**: 120b with direct prompting outperforms 6-expert voting (86% vs 71% on edge cases). CoT hurts large models.

---

## Evaluation Scripts

### Standard Evaluation

```bash
# Quick 100-sample test
python evaluate_cascade.py --samples 100

# Full 1000-sample evaluation with L2
python evaluate_cascade.py --samples 1000 --l2
```

### Batch Mode (Recommended)

Memory-efficient: loads one model at a time.

```bash
# 1000 samples with default thresholds
python evaluate_cascade_batch.py --samples 1000

# Custom thresholds
python evaluate_cascade_batch.py --samples 1000 --l0-threshold 0.8 --l1-threshold 0.8
```

### Heretic Adversarial Dataset

Tests robustness against jailbreak attempts.

```bash
# Batch mode (memory efficient)
python evaluate_heretic.py --batch

# Standard mode
python evaluate_heretic.py
```

---

## Results Output

All evaluation scripts save detailed results to JSON:

```json
{
  "summary": {
    "accuracy": 97.2,
    "precision": 97.4,
    "recall": 96.6,
    "f1": 97.0
  },
  "confusion_matrix": {
    "tp": 473, "tn": 499, "fp": 12, "fn": 16
  },
  "layer_distribution": {
    "L0": 752, "L1": 198, "L2": 50
  }
}
```

---

## Design Principles

### 1. Speed First
Most requests (70-80%) are handled by L0 in <5ms. Only uncertain cases escalate.

### 2. High Recall
Safety systems must catch harmful content. We prioritize recall over precision.

### 3. Transparent Reasoning
L1 and L2 provide reasoning traces for auditability.

### 4. Graceful Degradation
Each layer can operate independently. If L2 is unavailable, L1 makes final decision.

---

## Comparison to Alternatives

| System | F1 | Speed | Notes |
|--------|-----|-------|-------|
| **Safety Cascade** | 97.0% | 2-200ms | Multi-tier, high recall |
| GuardReasoner-8B | 98% | 40s | Single model, slow |
| WildGuard | 87.6% | 200ms | Good but lower accuracy |
| GPT-4 Moderation | ~90% | 500ms | API cost, latency |

---

## Research Foundation

This project builds on:

- **GuardReasoner** (Liu et al., 2025) - Reasoning-based safety classification
- **gpt-oss-safeguard** (OpenAI, 2025) - Multi-policy safety models
- **WildGuard** (Han et al., 2024) - Safety benchmark dataset

---

## Next Steps

1. **Threshold Optimization**: Tune L0/L1 thresholds per use case
2. **Model Distillation**: Distill L1 reasoning into faster L0
3. **Multi-language**: Extend to non-English content
4. **Online Learning**: Update L0 on false negatives

---

## License

MIT License - Educational & Research Use

Copyright (c) 2025 bigsnarfdude

---

