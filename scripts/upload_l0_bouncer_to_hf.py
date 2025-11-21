#!/usr/bin/env python3
"""
Upload L0 Bouncer model to HuggingFace Hub

This uploads the DeBERTa-v3-xsmall based safety classifier
that serves as the first tier in the cascade system.
"""
from huggingface_hub import HfApi, create_repo
import os
import shutil

# Configuration
MODEL_PATH = os.path.expanduser("~/wizard101/experiments/cascade/models/l0_bouncer_12k")
REPO_NAME = "deberta-v3-xsmall-l0-bouncer"
REPO_ID = f"vincentoh/{REPO_NAME}"

# Model card content
MODEL_CARD = """---
license: mit
base_model: microsoft/deberta-v3-xsmall
tags:
  - safety
  - content-moderation
  - text-classification
  - deberta
  - guardreasoner
datasets:
  - GuardReasoner
language:
  - en
metrics:
  - f1
  - recall
  - precision
  - accuracy
library_name: transformers
pipeline_tag: text-classification
---

# L0 Bouncer - DeBERTa Safety Classifier

A fast, lightweight safety classifier based on DeBERTa-v3-xsmall (22M parameters) that serves as the first tier (L0) in a multi-tier safety cascade system.

## Model Description

The L0 Bouncer is designed for **high-throughput, low-latency safety screening** of text inputs. It provides binary classification (safe vs. harmful) with a focus on maximizing recall to catch potentially harmful content.

### Key Features
- **Ultra-fast inference**: ~5.7ms per input
- **High recall**: 99% (catches nearly all harmful content)
- **Lightweight**: Only 22M parameters
- **Production-ready**: Designed for real-time content moderation

## Performance Metrics

| Metric | Value |
|--------|-------|
| **F1 Score** | 93.0% |
| **Recall** | 99.0% |
| **Precision** | 87.6% |
| **Accuracy** | 92.5% |
| **Mean Latency** | 5.74ms |
| **P99 Latency** | 5.86ms |

## Training Data

Trained on 12,000 balanced samples from the GuardReasoner dataset, which contains diverse examples of safe and harmful content with reasoning annotations.

### Training Details
- **Base Model**: microsoft/deberta-v3-xsmall
- **Learning Rate**: 2e-5
- **Batch Size**: 32 (effective, with gradient accumulation)
- **Epochs**: 3
- **Max Sequence Length**: 256 tokens
- **Class Weighting**: 1.5x weight on harmful class for higher recall

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "vincentoh/deberta-v3-xsmall-l0-bouncer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Classify text
text = "What is the capital of France?"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)

# Labels: 0 = safe, 1 = harmful
safe_prob = probs[0][0].item()
harmful_prob = probs[0][1].item()

label = "safe" if safe_prob > harmful_prob else "harmful"
confidence = max(safe_prob, harmful_prob)

print(f"Label: {label}, Confidence: {confidence:.2%}")
```

## Cascade Architecture

This model is designed to work as the first tier (L0) in a multi-tier safety cascade:

```
Input → L0 Bouncer (6ms) → 70% pass through
            ↓ 30% escalate
        L1 Analyst (50ms) → Deeper reasoning
            ↓
        L2 Gauntlet (200ms) → Expert ensemble
            ↓
        L3 Judge (async) → Final review
```

### Design Philosophy
- **Safety-first**: Prioritizes catching harmful content (high recall) over avoiding false positives
- **Efficient routing**: 70% of safe traffic passes at L0, saving compute
- **Graceful escalation**: Uncertain cases are escalated to more capable models

## Intended Use

### Primary Use Cases
- Content moderation pipelines
- Safety screening for LLM inputs/outputs
- First-pass filtering in multi-stage systems
- Real-time safety classification

### Limitations
- Binary classification only (safe/harmful)
- Optimized for English text
- May require calibration for specific domains
- Should be used with escalation to more capable models for uncertain cases

## Citation

If you use this model, please cite:

```bibtex
@misc{l0-bouncer-2024,
  author = {Vincent Oh},
  title = {L0 Bouncer: A Fast Safety Classifier for Content Moderation},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/vincentoh/deberta-v3-xsmall-l0-bouncer}
}
```

## License

MIT License - Free for commercial and non-commercial use.

## Contact

For questions or issues, please open an issue on the model repository.
"""

def main():
    print(f"Uploading L0 Bouncer model from: {MODEL_PATH}")
    print(f"Target repository: {REPO_ID}")

    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model path not found: {MODEL_PATH}")
        return

    # Create README.md in the model directory
    readme_path = os.path.join(MODEL_PATH, "README.md")
    print(f"\nCreating model card at: {readme_path}")
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD)
    print("✅ Model card created!")

    # Initialize HF API
    api = HfApi()

    # Create repository (will skip if exists)
    try:
        print(f"\nCreating repository: {REPO_ID}")
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print("✅ Repository created/verified!")
    except Exception as e:
        print(f"Repository creation: {e}")

    # Upload all files from the model directory (excluding checkpoints)
    print("\nUploading model files...")
    try:
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload L0 Bouncer: DeBERTa safety classifier (93% F1, 99% recall, 5.7ms latency)",
            ignore_patterns=[
                "*.pyc",
                "__pycache__",
                ".git",
                "checkpoint-*",  # Exclude training checkpoints
                "trainer_state.json",
                "training_args.bin"
            ]
        )
        print("\n✅ Model uploaded successfully!")
        print(f"\n🔗 View your model at: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise

if __name__ == "__main__":
    main()
