#!/usr/bin/env python3
"""
Upload all L0 Bouncer model variants to HuggingFace Hub
"""
from huggingface_hub import HfApi, create_repo
import os

# Model configurations
MODELS = [
    {
        "path": "l0_bouncer_full",
        "repo_name": "deberta-v3-xsmall-l0-bouncer-full",
        "description": "Full GuardReasoner dataset (124K samples), 10K+ training steps",
        "metrics": {
            "f1": "95.2%",
            "recall": "97%",
            "precision": "93.5%",
            "accuracy": "95.2%",
            "samples": "124,000",
            "steps": "10,719"
        }
    },
    {
        "path": "l0_bouncer_mega",
        "repo_name": "deberta-v3-xsmall-l0-bouncer-mega",
        "description": "Mega dataset iteration (2.5K samples)",
        "metrics": {
            "f1": "85.6%",
            "recall": "91%",
            "precision": "81%",
            "accuracy": "87.8%",
            "samples": "2,500",
            "steps": "~750"
        }
    }
]

def create_model_card(model_config):
    """Generate model card for each variant"""
    m = model_config["metrics"]
    return f"""---
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

# L0 Bouncer ({model_config['path']}) - DeBERTa Safety Classifier

A fast, lightweight safety classifier based on DeBERTa-v3-xsmall (22M parameters) that serves as the first tier (L0) in a multi-tier safety cascade system.

**Variant**: {model_config['description']}

## Performance Metrics

| Metric | Value |
|--------|-------|
| **F1 Score** | {m['f1']} |
| **Recall** | {m['recall']} |
| **Precision** | {m['precision']} |
| **Accuracy** | {m['accuracy']} |
| **Training Samples** | {m['samples']} |
| **Training Steps** | {m['steps']} |
| **Mean Latency** | ~5.7ms |

## Model Description

The L0 Bouncer is designed for **high-throughput, low-latency safety screening** of text inputs. It provides binary classification (safe vs. harmful) with a focus on maximizing recall to catch potentially harmful content.

### Key Features
- **Ultra-fast inference**: ~5.7ms per input
- **Lightweight**: Only 22M parameters
- **Production-ready**: Designed for real-time content moderation

## Training Data

Trained on the GuardReasoner dataset, which contains diverse examples of safe and harmful content with reasoning annotations.

### Training Details
- **Base Model**: microsoft/deberta-v3-xsmall
- **Learning Rate**: 2e-5
- **Batch Size**: 32 (effective, with gradient accumulation)
- **Max Sequence Length**: 256 tokens
- **Class Weighting**: Higher weight on harmful class for better recall

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "vincentoh/{model_config['repo_name']}"
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

print(f"Label: {{label}}, Confidence: {{confidence:.2%}}")
```

## Model Variants

| Variant | Samples | F1 | Recall | Best For |
|---------|---------|----|----|----------|
| [l0-bouncer-12k](https://huggingface.co/vincentoh/deberta-v3-xsmall-l0-bouncer) | 12K | 93% | 99% | Balanced performance |
| [l0-bouncer-full](https://huggingface.co/vincentoh/deberta-v3-xsmall-l0-bouncer-full) | 124K | 95.2% | 97% | Maximum accuracy |
| [l0-bouncer-mega](https://huggingface.co/vincentoh/deberta-v3-xsmall-l0-bouncer-mega) | 2.5K | 85.6% | 91% | Lightweight/iterative |

## Cascade Architecture

This model is designed to work as the first tier (L0) in a multi-tier safety cascade:

```
Input → L0 Bouncer (6ms) → 70% pass through
            ↓ 30% escalate
        L1 Analyst (50ms) → Deeper reasoning
            ↓
        L2 Gauntlet (200ms) → Expert ensemble
```

## License

MIT License - Free for commercial and non-commercial use.

## Citation

```bibtex
@misc{{l0-bouncer-2024,
  author = {{Vincent Oh}},
  title = {{L0 Bouncer: A Fast Safety Classifier for Content Moderation}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/vincentoh/{model_config['repo_name']}}}
}}
```
"""

def main():
    base_path = os.path.expanduser("~/wizard101/experiments/cascade/models")
    api = HfApi()

    for model_config in MODELS:
        model_path = os.path.join(base_path, model_config["path"])
        repo_id = f"vincentoh/{model_config['repo_name']}"

        print(f"\n{'='*60}")
        print(f"Uploading: {model_config['path']}")
        print(f"Repository: {repo_id}")
        print(f"{'='*60}")

        # Check if model path exists
        if not os.path.exists(model_path):
            print(f"❌ Model path not found: {model_path}")
            continue

        # Create README.md
        readme_path = os.path.join(model_path, "README.md")
        print(f"Creating model card...")
        with open(readme_path, "w") as f:
            f.write(create_model_card(model_config))
        print("✅ Model card created!")

        # Create repository
        try:
            print(f"Creating repository...")
            create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print("✅ Repository created/verified!")
        except Exception as e:
            print(f"Repository creation: {e}")

        # Upload
        print("Uploading model files...")
        try:
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload L0 Bouncer {model_config['path']}: {model_config['description']}",
                ignore_patterns=[
                    "*.pyc",
                    "__pycache__",
                    ".git",
                    "checkpoint-*",
                    "trainer_state.json",
                    "training_args.bin"
                ]
            )
            print(f"\n✅ Model uploaded successfully!")
            print(f"🔗 https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")

    print(f"\n{'='*60}")
    print("All uploads complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
