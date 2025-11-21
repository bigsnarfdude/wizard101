# Dataset Documentation

This folder contains all datasets used in the wizard101 safety classification project.

## Folder Structure

```
data/
├── training/       # Small training datasets (<100MB)
├── evaluation/     # Test and validation datasets
├── benchmark/      # Standardized benchmark datasets
└── raw/            # Original source datasets

experiments/guardreasoner/guardreasoner_data/  # Large training datasets (>100MB, not in git)
├── all_combined.json       # 127k samples, 357 MB
├── WildGuardTrainR.json    # 87k samples, 263 MB
├── BeaverTailsTrainR.json  # 27k samples, 58 MB
├── AegisTrainR.json        # 11k samples, 26 MB
└── ToxicChatTrainR.json    # 3k samples, 5.5 MB
```

**Note**: Large training datasets (>100MB) are kept in `experiments/guardreasoner/guardreasoner_data/` and excluded from git due to GitHub's file size limits.

## Dataset Summary

| Category | Files | Total Samples | Size |
|----------|-------|---------------|------|
| Training | 8 | ~170,000 | ~350 MB |
| Evaluation | 4 | ~17,000 | ~8 MB |
| Benchmark | 6 | ~21,000 | ~10 MB |
| Raw | 2 | ~1,000 | ~0.5 MB |

---

## Training Datasets

### `all_combined.json` (127,544 samples, 357 MB)
**L0 BOUNCER TRAINING DATASET** - This is what we trained the L0 classifier on. Combined from all sources with 3-task reasoning format.

Format:
```json
{
  "instruction": "System prompt with 3-task classification...",
  "input": "Human user:\n{prompt}\n\nAI assistant:\n{response}\n\n",
  "output": "Reasoning trace + labels",
  "prompt_harm_label": "harmful|unharmful",
  "response_refusal_label": "refusal|compliance",
  "response_harm_label": "harmful|unharmful",
  "source_split": "WildGuardTrainR|BeaverTailsTrainR|..."
}
```

### `WildGuardTrainR.json` (86,759 samples, 263 MB)
WildGuard dataset with GPT-4o reasoning traces. Primary safety benchmark source.

### `BeaverTailsTrainR.json` (27,186 samples, 58 MB)
BeaverTails jailbreak dataset with reasoning traces.

### `AegisTrainR.json` (10,798 samples, 26 MB)
Aegis safety framework dataset with reasoning traces.

### `ToxicChatTrainR.json` (2,801 samples, 5.5 MB)
ToxicChat toxicity detection dataset with reasoning traces.

### `train_12k.json` (12,000 samples, 5.7 MB)
Earlier experiment - smaller training set (superseded by all_combined.json).

Format:
```json
{
  "text": "prompt text",
  "label": "harmful|safe"
}
```

### `mega_train.json` (2,540 samples, 1 MB)
Earlier experiment - extended samples (superseded by all_combined.json).

### `combined_train.json` (800 samples)
Earlier experiment - simple combined dataset (superseded by all_combined.json).

---

## Evaluation Datasets

### `guardreasoner_test_10k.json` (10,000 samples, 5 MB)
Large test set for GuardReasoner evaluation.

### `guardreasoner_test_5k.json` (5,000 samples, 2.5 MB)
Medium test set for faster evaluation cycles.

### `wildguard_full_benchmark.json` (1,554 samples, 0.9 MB)
Original WildGuard test set - **primary evaluation benchmark**.

Format:
```json
{
  "prompt": "user prompt",
  "response": "AI response (optional)",
  "label": "harmful|benign",
  "subcategory": "specific harm type"
}
```

### `combined_test.json` (200 samples)
Validation set matching combined_train.json format.

---

## Benchmark Datasets

### `combined_benchmark.json` (10,384 samples, 4.7 MB)
**Comprehensive benchmark** combining multiple sources for thorough evaluation.

### `toxicchat_test.json` (5,083 samples, 1.3 MB)
ToxicChat official test split for toxicity detection.

### `beavertails_30k.json` (3,021 samples, 2 MB)
BeaverTails subset for jailbreak detection evaluation.

### `openai_moderation.json` (1,680 samples, 1.3 MB)
OpenAI moderation API comparison dataset.

### `harmbench_test.json` (500 samples)
HarmBench structured harmful behaviors test set.

### `simplesafetytests.json` (100 samples)
Basic safety sanity check dataset.

---

## Raw Datasets

### `harmless_alpaca.json` (500 samples)
Benign instruction-following prompts from Alpaca dataset.

### `harmful_behaviors.json` (500 samples)
Harmful/adversarial prompts for safety testing.

---

## Data Sources

| Source | URL | License |
|--------|-----|---------|
| WildGuard | [allenai/wildguard](https://huggingface.co/datasets/allenai/wildguard) | Apache 2.0 |
| BeaverTails | [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | CC BY-NC 4.0 |
| ToxicChat | [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | CC BY-NC 4.0 |
| Aegis | [nvidia/Aegis-AI-Content-Safety-Dataset](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset) | CC BY 4.0 |
| HarmBench | [cais/HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT |

---

## Usage Examples

### Load Training Data
```python
import json

# Load full training set
with open('data/training/all_combined.json') as f:
    train_data = json.load(f)
print(f"Training samples: {len(train_data):,}")

# Load specific source
with open('data/training/WildGuardTrainR.json') as f:
    wildguard = json.load(f)
```

### Load Evaluation Data
```python
# Load primary benchmark
with open('data/evaluation/wildguard_full_benchmark.json') as f:
    test_data = json.load(f)

# Evaluate
for sample in test_data:
    prompt = sample['prompt']
    label = sample['label']
    # ... run model prediction
```

### Load Benchmark Suite
```python
import os
import json

benchmark_dir = 'data/benchmark'
results = {}

for filename in os.listdir(benchmark_dir):
    if filename.endswith('.json'):
        with open(os.path.join(benchmark_dir, filename)) as f:
            data = json.load(f)
        results[filename] = len(data)
        print(f"{filename}: {len(data):,} samples")
```

---

## Label Mappings

Different datasets use different label conventions:

| Dataset | Safe Label | Harmful Label |
|---------|------------|---------------|
| GuardReasoner | unharmful | harmful |
| WildGuard | benign | harmful |
| Cascade | safe | harmful |
| SimpleSafety | safe | unsafe |

Normalize labels before training/evaluation:
```python
def normalize_label(label):
    if label.lower() in ['safe', 'benign', 'unharmful', 'harmless']:
        return 'safe'
    return 'harmful'
```

---

## Dataset Statistics

### By Harm Category (WildGuard)
- Violence: ~15%
- Hate/Harassment: ~12%
- Sexual Content: ~10%
- Self-Harm: ~8%
- Illegal Activity: ~20%
- Other: ~35%

### Label Distribution
- Training: ~50% harmful, ~50% safe (balanced)
- Evaluation: ~40% harmful, ~60% safe (realistic)

---

## Notes

1. **3-Task Format**: GuardReasoner datasets include three classification tasks:
   - Task 1: Is the prompt harmful?
   - Task 2: Did the AI refuse?
   - Task 3: Is the response harmful?

2. **Reasoning Traces**: Training datasets include GPT-4o generated reasoning traces for R-SFT (Reasoning Supervised Fine-Tuning).

3. **No PII**: All datasets have been reviewed - no real personal information or API keys are included.

4. **Duplicates Removed**: CSV, JSONL, and TXT format duplicates have been removed. Only JSON files are kept.
