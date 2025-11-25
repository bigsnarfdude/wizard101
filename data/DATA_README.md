# Dataset Documentation

This folder contains all datasets used in the wizard101 safety classification project.

## Folder Structure

```
data/
├── training/       # Training datasets
├── evaluation/     # Test and validation datasets
├── benchmark/      # Standardized benchmark datasets
├── raw/            # Original source datasets
└── archived/       # Historical R&D datasets (~710 MB)
```

---

## Current Data Summary

| Category | Files | Total Samples | Size |
|----------|-------|---------------|------|
| Training | 3 | 15,340 | ~7 MB |
| Evaluation | 5 | 17,204 | ~8.5 MB |
| Benchmark | 10 | 111,167 | ~56 MB |
| Raw | 2 | ~800 | ~58 KB |
| Archived | 7 | ~127,544 | ~710 MB |

---

## Training Datasets (data/training/)

### `train_12k.json` (12,000 samples, 5.7 MB)
Primary training dataset with binary labels.

Format:
```json
{
  "text": "prompt text",
  "label": "harmful|safe"
}
```

### `mega_train.json` (2,540 samples, 1 MB)
Extended training samples.

### `combined_train.json` (800 samples, 111 KB)
Simple combined dataset for quick experiments.

---

## Evaluation Datasets (data/evaluation/)

### `guardreasoner_test_10k.json` (10,000 samples, 5 MB)
Large test set for GuardReasoner evaluation.

### `guardreasoner_test_5k.json` (5,000 samples, 2.5 MB)
Medium test set for faster evaluation cycles.

### `wildguard_full_benchmark.json` (1,554 samples, 894 KB)
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

### `xstest.json` (450 samples, 66 KB)
XSTest benchmark for over-refusal testing (false positives).

### `combined_test.json` (200 samples, 28 KB)
Validation set matching combined_train.json format.

---

## Benchmark Datasets (data/benchmark/)

### `wildjailbreak.json` (88,444 samples, 46 MB)
Large-scale jailbreak detection benchmark from Allen AI.

### `combined_benchmark.json` (10,384 samples, 4.7 MB)
Comprehensive benchmark combining multiple sources.

### `toxicchat_test.json` (5,083 samples, 1.3 MB)
ToxicChat official test split for toxicity detection.

### `beavertails_30k.json` (3,021 samples, 2 MB)
BeaverTails subset for jailbreak detection evaluation.

### `openai_moderation.json` (1,680 samples, 1.3 MB)
OpenAI moderation API comparison dataset.

### `sgbench.json` (1,442 samples, 178 KB)
Safety benchmark dataset.

### `harmbench_test.json` (500 samples, 88 KB)
HarmBench structured harmful behaviors test set.

### `strongreject.json` (313 samples, 82 KB)
StrongREJECT benchmark for refusal evaluation.

### `jailbreakbench.json` (200 samples, 25 KB)
JailbreakBench evaluation dataset.

### `simplesafetytests.json` (100 samples, 18 KB)
Basic safety sanity check dataset.

---

## Raw Datasets (data/raw/)

### `harmless_alpaca.json` (~400 samples, 26 KB)
Benign instruction-following prompts from Alpaca dataset.

### `harmful_behaviors.json` (~400 samples, 32 KB)
Harmful/adversarial prompts for safety testing.

---

## Archived Datasets (data/archived/)

Historical datasets from GuardReasoner R-SFT training experiments (Nov 2024).

### `all_combined.json` (127,544 samples, 357 MB)
**PRIMARY TRAINING DATASET** for GuardReasoner experiments.
Combined from all sources with 3-task reasoning format.

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
WildGuard dataset with GPT-4o reasoning traces for R-SFT.

### `BeaverTailsTrainR.json` (27,186 samples, 58 MB)
BeaverTails jailbreak dataset with reasoning traces.

### `AegisTrainR.json` (10,798 samples, 26 MB)
Aegis safety framework dataset with reasoning traces.

### `ToxicChatTrainR.json` (2,801 samples, 5.5 MB)
ToxicChat toxicity detection dataset with reasoning traces.

### `failure_analysis_full.json` (17 MB)
DLP false negative analysis from early experiments.

---

## Data Sources

| Source | URL | License |
|--------|-----|---------|
| WildGuard | [allenai/wildguard](https://huggingface.co/datasets/allenai/wildguard) | Apache 2.0 |
| BeaverTails | [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | CC BY-NC 4.0 |
| ToxicChat | [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | CC BY-NC 4.0 |
| Aegis | [nvidia/Aegis-AI-Content-Safety-Dataset](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset) | CC BY 4.0 |
| HarmBench | [cais/HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT |
| WildJailbreak | [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) | Apache 2.0 |
| StrongREJECT | [dsbowen/strongreject](https://huggingface.co/datasets/dsbowen/strongreject) | MIT |
| JailbreakBench | [jailbreakbench](https://github.com/JailbreakBench/jailbreakbench) | MIT |
| XSTest | [Paul/XSTest](https://huggingface.co/datasets/Paul/XSTest) | MIT |

---

## Usage Examples

### Load Training Data
```python
import json

with open('data/training/train_12k.json') as f:
    train_data = json.load(f)
print(f"Training samples: {len(train_data):,}")
```

### Load Archived R-SFT Data
```python
# Large files - load from data/archived/
with open('data/archived/all_combined.json') as f:
    rsft_data = json.load(f)
print(f"R-SFT samples: {len(rsft_data):,}")
```

### Load Evaluation Data
```python
with open('data/evaluation/wildguard_full_benchmark.json') as f:
    test_data = json.load(f)

for sample in test_data:
    prompt = sample['prompt']
    label = sample['label']
    # ... run model prediction
```

---

## Label Mappings

Different datasets use different label conventions:

| Dataset | Safe Label | Harmful Label |
|---------|------------|---------------|
| WildGuard | benign | harmful |
| train_12k | safe | harmful |
| GuardReasoner | unharmful | harmful |
| SimpleSafety | safe | unsafe |

Normalize labels before training/evaluation:
```python
def normalize_label(label):
    if label.lower() in ['safe', 'benign', 'unharmful', 'harmless']:
        return 'safe'
    return 'harmful'
```

---

## Notes

1. **Archived Datasets**: R-SFT training datasets (~710MB) are in `data/archived/` - used for GuardReasoner LoRA fine-tuning experiments.

2. **3-Task Format**: GuardReasoner datasets include three classification tasks:
   - Task 1: Is the prompt harmful?
   - Task 2: Did the AI refuse?
   - Task 3: Is the response harmful?

3. **Reasoning Traces**: Archived training datasets include GPT-4o generated reasoning traces for R-SFT (Reasoning Supervised Fine-Tuning).

4. **No PII**: All datasets have been reviewed - no real personal information or API keys are included.
