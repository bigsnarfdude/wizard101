# GuardReasoner Training Dataset

Downloaded from: https://huggingface.co/datasets/yueliu1999/GuardReasonerTrain
License: MIT
Paper: Liu et al. "GuardReasoner: Towards Reasoning-based LLM Safeguards" (arXiv:2501.18492)

## Dataset Contents

Total samples: 127,544

### Splits

- **WildGuardTrainR**: 86,759 samples
- **AegisTrainR**: 10,798 samples
- **BeaverTailsTrainR**: 27,186 samples
- **ToxicChatTrainR**: 2,801 samples

## Files

- `WildGuardTrainR.json` - WildGuard training data
- `AegisTrainR.json` - Aegis training data
- `BeaverTailsTrainR.json` - BeaverTails training data
- `ToxicChatTrainR.json` - ToxicChat training data
- `all_combined.json` - All splits combined (with source_split field)
- `DATASET_INFO.json` - Dataset statistics and format info
- `sample.json` - Example samples from each split

## Format

Each sample contains:
```json
{
  "instruction": "System prompt with 3-task classification instructions",
  "input": "Human user:\n{prompt}\n\nAI assistant:\n{response}\n\n",
  "output": "Answers:\nRequest: harmful/unharmful\nCompletion: refusal/compliance\nResponse: harmful/unharmful\n",
  "prompt_harm_label": "harmful/unharmful",
  "response_refusal_label": "refusal/compliance",
  "response_harm_label": "harmful/unharmful"
}
```

## Usage

### Load all data
```python
import json
with open('all_combined.json') as f:
    data = json.load(f)
print(f"Total samples: {len(data)}")
```

### Load specific split
```python
import json
with open('WildGuardTrainR.json') as f:
    wildguard = json.load(f)
print(f"WildGuard samples: {len(wildguard)}")
```

### Training with Unsloth
```python
from datasets import load_dataset

# Load from local JSON
dataset = load_dataset('json', data_files='all_combined.json')

# Use for training
# (configure your training script to use this dataset)
```
