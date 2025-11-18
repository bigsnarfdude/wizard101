# GuardReasoner: CORRECTED Training Plan

## Paper's ACTUAL 2-Stage Approach

### Stage 1: Reasoning Supervised Fine-Tuning (R-SFT) ✅ (We did this!)

**Data Preparation:**
- **Seed datasets**: WildGuardTrain, AegisTrain, BeaverTailsTrain, ToxicChatTrain
- **Synthesis**: Use GPT-4o to generate step-by-step reasoning
- **Final dataset**: GuardReasonerTrain (~127K samples, 460K reasoning steps, ~3.6 steps/sample)

**Training:**
- **Base models**: LLaMA 3.2 1B/3B or LLaMA 3.1 8B
- **Epochs**: 3 (NOT 5!) ⚠️
- **Objective**: Train model to output reasoning steps + classification
- **Result**: Model M_R-SFT with basic reasoning capability

**Our Status:**
- ✅ Dataset: 11,396 samples (smaller but similar approach)
- ✅ Model: Llama-3.2-3B-Instruct
- ⚠️ Epochs: 1 (need 2 more to match paper's 3 epochs)
- ✅ Objective: Reasoning + classification ✓

---

### Stage 2: Hard Sample DPO (HS-DPO) 📋 (Next step!)

This is NOT standard GRPO - it's a specialized DPO variant!

#### Hard Sample Mining

**Step 1: Generate Multiple Outputs**
```python
for sample in dataset:
    outputs = []
    for i in range(k):  # k = 4 to 8
        output = M_R-SFT.generate(sample, temperature=0.8)
        outputs.append(output)

    # Classify outputs
    correct = [o for o in outputs if is_correct(o)]
    incorrect = [o for o in outputs if not is_correct(o)]

    # Keep only ambiguous samples
    if len(correct) > 0 and len(incorrect) > 0:
        ambiguous_samples.append((sample, correct, incorrect))
```

**Step 2: Ensemble Hard Sample Mining**
- Train 3 different M_R-SFT variants on data subsets
- Each model finds its own hard samples
- Merge: H_ensemble = hard samples from all 3 models
- **Why?** Different models disagree on different samples → more diverse hard set

#### HS-DPO Training

**Preference Pairs:**
```python
for sample, correct_outputs, incorrect_outputs in H_ensemble:
    # Positive examples: correct reasoning + correct label
    chosen = random.choice(correct_outputs)

    # Negative examples: incorrect reasoning OR incorrect label
    rejected = random.choice(incorrect_outputs)

    # Adaptive weight based on difficulty
    k_correct = len(correct_outputs)
    k_incorrect = len(incorrect_outputs)
    weight = 1 + normalize(k_incorrect - k_correct, gamma)

    dpo_pairs.append({
        'prompt': sample,
        'chosen': chosen,
        'rejected': rejected,
        'weight': weight
    })
```

**Key Innovation: Adaptive Weighting**
- **Hard samples** (many incorrect, few correct) → **higher weight**
- **Easy samples** (many correct, few incorrect) → **lower weight**
- Formula: `α = 1 + Norm(k_incorrect - k_correct, γ)`

**Training Hyperparameters:**
- Learning rate: 5e-6 (10× lower than R-SFT)
- Batch size: 256
- Epochs: 2
- β (KL constraint): 0.01
- R-SFT loss mix ratio: 2 (mix in some SFT loss to maintain reasoning quality)

---

## Why Not GRPO?

**GRPO (Group Relative Policy Optimization):**
- On-policy RL algorithm
- Requires online sampling during training
- Group-based reward computation
- More complex, less stable

**HS-DPO (Hard Sample Direct Preference Optimization):**
- Off-policy (uses pre-generated samples)
- Direct preference learning
- Focuses on decision boundary
- Adaptive sample weighting
- **Simpler and more effective** for this task

---

## Our Implementation Plan

### Phase 1: Complete R-SFT (This Week)
**Current status:** 1/3 epochs complete

**Tasks:**
1. ✅ Evaluate 1-epoch model (running now)
2. 📋 Continue training for 2 more epochs
3. 📋 Evaluate 3-epoch model
4. 📋 Save as M_R-SFT checkpoint

**Expected time:** ~16 hours (2 epochs × 8 hours)

---

### Phase 2: Hard Sample Mining (Next Week)

#### Step 1: Generate Multiple Outputs per Sample
```python
# experiments/guardreasoner/mine_hard_samples.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm

# Load M_R-SFT
model = load_model("exp_18_rsft_lora")
tokenizer = load_tokenizer()

# Load test data
test_data = load_json("data/combined_test.json")

k = 4  # Generate 4 outputs per sample
temperature = 0.8

hard_samples = []

for sample in tqdm(test_data):
    outputs = []

    for i in range(k):
        output = model.generate(
            sample['text'],
            temperature=temperature,
            max_new_tokens=512
        )

        # Parse output
        reasoning, label = parse_output(output)
        correct = (label == sample['label'])

        outputs.append({
            'reasoning': reasoning,
            'label': label,
            'correct': correct
        })

    # Check if ambiguous
    correct_count = sum(1 for o in outputs if o['correct'])
    incorrect_count = k - correct_count

    if correct_count > 0 and incorrect_count > 0:
        # This is a hard sample!
        hard_samples.append({
            'prompt': sample['text'],
            'ground_truth': sample['label'],
            'outputs': outputs,
            'k_correct': correct_count,
            'k_incorrect': incorrect_count,
            'difficulty': incorrect_count - correct_count
        })

# Save hard samples
save_json(hard_samples, "data/hard_samples_exp18.json")
print(f"Found {len(hard_samples)} hard samples out of {len(test_data)}")
```

**Expected output:**
- ~500-1000 hard samples (from 11k total)
- These are decision boundary samples

#### Step 2: Create DPO Preference Dataset
```python
# experiments/guardreasoner/create_dpo_dataset.py

import random
import json

hard_samples = load_json("data/hard_samples_exp18.json")

dpo_pairs = []
gamma = 0.5  # Normalization parameter

for sample in hard_samples:
    correct_outputs = [o for o in sample['outputs'] if o['correct']]
    incorrect_outputs = [o for o in sample['outputs'] if not o['correct']]

    # Create preference pair
    chosen = random.choice(correct_outputs)
    rejected = random.choice(incorrect_outputs)

    # Calculate adaptive weight
    difficulty = sample['k_incorrect'] - sample['k_correct']
    weight = 1 + normalize(difficulty, gamma)

    dpo_pairs.append({
        'prompt': sample['prompt'],
        'chosen': format_response(chosen['reasoning'], chosen['label']),
        'rejected': format_response(rejected['reasoning'], rejected['label']),
        'weight': weight
    })

# Save DPO dataset
save_json(dpo_pairs, "data/guardreasoner_dpo_pairs.json")
print(f"Created {len(dpo_pairs)} DPO preference pairs")
```

---

### Phase 3: HS-DPO Training (Week 3)

```python
# experiments/guardreasoner/train_exp_19_hsdpo.py

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load R-SFT model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, "models/exp_18_rsft_lora")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")

# Load DPO dataset
dpo_dataset = load_json("data/guardreasoner_dpo_pairs.json")

# HS-DPO Configuration
config = DPOConfig(
    learning_rate=5e-6,          # 10× lower than R-SFT
    per_device_train_batch_size=4,
    gradient_accumulation_steps=64,  # effective batch = 256
    num_train_epochs=2,
    beta=0.01,                   # KL constraint
    loss_type="sigmoid",         # Standard DPO loss
    max_length=2048,
    max_prompt_length=512,

    # R-SFT loss mixing
    sft_loss_weight=2.0,         # Mix in SFT loss to maintain quality

    # Adaptive weighting
    use_weighting=True,          # Use sample weights
)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save final model
trainer.save_model("models/exp_19_hsdpo_lora")
```

**Expected time:** ~4-6 hours (2 epochs)

---

### Phase 4: Evaluation & Comparison

**Compare 3 models:**
1. **Exp 18 (1-epoch R-SFT)**: Baseline
2. **Exp 18 continued (3-epoch R-SFT)**: Full R-SFT
3. **Exp 19 (HS-DPO)**: R-SFT + HS-DPO

**Metrics:**
- Overall accuracy
- Per-class F1 scores
- Hard sample accuracy (most important!)
- Reasoning quality (manual inspection)

---

## Key Differences from Our Original Plan

| Aspect | Our Original Plan | Paper's Actual Method |
|--------|------------------|----------------------|
| **Stage 1** | R-SFT for 5 epochs | R-SFT for 3 epochs |
| **Stage 2** | DPO on reasoning quality | HS-DPO on hard samples |
| **Hard Samples** | Generate from scratch | Mine from R-SFT model |
| **Weighting** | Equal weights | Adaptive by difficulty |
| **Focus** | Reasoning quality | Decision boundary |
| **Loss** | Pure DPO | DPO + SFT mix (2:1) |

---

## Timeline

### Week 1 (Current)
- ✅ Day 1: 1-epoch R-SFT training complete
- ⏳ Day 1: Quick evaluation running
- 📋 Day 2-3: Continue to 3 epochs R-SFT
- 📋 Day 4: Full evaluation

### Week 2
- 📋 Day 1-2: Hard sample mining (4 outputs × 11k samples)
- 📋 Day 3: Create DPO preference dataset
- 📋 Day 4: Verify dataset quality

### Week 3
- 📋 Day 1-2: HS-DPO training (2 epochs)
- 📋 Day 3-4: Evaluation & comparison
- 📋 Day 5: Analysis & write-up

---

## Success Criteria

### R-SFT (3 epochs)
- ✅ Accuracy > 55% on test set
- ✅ Generates coherent reasoning
- ✅ Stable training (loss converges)

### HS-DPO
- ✅ Hard sample accuracy improves by 5-10%
- ✅ Maintains overall accuracy (±2%)
- ✅ Reasoning quality maintained or improved
- ✅ Outperforms 3-epoch R-SFT baseline

---

## Research Contribution

If successful, we validate:
1. **2-stage training works** on smaller model (3B vs 8B)
2. **HS-DPO is effective** for safety classification
3. **Memory-efficient** (4-bit quantization + LoRA)
4. **Accessible** (single GPU, 24-hour training)

**Paper comparison:**
- Paper: 8B model, 8×A100, ~20 hours
- Ours: 3B model, 1×GPU (4-bit), ~24 hours
- Trade-off: Slightly lower accuracy for 8× less compute

---

**Status**: Ready to continue R-SFT training to 3 epochs
**Next**: Wait for eval results, then decide on training continuation
