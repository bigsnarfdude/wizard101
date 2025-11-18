# R-SFT Explained: What We're Building

## What is R-SFT?

**R-SFT** = **Reasoning Supervised Fine-Tuning**

A training method from the GuardReasoner paper where models learn to:
1. **Reason** step-by-step about safety violations
2. **Then** make a classification decision

Instead of just training: `Input → UNSAFE`
We train: `Input → Reasoning steps → UNSAFE`

---

## Why Does This Work?

### Current Problem (Your Baseline)
```
Prompt: "How can I annihilate my competition in the market?"

Model thinks:
- Sees "annihilate" → UNSAFE! ❌
- Sees "competition" → UNSAFE! ❌
- No reasoning, just keyword matching

Output: UNSAFE (FALSE POSITIVE)
Actual: SAFE (business metaphor)
```

### With R-SFT
```
Prompt: "How can I annihilate my competition in the market?"

Model thinks:
Step 1: User is asking about business strategy
Step 2: "Annihilate" is metaphorical, not literal violence
Step 3: Competitive business is legal and common
Conclusion: No actual harm, no policy violation

Output: SAFE ✓
Actual: SAFE ✓
```

**Key insight:** Forcing the model to explain its reasoning BEFORE deciding makes it more careful and accurate.

---

## The Training Process

### Step 1: Create Reasoning Traces

For each sample + policy, create reasoning like:

```
Sample: "Write a guide on hacking websites"
Policy: illegal

Reasoning trace:
Step 1: User requests instructions for unauthorized system access
Step 2: Website hacking violates computer fraud laws
Step 3: This clearly falls under illegal activity policy
Classification: UNSAFE

Sample: "Write a guide on website security best practices"
Policy: illegal

Reasoning trace:
Step 1: User requests security education content
Step 2: Learning defensive security is legitimate and legal
Step 3: This does not violate illegal activity policy
Classification: SAFE
```

### Step 2: Format as Training Data

Convert to conversation format:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Analyze this content for illegal activity policy violations:\n\nContent: Write a guide on hacking websites"
    },
    {
      "role": "assistant",
      "content": "Reasoning:\nStep 1: User requests instructions for unauthorized system access\nStep 2: Website hacking violates computer fraud laws\nStep 3: This clearly falls under illegal activity policy\n\nClassification: UNSAFE"
    }
  ]
}
```

### Step 3: Fine-Tune with LoRA

Train model on these conversation pairs:
- Use LoRA (efficient fine-tuning, ~1-10% of parameters)
- Train for 3 epochs
- Only backpropagate on assistant responses (not user prompts)
- Takes ~4-6 hours on T4 GPU

### Step 4: Inference

At test time, model now outputs reasoning automatically:

```
Input: "How to manipulate someone into staying in a relationship?"

Model output:
Reasoning:
Step 1: User asks for psychological manipulation tactics
Step 2: This enables emotional abuse and control
Step 3: Violates harassment policy (targeting individuals for harm)

Classification: UNSAFE
Violated policies: harassment
```

---

## Technical Details from GuardReasoner Paper

### Architecture
- **Base model:** LLaMA 3.2 (1B or 3B) or LLaMA 3.1 (8B)
- **Training method:** Supervised Fine-Tuning (SFT) with LoRA adapters
- **LoRA rank:** 16 (affects how many parameters are tuned)
- **Training data:** 127K samples with GPT-4o generated reasoning traces

### Hyperparameters
| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 5e-5 | Standard for LLaMA fine-tuning |
| Batch size | 128 (effective) | Via gradient accumulation |
| Epochs | 3 | Avoids overfitting |
| Max sequence length | 2048 tokens | Fits reasoning + classification |
| Optimizer | AdamW 8-bit | Memory efficient |
| LoRA rank | 16 | Balance between efficiency and capacity |

### Training Time
- **LLaMA 3.2-3B on T4:** ~4-6 hours for 1,554 samples
- **LLaMA 3.1-8B on H100:** ~25 GPU hours for 127K samples

### Performance Gains (from paper)
- **R-SFT alone:** +5.74% F1 improvement
- **R-SFT + Hard Sample DPO:** +8.54% F1 total
- **Final result:** 84.09% F1 (vs 75.35% for GPT-4o baseline)

---

## What You'll Build

### Training Script Components

1. **Data Loader**
   - Reads WildGuard samples + reasoning traces
   - Formats as conversations
   - Creates train/eval splits

2. **Model Setup**
   - Loads LLaMA 3.2 (1B or 3B) with 4-bit quantization
   - Adds LoRA adapters to attention + MLP layers
   - Configures chat template

3. **Trainer**
   - Uses Unsloth for fast training
   - Trains only on assistant responses (masks user prompts)
   - Saves checkpoints every N steps

4. **Inference Engine**
   - Formats test prompts
   - Gets model reasoning + classification
   - Parses output

5. **Evaluation**
   - Compares predicted vs true labels
   - Calculates accuracy, precision, recall, F1
   - Generates error analysis

---

## What Do You Need?

### Required
- ✅ **Training data:** WildGuard samples (you have 1,554)
- ✅ **Reasoning traces:** GPT-4 generated or manual (need to create)
- ✅ **GPU:** Google Colab T4 (free) or A100 (paid)
- ✅ **Time:** 4-6 hours for training
- ✅ **Storage:** 10GB for model checkpoints

### Optional but Recommended
- Google Drive (save checkpoints)
- HuggingFace account (share model)
- WandB account (track training metrics)

---

## Expected Results

### Accuracy Improvement
```
Baseline (Exp 12):     57.5%
After R-SFT (Exp 36):  75-77% (+18-20%)
After DPO (Exp 41):    78-80% (+2-3% more)
```

### Per-Policy Improvements
| Policy | Baseline F1 | After R-SFT | Gain |
|--------|-------------|-------------|------|
| hate_speech | 48.9% | 68% | +19% |
| violence | 22.3% | 54% | +32% |
| illegal | 58.6% | 78% | +19% |
| self_harm | 20.8% | 48% | +27% |
| harassment | 11.4% | 35% | +24% |

### False Positive/Negative Rates
```
Baseline FP: 7.5% → After R-SFT: 4.2% (better precision)
Baseline FN: 7.4% → After R-SFT: 5.1% (better recall)
```

---

## Why Use Our Script Instead of GuardReasoner Code?

### GuardReasoner GitHub Code
- ❌ Uses their dataset (harmful/unharmful binary)
- ❌ Doesn't match our 6-policy framework
- ❌ Requires converting data format
- ✅ Good reference for methodology

### Our Script
- ✅ Uses YOUR WildGuard data (1,554 samples)
- ✅ Matches YOUR 6 policies exactly
- ✅ Outputs per-policy classifications
- ✅ Easy to run on free Colab
- ✅ Saves in formats you need (LoRA, merged, GGUF)

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│ 1. PREPARE DATA                                         │
│    - Use your 1,554 WildGuard samples                  │
│    - Generate reasoning traces (GPT-4 or manual)       │
│    - Format: samples + reasoning_traces JSON           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. UPLOAD TO COLAB                                      │
│    - Upload training_data.json                         │
│    - Upload rsft_training_colab.py                     │
│    - Configure settings (model, epochs, etc.)          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. TRAIN MODEL (4-6 hours)                             │
│    - Fine-tune LLaMA 3.2 with LoRA                     │
│    - Learn to output reasoning → classification        │
│    - Monitor GPU memory and loss                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. EVALUATE                                             │
│    - Run on WildGuard test set                         │
│    - Measure accuracy (target: 75-77%)                 │
│    - Compare to baseline (57%)                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 5. SAVE & DEPLOY                                        │
│    - Save LoRA adapters                                │
│    - Export to GGUF (for Ollama)                       │
│    - Push to HuggingFace (optional)                    │
└─────────────────────────────────────────────────────────┘
```

---

## Common Questions

### Q: Do I need reasoning traces for all 1,554 samples?
**A:** Yes, for each sample × 6 policies = 9,324 reasoning traces total. But you can:
- Start with 100 samples to test (~$10 GPT-4 cost)
- Generate in batches
- Use the trained model to generate more traces (bootstrapping)

### Q: Can I use GuardReasoner's existing reasoning traces?
**A:** Partially. Their traces are for binary (harmful/unharmful) classification. You'd need to:
1. Download their dataset (127K samples)
2. Convert to your 6-policy format
3. Re-label which policies each sample violates
4. Generate additional reasoning traces for missing policies

### Q: What if I don't have GPT-4 API access?
**A:** Options:
1. Use Claude API (similar quality)
2. Use open models like LLaMA 70B or Mixtral 8x22B
3. Manually write reasoning for subset (labor intensive)
4. Use simpler reasoning templates (lower quality but free)

### Q: How much does this cost?
**A:**
- **GPU:** $0 (Colab free T4) or ~$10 (Colab Pro A100)
- **Reasoning generation:** ~$50-100 for GPT-4 on 1,554 samples
- **Total:** $50-110

### Q: How long does the whole process take?
**A:**
- Reasoning generation: 2-3 hours (parallel GPT-4 calls)
- Training: 4-6 hours (T4) or 1-2 hours (A100)
- Evaluation: 4 hours (same as current experiments)
- **Total:** ~10-13 hours end-to-end

### Q: What's the difference between R-SFT and DPO?
**A:**
- **R-SFT:** Train on correct reasoning traces → classification
- **DPO:** Train on preferences (chosen vs rejected reasoning)
- **Order:** R-SFT first (Phase 4), then DPO on hard samples (Phase 5)
- **Gains:** R-SFT gives +18%, DPO adds another +2-3%

---

## Next Steps

1. ✅ **We have:** Training script ready (`rsft_training_colab.py`)
2. ⏳ **Need:** Reasoning traces for your 1,554 samples
3. 🎯 **Next:** Generate reasoning traces (Experiment 31 in plan)
4. 🚀 **Then:** Run R-SFT training (Experiment 36 in plan)

**Current Status:** Ready to generate reasoning traces once you have GPT-4 access or budget approval.

---

## Files Created

1. ✅ `rsft_training_colab.py` - Complete training script
2. ✅ `R-SFT_Training_Colab.md` - Colab notebook guide
3. ✅ `RSFT_EXPLAINED.md` - This explanation document

**All ready for T4 Colab!**
