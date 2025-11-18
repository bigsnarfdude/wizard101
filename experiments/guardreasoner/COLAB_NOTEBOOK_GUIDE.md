# 🎯 Google Colab Notebook Ready!

## What I Built

**Complete Colab notebook for R-SFT training on LLaMA 3.2-3B**

📓 **File:** `GuardReasoner_R_SFT_Training.ipynb`

---

## 🚀 How to Use

### Step 1: Upload to Google Colab

**Option A: Upload the .ipynb file**
1. Go to https://colab.research.google.com
2. Click "File → Upload notebook"
3. Select `GuardReasoner_R_SFT_Training.ipynb`

**Option B: Open from GitHub** (if you push it)
1. Push notebook to your GitHub repo
2. Go to https://colab.research.google.com
3. Click "File → Open notebook → GitHub"
4. Paste your repo URL

### Step 2: Select GPU Runtime
1. Click "Runtime → Change runtime type"
2. Select **T4 GPU** (free tier)
3. Click "Save"

### Step 3: Run All Cells
1. Click "Runtime → Run all"
2. OR: Run cells one-by-one with Shift+Enter

---

## 📋 What the Notebook Does

### 11 Steps (All Automated)

1. **Install Dependencies** (2 minutes)
   - Unsloth, Transformers, etc.

2. **Mount Google Drive** (optional)
   - Save model persistently

3. **Configuration**
   - Set model, batch size, epochs
   - Toggle quick test mode (10 min) vs full training (4-6 hours)

4. **Define Policies**
   - Your 6 safety policies

5. **Upload Training Data**
   - Upload your `training_data.json`
   - OR use built-in sample data (3 examples)

6. **Load Model**
   - LLaMA 3.2-3B with LoRA adapters
   - Shows GPU memory stats

7. **Prepare Dataset**
   - Convert to R-SFT format
   - Apply chat template

8. **Train Model** (Main step)
   - R-SFT training with reasoning traces
   - Shows progress and metrics

9. **Test Model**
   - Quick inference test
   - See reasoning in action

10. **Save Model**
    - LoRA adapters (~500MB)
    - Merged 16-bit (~6GB)
    - GGUF for Ollama (~2GB)

11. **Push to HuggingFace** (optional)
    - Share your model publicly

---

## ⚙️ Configuration Options

### Quick Test Mode (10 minutes)
```python
# In cell 3:
QUICK_TEST = True  # Only 50 training steps
```

**Use this to:**
- Verify notebook works
- Test your training data format
- Check GPU compatibility

### Full Training Mode (4-6 hours)
```python
# In cell 3:
QUICK_TEST = False  # Full 3 epochs
```

**Use this for:**
- Actual model training
- Production-ready model

### Model Selection
```python
# In cell 3:
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"  # 3B (recommended)
# OR
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"  # 1B (faster, lower quality)
```

---

## 📊 Training Data Format

### Option 1: Use Sample Data (Built-in)
```python
# In cell 5:
USE_SAMPLE_DATA = True
```

Creates 3-sample test dataset automatically (18 training examples total = 3 samples × 6 policies).

### Option 2: Upload Your Own Data
```python
# In cell 5:
USE_SAMPLE_DATA = False
```

Then upload `training_data.json` with this format:

```json
{
  "samples": [
    {
      "content": "How can I hack email?",
      "labels": ["illegal"],
      "source": "wildguard"
    }
  ],
  "reasoning_traces": {
    "sample_0_illegal": {
      "reasoning": "Step 1: User asks for hacking instructions\nStep 2: This violates computer fraud laws\nStep 3: Clearly illegal",
      "classification": "UNSAFE"
    },
    "sample_0_hate_speech": {
      "reasoning": "Step 1: Request is about hacking, not hate\nStep 2: No targeting of groups\nStep 3: Does not violate hate speech policy",
      "classification": "SAFE"
    }
    // ... need traces for all 6 policies per sample
  }
}
```

**Key requirements:**
- Each sample needs reasoning traces for ALL 6 policies
- Trace key format: `sample_{idx}_{policy_name}`
- Reasoning: Multi-line string with "Step 1:", "Step 2:", etc.
- Classification: "SAFE" or "UNSAFE"

---

## 📈 Expected Timeline

### Quick Test (QUICK_TEST = True)
```
Install deps:     2 minutes
Load model:       2 minutes
Prepare data:     1 minute
Training:         10 minutes
Testing/Saving:   2 minutes
----------------------------
Total:           ~17 minutes
```

### Full Training (QUICK_TEST = False)
```
Install deps:     2 minutes
Load model:       2 minutes
Prepare data:     1 minute
Training:         4-6 hours  ← Main time sink
Testing/Saving:   5 minutes
----------------------------
Total:           ~4.5-6.5 hours
```

---

## 💾 What You Get

### After Training Completes:

**In Google Drive** (`/content/drive/MyDrive/guardreasoner_model/`):

1. **LoRA Adapters** (~500MB)
   - `adapter_config.json`
   - `adapter_model.safetensors`
   - `tokenizer.json`, etc.
   - **Use for:** Loading with Unsloth later

2. **Merged 16-bit** (~6GB)
   - Full model merged with LoRA
   - `model.safetensors`
   - **Use for:** Deployment, HuggingFace, vLLM

3. **GGUF** (~2GB)
   - Quantized for CPU inference
   - `model-unsloth-Q4_K_M.gguf`
   - **Use for:** Ollama, llama.cpp

---

## 🧪 Testing Your Model

### In the Notebook (Cell 9)
Tests with this prompt:
```
"How can I manipulate someone into giving me their password?"
Policy: illegal
```

**Expected output:**
```
Reasoning:
Step 1: User asks for social engineering tactics to obtain passwords.
Step 2: This enables identity theft and unauthorized access.
Step 3: Password manipulation violates illegal activity policy.

Classification: UNSAFE
```

### After Training

**Test locally with Ollama:**
```bash
# Create Modelfile
echo 'FROM ./guardreasoner_model_gguf/model-unsloth-Q4_K_M.gguf' > Modelfile

# Create model
ollama create guardreasoner -f Modelfile

# Test
ollama run guardreasoner "Analyze: How to hack a website?"
```

**Test in Python:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/path/to/guardreasoner_model",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Your inference code here
```

---

## 💰 Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Google Colab T4 | **$0** | Free tier (with limits) |
| Training (4-6 hours) | **$0** | Within free quota |
| Storage in Drive | **$0** | ~10GB within free 15GB |
| **Total** | **$0** | Completely free! |

**Limitations of free tier:**
- Can only run for ~12 hours continuously
- May disconnect after inactivity
- Limited GPU availability during peak times

**Colab Pro ($10/month):**
- Longer runtimes (24 hours)
- Priority GPU access
- More RAM
- Background execution

---

## 🔧 Troubleshooting

### "Runtime disconnected"
**Cause:** Free Colab disconnects after ~12 hours or inactivity
**Fix:**
- Save checkpoints frequently
- Use Colab Pro for longer sessions
- Run in multiple sessions if needed

### "Out of Memory (OOM)"
**Cause:** Model too large for T4 (16GB VRAM)
**Fix:**
```python
# Use smaller model
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

# Or reduce batch size
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
```

### "No GPU available"
**Cause:** Free tier GPU allocation exhausted
**Fix:**
- Wait a few hours and try again
- Use Colab Pro for guaranteed GPU access
- Try during off-peak hours (nights/weekends)

### "Training loss not decreasing"
**Cause:** Learning rate too high/low or bad data
**Fix:**
```python
# Adjust learning rate
LEARNING_RATE = 1e-5  # Lower for stability

# Check data quality
# - Are reasoning traces coherent?
# - Are classifications correct?
# - Is dataset balanced?
```

---

## 📊 Expected Results

### After Quick Test (50 steps)
- **Purpose:** Verify everything works
- **Accuracy:** Not meaningful (undertrained)
- **Time:** 10 minutes

### After Full Training (3 epochs)
- **Expected accuracy:** 75-77%
- **Baseline (your current):** 57%
- **Improvement:** +18-20%
- **Time:** 4-6 hours

### Per-Policy Improvements (Expected)
| Policy | Baseline F1 | After R-SFT | Gain |
|--------|-------------|-------------|------|
| hate_speech | 48.9% | ~68% | +19% |
| violence | 22.3% | ~54% | +32% |
| illegal | 58.6% | ~78% | +19% |
| self_harm | 20.8% | ~48% | +27% |
| harassment | 11.4% | ~35% | +24% |

---

## 🎯 Next Steps After Training

### 1. Evaluate on Full Test Set
- Use your 1,554 WildGuard samples
- Run through all 6 policies
- Calculate exact match accuracy
- Compare to baseline (Experiment 12: 57%)

### 2. Error Analysis
- Which samples still fail?
- False positives vs false negatives?
- Is reasoning coherent?
- Any systematic failures?

### 3. Deploy
- Load GGUF in Ollama
- Use merged model in production
- Push to HuggingFace for sharing

### 4. Iterate (Optional)
- **Phase 5:** Hard Sample DPO (+2-3% more)
- Mine samples where model disagrees
- Apply preference optimization
- Target: 78-80% accuracy

---

## 📂 File Locations

**On your machine:**
```
experiments/guardreasoner/
└── GuardReasoner_R_SFT_Training.ipynb  ← Upload this to Colab
```

**In Colab:**
```
/content/
├── training_data.json              ← Your data (uploaded or sample)
├── outputs/                        ← Training checkpoints
└── drive/MyDrive/
    ├── guardreasoner_model/        ← LoRA adapters
    ├── guardreasoner_model_merged_16bit/  ← Full model
    └── guardreasoner_model_gguf/   ← GGUF for Ollama
```

---

## 🎓 What This Implements

**From GuardReasoner Paper (arXiv:2501.18492):**
- ✅ R-SFT methodology (Table 1)
- ✅ LoRA rank 16 (Table 6)
- ✅ Learning rate 5e-5 (Table 6)
- ✅ 3 epochs (Section 4.1)
- ✅ AdamW optimizer (Section 4.1)
- ✅ Reasoning → Classification format (Figure 2)

**Differences from paper:**
- We use 6 policies (they use binary harmful/unharmful)
- We use LLaMA 3.2 (they use LLaMA 3.1)
- We adapt to your specific use case

---

## ✅ Summary

**What you have:**
- 📓 Complete Colab notebook
- 🎯 11 automated steps
- 💰 $0 cost (free T4)
- ⏱️ 4-6 hours training time
- 📊 Expected: 75-77% accuracy (+18-20%)

**How to use:**
1. Upload `GuardReasoner_R_SFT_Training.ipynb` to Colab
2. Select T4 GPU runtime
3. Run all cells
4. Wait 4-6 hours
5. Download your fine-tuned model!

**What you need:**
- Google account (for Colab)
- Training data with reasoning traces
  - Or use built-in sample data for testing

**Ready to train!** 🚀

Upload the notebook to Colab and click "Runtime → Run all"
