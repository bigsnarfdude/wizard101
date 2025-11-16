# Datasets for Multi-Policy Safety Evaluation

## Quick Start

```bash
cd /Users/vincent/development/wizard101/experiments

# Generate test dataset (100 samples)
python load_datasets.py

# This creates: test_dataset.json
```

---

## Available Datasets

### 1. ToxicChat (HuggingFace) ✅ WORKING

**What:** Toxicity detection dataset from LMSYS
**Coverage:** Hate speech, harassment
**Size:** 10,166 samples (test set)
**HuggingFace:** `lmsys/toxic-chat`

**Already used in your llm-abuse-patterns experiments!**

```python
from datasets import load_dataset

ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
test_data = ds['test']  # 10,166 samples

# Fields:
# - user_input: str (the content)
# - toxicity: int (0=safe, 1=toxic)
# - jailbreaking: int (0=normal, 1=jailbreak)
```

**Pros:**
- ✅ Free, public, curated
- ✅ You already have experience with it (73.3% F1)
- ✅ Good for hate_speech + harassment policies

**Cons:**
- ❌ Only 2 of 6 policies covered
- ❌ Single-label (not multi-policy)

---

### 2. OpenAI Moderation Benchmark (Internal) ❌ NOT PUBLIC

**What:** OpenAI's internal multi-policy safety benchmark
**Coverage:** All 6 categories (hate, violence, self-harm, sexual/minors, harassment, illegal)
**Size:** Unknown (not publicly released)
**Access:** Not available

**This is what OpenAI uses for their 82.9% F1 score**

**Alternatives:**
- Use OpenAI Moderation API to label your own data
- Create synthetic multi-policy dataset (see load_datasets.py)
- Use combination of public datasets

---

### 3. StrongREJECT (Jailbreak) ⚠️ CHECK AVAILABILITY

**What:** Jailbreak robustness benchmark
**Coverage:** All 6 categories (adversarial)
**Size:** 313 jailbreak attempts
**Paper:** https://arxiv.org/abs/2402.10260

**HuggingFace (if available):**
```python
ds = load_dataset("alexandreteles/strongreject")
```

**Use for:** Testing robustness against adversarial attacks

---

### 4. Create Your Own Multi-Policy Dataset ✅ RECOMMENDED

**Why:** No single public dataset covers all 6 policies well

**Approach:** Mix multiple sources

```python
# 40% ToxicChat (hate, harassment)
# 40% Synthetic multi-policy (custom labeled)
# 20% Jailbreaks (adversarial)

test_data = create_test_dataset(total_samples=100)
```

**See:** `load_datasets.py` for implementation

---

## Recommended Sunday Experiment Strategy

### Option A: Quick Test (100 samples)
```bash
# Use load_datasets.py to create mixed dataset
python load_datasets.py

# This gives you:
# - 40 ToxicChat samples (real data)
# - 40 synthetic multi-policy samples (curated)
# - 20 jailbreak attempts (adversarial)
```

**Pros:**
- ✅ Fast to generate
- ✅ Covers all 6 policies
- ✅ Includes adversarial cases

**Cons:**
- ❌ Synthetic data (not real-world)
- ❌ Small sample size

---

### Option B: ToxicChat Only (1000+ samples)
```python
# Use your existing ToxicChat approach
from datasets import load_dataset

ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
test_data = ds['test'][:1000]

# Run gauntlet on all 1000
# Compare multi-policy accuracy to your 73.3% baseline
```

**Pros:**
- ✅ Real-world data
- ✅ Large sample size
- ✅ You already know it works

**Cons:**
- ❌ Only tests 2 of 6 policies (hate, harassment)
- ❌ Single-label (not multi-policy)

---

### Option C: Label Your Own (Best for Production)

**Create custom multi-policy dataset:**

1. **Collect diverse content**
   - Reddit comments (pushshift)
   - Twitter/X posts
   - Forum posts
   - Chat logs (anonymized)

2. **Label with OpenAI Moderation API**
```python
import openai

response = openai.moderations.create(input="content here")
labels = response.results[0].categories

# Maps to your 6 policies:
# - hate → hate_speech
# - violence → violence
# - self-harm → self_harm
# - sexual/minors → sexual_minors
# - harassment → harassment
# - (illegal not in OpenAI categories, add manually)
```

3. **Manual review** - Validate API labels for accuracy

**Pros:**
- ✅ High quality
- ✅ Real multi-policy labels
- ✅ Production-ready

**Cons:**
- ❌ Time-consuming (weeks)
- ❌ Costs money (OpenAI API)

---

## For Sunday (Recommended)

**Use Option A + Option B combo:**

1. **Quick validation:** Run 100-sample synthetic dataset
   - Tests all 6 policies
   - Verifies gauntlet works
   - ~20 minutes

2. **Real-world test:** Run 1000 ToxicChat samples
   - Compare to your 73.3% baseline
   - Validate findings on real data
   - ~2 hours (1000 samples × 12s each = 3.3 hours)

**Commands:**
```bash
# 1. Generate synthetic test set
python load_datasets.py

# 2. Run gauntlet on synthetic
python serial_gauntlet_simple.py

# 3. Create ToxicChat loader
python create_toxicchat_eval.py  # TODO: Create this

# 4. Run gauntlet on ToxicChat
python eval_toxicchat.py  # TODO: Create this
```

---

## Dataset Sources Summary

| Dataset | Coverage | Size | Access | Multi-Policy |
|---------|----------|------|--------|--------------|
| **ToxicChat** | Hate, Harassment | 10K | ✅ Free | ❌ Single |
| **OpenAI Benchmark** | All 6 | Unknown | ❌ Internal | ✅ Yes |
| **StrongREJECT** | All 6 (adversarial) | 313 | ⚠️ Check | ✅ Yes |
| **Synthetic (ours)** | All 6 | Custom | ✅ Free | ✅ Yes |
| **OpenAI API labeled** | All 6 (5+manual) | Custom | 💰 Paid | ✅ Yes |

---

## HuggingFace Commands

```bash
# Install
pip install datasets

# Cache location
~/.cache/huggingface/datasets/

# List cached
ls -lh ~/.cache/huggingface/datasets/

# Clear cache (if needed)
rm -rf ~/.cache/huggingface/datasets/
```

---

## Next Steps

1. **Test load_datasets.py locally**
   ```bash
   python load_datasets.py
   cat test_dataset.json | head -50
   ```

2. **Copy to remote-server for Sunday**
   ```bash
   scp load_datasets.py test_dataset.json user@remote-server:~/wizard101/experiments/
   ```

3. **Create evaluation script** (integrates dataset → gauntlet)
   - Loads test_dataset.json
   - Runs serial_gauntlet for each sample
   - Calculates metrics (accuracy, F1, per-policy performance)

4. **Run experiments Sunday** 🎯
