# GuardReasoner Experiments - Complete Index

**Created:** 2025-11-17
**Status:** Ready for Phase 1-4 implementation
**Goal:** Improve safety classification from 57% → 80%+ using reasoning-based approaches

---

## 📚 Documentation Index

### Start Here
- **[READY_TO_TRAIN.md](READY_TO_TRAIN.md)** ⭐ START HERE
  - Quick summary of what's ready
  - What you can do now
  - What you need next
  - Cost/time breakdown

### Planning & Methodology
- **[PLAN.md](PLAN.md)** - Complete 45-experiment methodology
  - 5 phases from prompt engineering to production
  - Timeline and resource requirements
  - Expected outcomes per phase

- **[QUICK_START.md](QUICK_START.md)** - TL;DR version of the plan
  - High-level overview
  - Key milestones
  - Quick reference

- **[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md)** - Live status tracking
  - Which experiments are done/running/pending
  - Current best results
  - Next experiment to run

### Training (R-SFT)
- **[RSFT_EXPLAINED.md](RSFT_EXPLAINED.md)** ⭐ READ BEFORE TRAINING
  - What is R-SFT and why it works
  - Technical details from GuardReasoner paper
  - Expected results and FAQ

- **[R-SFT_Training_Colab.md](R-SFT_Training_Colab.md)** - Colab notebook guide
  - Step-by-step instructions
  - Cell-by-cell code
  - Configuration options
  - Troubleshooting

- **[scripts/rsft_training_colab.py](scripts/rsft_training_colab.py)** - Training script
  - Complete implementation
  - Ready for T4 Colab
  - 500+ lines with documentation

### Datasets
- **[DATASET_SUMMARY.md](DATASET_SUMMARY.md)** - What data we have
  - WildGuard full benchmark (1,554 samples)
  - Policy definitions
  - Sample distribution

- **[GUARDREASONER_DATASET.md](GUARDREASONER_DATASET.md)** - Training data
  - GuardReasoner's 127K sample dataset
  - How to download and use
  - Format specifications

### General
- **[README.md](README.md)** - Project overview
  - What this folder contains
  - How everything fits together
  - Links to key resources

---

## 🗂️ Folder Structure

```
guardreasoner/
├── INDEX.md                          ← YOU ARE HERE
├── READY_TO_TRAIN.md                 ← Start here
├── RSFT_EXPLAINED.md                 ← Read before training
├── R-SFT_Training_Colab.md           ← Colab guide
├── PLAN.md                           ← Full methodology
├── EXPERIMENT_TRACKER.md             ← Status tracking
├── QUICK_START.md                    ← TL;DR
├── README.md                         ← Overview
├── DATASET_SUMMARY.md                ← Data we have
├── GUARDREASONER_DATASET.md          ← Training datasets
│
├── scripts/
│   └── rsft_training_colab.py        ← Training script (T4)
│
├── prompts/                          ← Reasoning prompt templates
├── data/                             ← Training datasets
├── results/                          ← Experiment outputs
└── models/                           ← Fine-tuned models
```

---

## 🎯 Quick Navigation

### I want to...

**Understand what we're building**
→ Read [READY_TO_TRAIN.md](READY_TO_TRAIN.md) (5 min)

**Learn about R-SFT methodology**
→ Read [RSFT_EXPLAINED.md](RSFT_EXPLAINED.md) (15 min)

**Train a model on Colab**
→ Follow [R-SFT_Training_Colab.md](R-SFT_Training_Colab.md) (4-6 hours)

**See the full experiment plan**
→ Read [PLAN.md](PLAN.md) (20 min)

**Check experiment status**
→ View [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) (2 min)

**Understand our datasets**
→ Read [DATASET_SUMMARY.md](DATASET_SUMMARY.md) (5 min)

**Find GuardReasoner training data**
→ Read [GUARDREASONER_DATASET.md](GUARDREASONER_DATASET.md) (10 min)

---

## 🚀 Implementation Status

### ✅ Completed
- [x] Full 45-experiment plan designed
- [x] R-SFT training script for Colab
- [x] Complete documentation (9 files)
- [x] Dataset analysis and inventory
- [x] Cost/time estimates

### 🟡 In Progress
- [ ] Experiments 16-17 running on nigel (ETA: ~6 hours)

### ⏳ Ready to Start
- [ ] **Phase 1:** Prompt engineering experiments (20-25)
  - No cost, uses existing setup
  - Expected: +10% accuracy

- [ ] **Phase 3:** Generate reasoning traces
  - Cost: $50-100 for GPT-4
  - Time: 2-3 hours

- [ ] **Phase 4:** R-SFT training
  - Script ready: `rsft_training_colab.py`
  - Needs: Reasoning traces from Phase 3
  - Expected: +18-20% accuracy

---

## 📊 Expected Journey

```
Current State:
├─ Experiments 01-15: Complete (baseline = 57%)
├─ Experiments 16-17: Running
└─ Phase 1 ready to start

Phase 1 (Prompt Engineering):
├─ Experiments 20-25: Add reasoning prompts
├─ Expected: 57% → 67% (+10%)
├─ Cost: $0
└─ Time: 1-2 days

Phase 3 (Data Generation):
├─ Generate 9,324 reasoning traces
├─ Cost: $50-100 (GPT-4)
└─ Time: 2-3 hours

Phase 4 (R-SFT Training):
├─ Experiments 36-40: Fine-tune with reasoning
├─ Expected: 67% → 75-77% (+8-10% more)
├─ Cost: $0 (Colab free tier)
└─ Time: 4-6 hours

Phase 5 (Hard Sample DPO):
├─ Experiments 41-45: Refine on edge cases
├─ Expected: 75-77% → 78-80% (+2-3% more)
└─ Time: 1 week

Final Result:
└─ 80%+ accuracy (vs 57% baseline)
   ✓ +23% absolute improvement
   ✓ Explainable reasoning
   ✓ Production-ready
```

---

## 🔬 Technical Stack

### Current Infrastructure
- **Server:** nigel.birs.ca
- **Model:** gpt-oss:20b (Ollama)
- **Dataset:** WildGuard (1,554 samples)
- **Policies:** 6 policies (hate_speech, violence, etc.)

### Training Infrastructure (Phase 4)
- **GPU:** Google Colab T4 (16GB VRAM, free tier)
- **Model:** LLaMA 3.2-3B or 1B
- **Method:** R-SFT with LoRA
- **Framework:** Unsloth (2x faster training)

### Data Generation (Phase 3)
- **API:** GPT-4 or Claude
- **Cost:** ~$50-100 for 9,324 traces
- **Format:** Step-by-step reasoning → classification

---

## 💡 Key Insights

### What We Learned (Experiments 01-17)
1. **Policy length doesn't matter** (minimal/medium/verbose all ~57%)
2. **Model choice barely matters** (baseline vs safeguard ~57%)
3. **Current approach has plateaued** at 57%

### Why R-SFT Will Work
1. **Reasoning traces force careful analysis** (not just keyword matching)
2. **Proven by GuardReasoner paper** (84% accuracy, +27% over baseline)
3. **Addresses our specific failure modes** (over-flagging, context misunderstanding)

### Expected Gains
- **Prompt engineering:** +10% (free)
- **R-SFT training:** +8-10% (requires $50-100 for data)
- **Hard sample DPO:** +2-3% (requires fine-tuned model)
- **Total:** +20-23% improvement

---

## 📈 Success Metrics

### Experiment 36 (R-SFT) Targets:
- [ ] Overall accuracy: 75-77% (vs 57% baseline)
- [ ] Violence F1: 50%+ (vs 22% baseline)
- [ ] Illegal F1: 75%+ (vs 59% baseline)
- [ ] False positive rate: <5% (vs 7.5% baseline)
- [ ] Reasoning coherence: >90% on manual review

### Phase 5 (Final) Targets:
- [ ] Overall accuracy: 78-80%
- [ ] Approaching GuardReasoner's 84%
- [ ] Production-ready deployment

---

## 🔗 External Resources

### Papers
- [GuardReasoner (arXiv:2501.18492)](https://arxiv.org/abs/2501.18492)
- [WildGuard dataset](https://huggingface.co/datasets/allenai/wildguardmix)

### Code
- [GuardReasoner GitHub](https://github.com/yueliu1999/GuardReasoner)
- [Unsloth framework](https://github.com/unslothai/unsloth)

### Datasets
- [GuardReasonerTrain (HuggingFace)](https://huggingface.co/datasets/yueliu1999/GuardReasonerTrain)

---

## 🤝 How to Contribute

### Add a New Experiment
1. Add to `PLAN.md` under appropriate phase
2. Update `EXPERIMENT_TRACKER.md` with status
3. Create script in `scripts/`
4. Document expected outcomes

### Report Results
1. Update `EXPERIMENT_TRACKER.md` with metrics
2. Add analysis to experiment's section in `PLAN.md`
3. Update this index if major findings

### Improve Documentation
1. Clarify confusing sections
2. Add examples or diagrams
3. Update quick reference guides

---

## 📝 Change Log

### 2025-11-17 (Initial Creation)
- Created complete experiment plan (45 experiments)
- Implemented R-SFT training script for Colab
- Documented all phases and methodologies
- Analyzed datasets and requirements
- Established cost/time estimates

### Next Updates
- Will track as experiments 16-17 complete
- Will update after Phase 1 experiments
- Will update after R-SFT training results

---

## ✨ Summary

You have everything needed to:
1. ✅ Understand the GuardReasoner approach
2. ✅ Train an R-SFT model on T4 Colab
3. ✅ Improve accuracy from 57% to 75-80%
4. ✅ Deploy a production-ready safety classifier

What's needed:
- ⏳ Reasoning traces (cost: $50-100, time: 2-3 hours)
- ⏳ Colab training session (free, time: 4-6 hours)

**Start here:** [READY_TO_TRAIN.md](READY_TO_TRAIN.md)

**Questions?** Check the relevant doc above or read [RSFT_EXPLAINED.md](RSFT_EXPLAINED.md)
