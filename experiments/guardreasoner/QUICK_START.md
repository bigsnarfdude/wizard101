# GuardReasoner Experiments - Quick Start

## 📊 Current Status

**Baseline (Exp 12):** 57.5% accuracy
**Target:** 75-80% accuracy (GuardReasoner level)
**Approach:** Reasoning-based safety classification

---

## 🎯 What We're Testing

### Problem with Current Approach
```
User: "How to hack a system?"
Current model: flags ALL policies → False positive
GuardReasoner: Step 1: User wants hacking info
               Step 2: Could enable illegal activity
               Step 3: Violates "illegal" policy only
               Decision: UNSAFE - illegal
```

### The GuardReasoner Method
1. **Intent Analysis** - What is the user asking?
2. **Harm Assessment** - What could go wrong?
3. **Policy Reasoning** - Which specific policies apply?
4. **Final Decision** - Classification with explanation

---

## 📁 Files Created

### Planning Documents
- `PLAN.md` - Complete 45-experiment roadmap
- `EXPERIMENT_TRACKER.md` - Live status tracking
- `README.md` - Comprehensive overview
- `QUICK_START.md` - This file

### Folders
- `scripts/` - Experiment Python scripts
- `prompts/` - Reasoning prompt templates
- `data/` - Generated datasets
- `results/` - Experiment outputs
- `models/` - Fine-tuned models

---

## 🚀 Phase 1 Experiments (Starting Next)

**Timeline:** 1-2 days
**Cost:** FREE (uses existing Ollama setup)
**GPU needed:** NO (inference only)

| Exp | Name | Goal | Expected Gain |
|-----|------|------|---------------|
| 20 | Single-step reasoning | Add "explain your reasoning" | +2-5% |
| 21 | Multi-step reasoning | GuardReasoner 4-step format | +5-8% |
| 22 | CoT with examples | Few-shot prompting | +7-10% |
| 23 | Safeguard + reasoning | Safety tuning + reasoning | +5-8% |
| 24 | Reasoning + minimal | Test if reasoning beats verbose | +5-8% |
| 25 | Self-consistency | Sample 3x, majority vote | +8-11% |

**Target:** 67% accuracy (best Phase 1 result)

---

## 📈 Full Roadmap

### Phase 1: Prompt Engineering (Week 1)
- **Zero training needed**
- Test reasoning prompts
- **Goal:** 57% → 67%

### Phase 2: Ensemble Mining (Week 2)
- **Zero training needed**
- Multi-model voting
- Find hard samples
- **Goal:** 67% → 68%

### Phase 3: Data Generation (Week 3)
- **GPT-4 API needed** (~$100)
- Generate reasoning traces
- **Output:** Training dataset

### Phase 4: Fine-Tuning (Week 4)
- **GPU needed** (A100/H100)
- Train with reasoning
- **Goal:** 68% → 77%

### Phase 5: Analysis (Ongoing)
- Failure analysis
- Optimization
- **Goal:** Production-ready

---

## 🎬 Next Actions

1. **Wait** - Exp 16-17 to complete (~6 hours remaining)
2. **Implement** - Experiment 20 script
3. **Create** - Reasoning prompt templates
4. **Run** - Phase 1 experiments on nigel
5. **Analyze** - Results and iterate

---

## 💡 Key Insights from Planning

### Why Current Approach Plateaued at 57%
1. **Over-flagging** - Models see keywords, flag everything
2. **No reasoning** - Direct classification without logic
3. **Policy verbosity irrelevant** - More text ≠ better understanding

### Why GuardReasoner Works
1. **Structured reasoning** - Step-by-step logic before decision
2. **Context awareness** - Understands intent vs harmful outcomes
3. **Explainability** - Can trace why decision was made

### Expected Results
- **Phase 1 (prompting):** 57% → 67% (+10%)
- **Phase 2 (ensemble):** 67% → 68% (+1%)
- **Phase 4 (fine-tuning):** 68% → 77% (+9%)
- **Total improvement:** +20% over baseline

---

## ❓ Open Questions

1. **GPU for Phase 4?**
   - Do you have cluster access?
   - Budget for cloud GPU? (~$30/experiment)

2. **GPT-4 API for Phase 3?**
   - Budget: ~$100 for reasoning generation
   - Alternative: Use local models (lower quality)

3. **Timeline priority?**
   - Fast track to Phase 4? (skip some Phase 1 experiments)
   - Complete methodology? (all 45 experiments)

---

## 📞 Status Check

**Experiments 16-17:** Running on nigel (ETA: 6 hours)
**GuardReasoner folder:** Created and documented
**Next experiment:** Exp 20 (ready to implement)
**Blocking:** None (can start coding prompts now)

---

**Ready to proceed with Phase 1 implementation?**

See `PLAN.md` for full details or `EXPERIMENT_TRACKER.md` for live status.
