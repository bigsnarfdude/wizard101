# Dataset Research & Future Additions

This document tracks datasets we've researched for safety classification.

---

## Currently Using

### Safety Classification

| Dataset | Samples | Location | Purpose |
|---------|---------|----------|---------|
| **train_12k** | 12,000 | `data/training/train_12k.json` | Primary training |
| **WildGuard** | 1,554 | `data/evaluation/wildguard_full_benchmark.json` | Primary evaluation |
| **ToxicChat** | 5,083 | `data/benchmark/toxicchat_test.json` | Toxicity evaluation |
| **BeaverTails** | 3,021 | `data/benchmark/beavertails_30k.json` | Jailbreak evaluation |
| **WildJailbreak** | 88,444 | `data/benchmark/wildjailbreak.json` | Large-scale jailbreak |
| **SimpleSafetyTests** | 100 | `data/benchmark/simplesafetytests.json` | Sanity checks |
| **HarmBench** | 500 | `data/benchmark/harmbench_test.json` | Harm evaluation |
| **OpenAI Moderation** | 1,680 | `data/benchmark/openai_moderation.json` | Baseline comparison |
| **XSTest** | 450 | `data/evaluation/xstest.json` | Over-refusal testing |
| **StrongREJECT** | 313 | `data/benchmark/strongreject.json` | Adversarial testing |
| **JailbreakBench** | 200 | `data/benchmark/jailbreakbench.json` | Jailbreak evaluation |
| **SGBench** | 1,442 | `data/benchmark/sgbench.json` | Safety benchmark |

### Prompt Injection Detection (cascade_quarantine)

| Dataset | Samples | Source | Purpose | Status |
|---------|---------|--------|---------|--------|
| **xTRam1/safe-guard-prompt-injection** | 10,296 | HuggingFace | Primary injection classifier | ✅ Using |
| reshabhs/SPML_Chatbot_Prompt_Injection | 16,012 | HuggingFace | Degree annotations | Available |
| deepset/prompt-injections | 662 | HuggingFace | Legacy (multilingual) | Deprecated |

**xTRam1 Dataset Details:**
- Train: 8,236 samples | Test: 2,060 samples
- Benign: 7,150 (69.4%) | Injection: 3,146 (30.6%)
- English-only, purpose-built for injection detection
- Achieved: 99.2% accuracy, 99.7% precision, 97.8% recall

---

## Researched - Not Yet Downloaded

### SALAD-Bench
- **Paper**: Li et al. (2024) "SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark"
- **HuggingFace**: `OpenSafetyLab/Salad-Data`
- **Size**: 30,000+ samples
- **Why useful**: Comprehensive, hierarchical harm taxonomy

### Do-Not-Answer
- **Paper**: Wang et al. (2023) "Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs"
- **HuggingFace**: `LibrAI/do-not-answer`
- **Size**: 939 prompts
- **Why useful**: Tests refusal quality, not just detection

### CValues
- **Paper**: Xu et al. (2023) "CValues: Measuring the Values of Chinese Large Language Models"
- **HuggingFace**: `daven3/cvalues_responsibility_mc`
- **Size**: 2,100 samples
- **Why useful**: Cross-cultural safety norms

### AdvBench
- **Paper**: Zou et al. (2023) "Universal and Transferable Adversarial Attacks on Aligned Language Models"
- **Size**: 520 harmful behaviors
- **Why useful**: Adversarial attack testing

---

## Dataset Sources by Category

### Hate Speech / Harassment
- ToxicChat (using)
- HateXplain
- Implicit Hate Corpus

### Violence
- SimpleSafetyTests (using)
- MGSM (Multi-lingual)

### Self-Harm
- SimpleSafetyTests (using)
- Crisis Text Line (restricted)

### Sexual Content / Minors
- SimpleSafetyTests (using)
- NSFW datasets (restricted access)

### Illegal Activity
- BeaverTails (using)
- CyberSecEval

### Jailbreaks / Adversarial
- WildGuard (using)
- WildJailbreak (using)
- StrongREJECT (using)
- JailbreakBench (using)
- AdvBench

---

## Priority Additions

### High Priority
1. **SALAD-Bench** - Comprehensive benchmark with hierarchy

### Medium Priority
2. **Do-Not-Answer** - Refusal quality measurement
3. **AdvBench** - Additional adversarial data

### Low Priority
4. **CValues** - Cross-cultural (if internationalizing)

---

## Download Commands

### SALAD-Bench
```python
from datasets import load_dataset

ds = load_dataset("OpenSafetyLab/Salad-Data")
```

### Do-Not-Answer
```python
from datasets import load_dataset

ds = load_dataset("LibrAI/do-not-answer")
```

---

## References

### Papers
- ToxicChat: Lin et al. (2023) - EMNLP Findings
- XSTest: Röttger et al. (2024) - NAACL
- SimpleSafetyTests: Vidgen et al. (2023)
- HarmBench: Mazeika et al. (2024)
- WildGuard: Han et al. (2024) - Allen AI
- BeaverTails: Ji et al. (2023) - NeurIPS
- WildJailbreak: Jiang et al. (2024) - Allen AI
- StrongREJECT: Souly et al. (2024)
- JailbreakBench: Chao et al. (2024)

---

## Action Items

- [x] Download XSTest for over-refusal testing
- [x] Download StrongREJECT
- [x] Download JailbreakBench
- [x] Download WildJailbreak
- [x] Download xTRam1/safe-guard-prompt-injection for injection detection
- [x] Train injection classifier (99%+ accuracy achieved)
- [ ] Consider SALAD-Bench for comprehensive evaluation
- [ ] Evaluate current models on all benchmarks
