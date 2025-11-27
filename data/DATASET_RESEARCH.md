# Dataset Research & Benchmark Status

This document tracks datasets and benchmark evaluations for wizard101 safety classification.

---

## Current Benchmark Evaluation (Nov 26, 2025)

Running full cascade evaluation on 12 datasets (~131K samples).

**Status**: In progress  
**Screen**: `195644.cascade-bench`  
**Log**: `experiments/benchmark_run_20251126_002112.log`

### Results So Far

| Dataset | Samples | Accuracy | Precision | Recall | F1 | Status |
|---------|---------|----------|-----------|--------|-----|--------|
| SimpleSafetyTests | 100 | 95.0% | 100.0% | 95.0% | 97.4% | ✅ |
| JailbreakBench | 200 | 65.5% | 59.3% | 99.0% | 74.2% | ✅ |
| StrongREJECT | 313 | 95.5% | 100.0% | 95.5% | 97.7% | ✅ |
| HarmBench | 500 | 99.6% | 100.0% | 99.6% | 99.8% | ✅ |
| SGBench | 1,442 | 89.6% | 100.0% | 89.6% | 94.5% | ✅ |
| OpenAI Moderation | 1,680 | 74.1% | 55.0% | 90.8% | 68.5% | ✅ |
| BeaverTails | 3,021 | 70.1% | 69.5% | 85.3% | 76.6% | ✅ |
| ToxicChat | 5,083 | - | - | - | - | 🔄 |
| SALAD-Bench Attack | 5,000 | - | - | - | - | ⏳ |
| SALAD-Bench Base | 21,318 | - | - | - | - | ⏳ |
| OR-Bench | 82,333 | - | - | - | - | ⏳ |
| Combined | 10,384 | - | - | - | - | ⏳ |

### Key Observations

1. **HarmBench** (99.6%): Best performance - well-structured harmful behavior prompts
2. **JailbreakBench** (65.5%): Lower accuracy but 99% recall - catches attacks, over-flags benign
3. **OpenAI Moderation** (74.1%): High recall (91%), low precision (55%) - borderline content
4. **BeaverTails** (70.1%): Similar pattern - complex/ambiguous prompts

---

## Datasets In Use

### Benchmark Suite (data/benchmark/)

| Dataset | Samples | File | Source |
|---------|---------|------|--------|
| SimpleSafetyTests | 100 | `simplesafetytests.json` | Vidgen et al. 2023 |
| JailbreakBench | 200 | `jailbreakbench.json` | Chao et al. 2024 |
| StrongREJECT | 313 | `strongreject.json` | Souly et al. 2024 |
| HarmBench | 500 | `harmbench_test.json` | Mazeika et al. 2024 |
| SGBench | 1,442 | `sgbench.json` | Safety benchmark |
| OpenAI Moderation | 1,680 | `openai_moderation.json` | OpenAI API comparison |
| BeaverTails | 3,021 | `beavertails_30k.json` | Ji et al. 2023 |
| ToxicChat | 5,083 | `toxicchat_test.json` | Lin et al. 2023 |
| SALAD-Bench Attack | 5,000 | `salad_bench_attack.json` | Li et al. 2024 |
| SALAD-Bench Base | 21,318 | `salad_bench_base.json` | Li et al. 2024 |
| OR-Bench | 82,333 | `or_bench.json` | Over-refusal benchmark |
| Combined | 10,384 | `combined_benchmark.json` | Multi-source |
| WildJailbreak | 88,444 | `wildjailbreak.json` | Allen AI (run separately) |

### Prompt Injection (cascade_quarantine)

| Dataset | Samples | Status | Accuracy |
|---------|---------|--------|----------|
| xTRam1/safe-guard-prompt-injection | 8,236 | ✅ Complete | 97.78% |

---

## Data Sources

| Source | HuggingFace | License |
|--------|-------------|---------|
| WildGuard | allenai/wildguard | Apache 2.0 |
| BeaverTails | PKU-Alignment/BeaverTails | CC BY-NC 4.0 |
| ToxicChat | lmsys/toxic-chat | CC BY-NC 4.0 |
| HarmBench | cais/HarmBench | MIT |
| WildJailbreak | allenai/wildjailbreak | Apache 2.0 |
| StrongREJECT | dsbowen/strongreject | MIT |
| JailbreakBench | JailbreakBench | MIT |
| XSTest | Paul/XSTest | MIT |
| SALAD-Bench | OpenSafetyLab/Salad-Data | Apache 2.0 |
| OR-Bench | bench-llm/or-bench | MIT |

---

## Future Additions (Lower Priority)

### Do-Not-Answer
- **Size**: 939 prompts
- **Purpose**: Tests refusal quality, not just detection
- **Status**: Not downloaded

### AdvBench
- **Size**: 520 harmful behaviors
- **Purpose**: Adversarial attack testing
- **Status**: Not downloaded

### CValues
- **Size**: 2,100 samples
- **Purpose**: Cross-cultural safety norms
- **Status**: Not downloaded (only if internationalizing)

---

## Action Items

- [x] Download SimpleSafetyTests
- [x] Download JailbreakBench
- [x] Download StrongREJECT
- [x] Download HarmBench
- [x] Download SGBench
- [x] Download OpenAI Moderation
- [x] Download BeaverTails
- [x] Download ToxicChat
- [x] Download SALAD-Bench (base + attack)
- [x] Download OR-Bench
- [x] Download WildJailbreak
- [x] Create evaluate_all_benchmarks.py
- [x] Train prompt injection classifier (97.78% on 8K samples)
- [ ] Complete full benchmark evaluation (~131K samples)
- [ ] Generate final report with all results
- [ ] Analyze false positives/negatives by category

---

## References

### Papers
- **ToxicChat**: Lin et al. (2023) - EMNLP Findings
- **XSTest**: Röttger et al. (2024) - NAACL
- **SimpleSafetyTests**: Vidgen et al. (2023)
- **HarmBench**: Mazeika et al. (2024)
- **WildGuard**: Han et al. (2024) - Allen AI
- **BeaverTails**: Ji et al. (2023) - NeurIPS
- **WildJailbreak**: Jiang et al. (2024) - Allen AI
- **StrongREJECT**: Souly et al. (2024)
- **JailbreakBench**: Chao et al. (2024)
- **SALAD-Bench**: Li et al. (2024) - ACL
