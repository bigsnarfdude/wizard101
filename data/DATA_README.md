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
| Benchmark | 15 | ~230,000 | ~97 MB |
| Raw | 2 | ~800 | ~58 KB |
| Archived | 7 | ~127,544 | ~710 MB |

---

## Benchmark Datasets (data/benchmark/)

### Primary Evaluation Suite (12 datasets, ~131K samples)

Used in `scripts/eval/evaluate_all_benchmarks.py`:

| Dataset | File | Samples | Purpose |
|---------|------|---------|---------|
| SimpleSafetyTests | `simplesafetytests.json` | 100 | Basic safety sanity checks |
| JailbreakBench | `jailbreakbench.json` | 200 | Jailbreak detection |
| StrongREJECT | `strongreject.json` | 313 | Adversarial refusal testing |
| HarmBench | `harmbench_test.json` | 500 | Harmful behavior detection |
| SGBench | `sgbench.json` | 1,442 | Safety benchmark |
| OpenAI Moderation | `openai_moderation.json` | 1,680 | Baseline comparison |
| BeaverTails | `beavertails_30k.json` | 3,021 | Jailbreak evaluation |
| ToxicChat | `toxicchat_test.json` | 5,083 | Toxicity detection |
| SALAD-Bench Attack | `salad_bench_attack.json` | 5,000 | Adversarial attack prompts |
| SALAD-Bench Base | `salad_bench_base.json` | 21,318 | Hierarchical harm taxonomy |
| OR-Bench | `or_bench.json` | 82,333 | Over-refusal benchmark (99% safe) |
| Combined | `combined_benchmark.json` | 10,384 | Multi-source combination |

### Large-Scale (run separately)

| Dataset | File | Samples | Purpose |
|---------|------|---------|---------|
| WildJailbreak | `wildjailbreak.json` | 88,444 | Large-scale jailbreak testing |

---

## Benchmark Results (Nov 26, 2025)

Full cascade evaluation on 12 datasets:

| Dataset | Samples | Accuracy | Precision | Recall | F1 | Latency |
|---------|---------|----------|-----------|--------|-----|---------|
| SimpleSafetyTests | 100 | 95.0% | 100.0% | 95.0% | 97.4% | 1284ms |
| JailbreakBench | 200 | 65.5% | 59.3% | 99.0% | 74.2% | 1640ms |
| StrongREJECT | 313 | 95.5% | 100.0% | 95.5% | 97.7% | 931ms |
| HarmBench | 500 | 99.6% | 100.0% | 99.6% | 99.8% | 5ms |
| SGBench | 1,442 | 89.6% | 100.0% | 89.6% | 94.5% | 1310ms |
| OpenAI Moderation | 1,680 | 74.1% | 55.0% | 90.8% | 68.5% | 6838ms |
| BeaverTails | 3,021 | 70.1% | 69.5% | 85.3% | 76.6% | 2114ms |
| ToxicChat | 5,083 | *running* | - | - | - | - |
| SALAD-Bench Attack | 5,000 | *pending* | - | - | - | - |
| SALAD-Bench Base | 21,318 | *pending* | - | - | - | - |
| OR-Bench | 82,333 | *pending* | - | - | - | - |
| Combined | 10,384 | *pending* | - | - | - | - |

Results file: `experiments/benchmark_run_20251126_002112.log`

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

### `xstest.json` (450 samples, 66 KB)
XSTest benchmark for over-refusal testing (false positives).

### `combined_test.json` (200 samples, 28 KB)
Validation set matching combined_train.json format.

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

### Other archived files:
- `WildGuardTrainR.json` (86,759 samples) - WildGuard + reasoning traces
- `BeaverTailsTrainR.json` (27,186 samples) - BeaverTails + reasoning
- `AegisTrainR.json` (10,798 samples) - Aegis + reasoning
- `ToxicChatTrainR.json` (2,801 samples) - ToxicChat + reasoning
- `failure_analysis_full.json` (17 MB) - DLP false negative analysis

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
| SALAD-Bench | [OpenSafetyLab/Salad-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data) | Apache 2.0 |
| OR-Bench | [bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench) | MIT |

---

## Label Mappings

Different datasets use different label conventions:

| Dataset | Safe Label | Harmful Label |
|---------|------------|---------------|
| WildGuard | benign | harmful |
| train_12k | safe | harmful |
| GuardReasoner | unharmful | harmful |
| SimpleSafety | safe | unsafe |
| SALAD-Bench | 0 | 1 |
| OR-Bench | safe | unsafe |

Normalize labels before training/evaluation:
```python
def normalize_label(label):
    if label in [0, "0"]:
        return "safe"
    if label in [1, "1"]:
        return "harmful"
    if str(label).lower() in ["safe", "benign", "unharmful", "harmless"]:
        return "safe"
    return "harmful"
```

---

## Notes

1. **OR-Bench**: Over-Refusal benchmark - 99% safe prompts, 1% harmful. Tests for false positives.

2. **SALAD-Bench**: ACL 2024 benchmark with hierarchical harm taxonomy. Attack subset contains adversarial prompts.

3. **Archived Datasets**: R-SFT training datasets (~710MB) are in `data/archived/` - used for GuardReasoner LoRA fine-tuning experiments.
