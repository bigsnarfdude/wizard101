# Archived Datasets

Historical datasets from R&D experiments. Not used by production cascade.

## GuardReasoner Training Data (~710 MB)

Datasets with reasoning traces (suffix "R") for LoRA fine-tuning experiments:

| File | Size | Samples | Description |
|------|------|---------|-------------|
| `all_combined.json` | 357M | ~127K | Combined training data |
| `WildGuardTrainR.json` | 263M | ~92K | WildGuard with reasoning |
| `BeaverTailsTrainR.json` | 58M | ~30K | BeaverTails with reasoning |
| `AegisTrainR.json` | 26M | ~5K | Aegis with reasoning |
| `ToxicChatTrainR.json` | 5.5M | ~10K | ToxicChat with reasoning |
| `sample.json` | 7.5K | 10 | Sample for testing |

## DLP Analysis

| File | Size | Description |
|------|------|-------------|
| `failure_analysis_full.json` | 17M | DLP false negative analysis |

## Origin

Moved from `archive/2024-11-experiments/guardreasoner-training/` on 2025-11-24.

These datasets were used for:
- GuardReasoner 3B/8B LoRA fine-tuning (exp_18, exp_19)
- R-SFT (Reasoning Supervised Fine-Tuning) experiments
- DPO (Direct Preference Optimization) training
