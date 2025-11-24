# Scripts

Utility scripts for training, evaluation, and model management.

## Directory Structure

```
scripts/
├── train/          # Model training scripts
├── eval/           # Evaluation and benchmarking
└── utils/          # Utility scripts
```

## Training Scripts (`train/`)

| Script | Purpose |
|--------|---------|
| `train_l0_bouncer.py` | Full L0 bouncer training (original) |
| `train_l0_mega.py` | Improved L0 training (recommended) |
| `train_l0_full.py` | Full dataset training variant |
| `train_l0_12k.py` | 12K sample training variant |
| `build_mega_dataset.py` | Build optimized training dataset |
| `build_12k_dataset.py` | Build 12K sample dataset |
| `build_test_set.py` | Build test/evaluation set |

**Recommended:** Use `train_l0_mega.py` with `build_mega_dataset.py` for best results.

## Evaluation Scripts (`eval/`)

| Script | Purpose |
|--------|---------|
| `evaluate_cascade.py` | Main cascade evaluation (recommended) |
| `evaluate_cascade_batch.py` | Batch evaluation for large datasets |
| `evaluate_heretic.py` | Heretic dataset evaluation |
| `evaluate_heretic_full.py` | Full Heretic evaluation suite |
| `eval_l1_baseline.py` | L1 (GuardReasoner) baseline testing |
| `tune_l0_threshold.py` | Threshold tuning for L0 bouncer |

**Recommended:** Use `evaluate_cascade.py` for standard benchmarking.

## Utility Scripts (`utils/`)

| Script | Purpose |
|--------|---------|
| `quantize_guardreasoner.py` | Quantize GuardReasoner model to 4-bit |
| `download_guardreasoner_test.py` | Download GuardReasoner test data |
| `compare_methods.py` | Compare different safety methods |

## Usage Examples

```bash
# Train L0 bouncer
python scripts/train/train_l0_mega.py

# Evaluate cascade
python scripts/eval/evaluate_cascade.py

# Quantize model
python scripts/utils/quantize_guardreasoner.py
```
