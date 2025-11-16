# Experiment Tracking Log

## Experiment Naming Convention
All experiment scripts follow: `experiment_NN_description.py`

## Completed Experiments

### Experiment 01: Baseline Balanced Benchmark
**Script**: `experiment_01_baseline_balanced.py` (formerly `eval_benchmark.py`)
**Dataset**: `balanced_benchmark.json` (90 samples, synthetic)
**Model**: `gpt-oss:20b`
**Results**:
- Multi-policy accuracy: 61.1%
- Overall accuracy: 98.9%
**Date**: 2025-11-16
**Notes**: First successful run after fixing parsing bug

### Experiment 02: Medium Policies Test
**Script**: `experiment_02_medium_policies.py` (formerly `eval_medium.py`)
**Dataset**: `balanced_benchmark.json` (90 samples)
**Model**: `gpt-oss:20b`
**Results**:
- Multi-policy accuracy: 37.8%
- Overall accuracy: ~98%
**Date**: 2025-11-16
**Notes**: Medium policies (300-500 tokens) performed WORSE than minimal (100-150 tokens)

### Experiment 03: Performance Baseline (Balanced)
**Script**: `experiment_03_perf_balanced.py` (formerly `eval_performance.py`)
**Dataset**: `balanced_benchmark.json` (90 samples, synthetic)
**Model**: `gpt-oss:20b`
**Status**: Running on remote server
**Purpose**: Establish throughput baseline (samples/hour)

### Experiment 04: Performance Test (Larger)
**Script**: `experiment_04_perf_larger.py` (copy of exp 03)
**Dataset**: `larger_benchmark.json` (270 samples, synthetic)
**Model**: `gpt-oss:20b`
**Status**: Running on remote server
**Purpose**: Test throughput on 3x larger dataset

### Experiment 05: Performance Test (WildGuard)
**Script**: `experiment_05_perf_wildguard.py` (copy of exp 03)
**Dataset**: `wildguard_benchmark.json` (300 samples, REAL)
**Model**: `gpt-oss:20b`
**Status**: Running on remote server
**Purpose**: Test throughput on real-world adversarial prompts

### Experiment 06: Safeguard Model Test
**Script**: `experiment_06_safeguard_only.py` (formerly `eval_safeguard_only.py`)
**Dataset**: `wildguard_benchmark.json` (300 samples, REAL)
**Model**: `gpt-oss-safeguard:latest`
**Status**: Ready to run
**Purpose**: Compare safeguard-tuned model to baseline (61.1% multi-policy)

## Future Experiments
- Experiment 07: Model comparison (baseline vs safeguard on all benchmarks)
- Experiment 08: Policy length optimization
- Experiment 09: Parallel gauntlet architecture

## Dataset Catalog

### balanced_benchmark.json
- Size: 90 samples
- Type: Synthetic
- Distribution: 15 per policy (8 violations, 7 safe each)
- Purpose: Initial baseline testing

### larger_benchmark.json
- Size: 270 samples
- Type: Synthetic (3x balanced)
- Purpose: Performance throughput testing

### wildguard_benchmark.json
- Size: 300 samples
- Type: REAL (from allenai/wildguardmix)
- Distribution: illegal(131), hate_speech(60), safe(53), violence(20), self_harm(19), harassment(17)
- Purpose: Real-world adversarial testing

## Model Catalog

### gpt-oss:20b (Baseline)
- Type: General-purpose reasoning model
- Performance: 61.1% multi-policy, 98.9% overall

### gpt-oss-safeguard:latest (Safeguard)
- Type: Safety-tuned version of 20b base
- Performance: TBD (Experiment 06)

## Key Findings
1. **Minimal policies optimal**: 100-150 token policies outperform 300-500 token policies
2. **Sample size matters**: 90 samples insufficient for statistical confidence (documented limitation)
3. **WildGuard access**: Real-world dataset available via HuggingFace CLI
