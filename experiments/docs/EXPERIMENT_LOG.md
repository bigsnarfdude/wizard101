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
**Results**:
- Multi-policy accuracy: 56.7%
- Overall accuracy: 98.9%
- Throughput: 168 samples/hour
- Total time: 32.1 minutes
**Date**: 2025-11-16
**Purpose**: Establish throughput baseline (samples/hour)

### Experiment 04: Performance Test (Larger)
**Script**: `experiment_04_perf_larger.py` (copy of exp 03)
**Dataset**: `larger_benchmark.json` (270 samples, synthetic)
**Model**: `gpt-oss:20b`
**Results**:
- Multi-policy accuracy: 59.3%
- Overall accuracy: 98.5%
- Throughput: 191 samples/hour
- Total time: 84.9 minutes
**Date**: 2025-11-16
**Purpose**: Test throughput on 3x larger dataset

### Experiment 05: Performance Test (WildGuard)
**Script**: `experiment_05_perf_wildguard.py` (copy of exp 03)
**Dataset**: `wildguard_benchmark.json` (300 samples, REAL)
**Model**: `gpt-oss:20b`
**Results**:
- Multi-policy accuracy: **23.0%** ⚠️ MAJOR DROP on real data!
- Overall accuracy: 79.7%
- Throughput: 199 samples/hour
- Total time: 90.3 minutes
**Date**: 2025-11-16
**Notes**: **CRITICAL FINDING** - Synthetic data results (56-59%) DO NOT GENERALIZE to real adversarial data (23%)
**Purpose**: Test throughput on real-world adversarial prompts

### Experiment 06-11: Model × Policy Matrix (MAJOR FINDINGS)
**Date**: 2025-11-16
**Dataset**: `wildguard_benchmark.json` (300 samples, REAL)
**Purpose**: Systematic test of model type × policy length interaction

#### Experiment Matrix Results

```
                    Minimal         Medium          Verbose
                    (100-150 tok)   (300-500 tok)   (800-900 tok)
Baseline (20b)      23.0%          21.3%           26.7%
Safeguard           21.0%          30.3%           36.0% 🏆
```

#### Individual Experiment Details

**Experiment 05**: Baseline + Minimal (already completed above)
- Multi-policy: 23.0%
- Throughput: 199 samples/hour

**Experiment 06**: Safeguard + Minimal
- Script: `experiment_06_matrix.py`
- Multi-policy: 21.0%
- Throughput: 200 samples/hour
- Time: 89.8 minutes

**Experiment 08**: Baseline + Medium
- Script: `experiment_08_matrix.py`
- Multi-policy: 21.3%
- Throughput: 308 samples/hour
- Time: 58.5 minutes

**Experiment 09**: Baseline + Verbose
- Script: `experiment_09_matrix.py`
- Multi-policy: 26.7%
- Throughput: 317 samples/hour
- Time: 56.8 minutes

**Experiment 10**: Safeguard + Medium
- Script: `experiment_10_matrix.py`
- Multi-policy: 30.3%
- Throughput: 195 samples/hour
- Time: 92.5 minutes

**Experiment 11**: Safeguard + Verbose 🏆
- Script: `experiment_11_matrix.py`
- Multi-policy: **36.0%** (BEST RESULT)
- Throughput: 206 samples/hour
- Time: 87.3 minutes

### Experiment 18: GuardReasoner R-SFT Training (1 epoch) ⏳
**Script**: `guardreasoner/train_exp_18_rsft_unsloth.py`
**Date**: 2025-11-18
**Model**: Llama-3.2-3B-Instruct (4-bit LoRA)
**Dataset**: Combined guardreasoner_train_chatml.json (11,396 samples)
- Harmful Behaviors: Adversarial prompts
- Harmless Alpaca: Benign instructions

**Training Configuration**:
- Method: Reasoning-guided Supervised Fine-Tuning (R-SFT)
- Base: unsloth/Llama-3.2-3B-Instruct (4-bit)
- LoRA rank: 16, alpha: 16, dropout: 0
- Learning rate: 5e-5
- Batch size: 2 (effective: 128 with grad accum)
- Epochs: 1 (⚠️ paper recommends 5)
- Training time: 8.09 hours
- Final loss: 0.833

**HuggingFace Release**:
- Model: vincentoh/guardreasoner-llama3.2-3b-lora-1epoch
- Status: Preliminary 1-epoch checkpoint uploaded
- URL: https://huggingface.co/vincentoh/guardreasoner-llama3.2-3b-lora-1epoch

**Paper Validation Notes** (GuardReasoner arXiv:2505.20087):
- Paper used: 5 epochs, batch_size=32, lr=1e-6 on 8xA100
- Our setup: 1 epoch, effective_batch=128, lr=5e-5 on 1xGPU (4-bit)
- Paper finding: "reasoning-based models robust to overfitting even with 50 epochs"
- Paper finding: 500 samples × 50 epochs = within 3% of full dataset performance

**Next Steps (Option 2: Evaluate First)** 📋
1. **Run WildGuard test evaluation** on 1-epoch model
   - Test set: 1,554 samples
   - Baseline (Exp 12): 57.5% accuracy
2. **Decision point**:
   - ✅ If performance ≥ 60%: 1 epoch might be sufficient (sample efficiency hypothesis)
   - 🔄 If performance < 60%: Continue training for 4 more epochs (match paper)
3. **Research validation goal**: Verify if reasoning traces improve sample efficiency
   - Hypothesis: With reasoning, 1 epoch might match paper's 5-epoch performance
   - Alternative: Need full 5 epochs to replicate paper results

**Status**: Awaiting evaluation before continuing training

## Future Experiments
- Experiment 19: GuardReasoner 5-epoch training (if needed)
- Experiment 20: Parallel gauntlet architecture
- Experiment 21: Ensemble voting systems
- Experiment 22: Chain-of-thought policy reasoning

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

### 1. 🏆 VERBOSE POLICIES + SAFEGUARD = BEST PERFORMANCE (36.0%)
**MAJOR DISCOVERY**: The combination of safety-tuned model + detailed policies achieves **36.0% multi-policy accuracy**
- **+13 percentage points** improvement over baseline minimal (23.0% → 36.0%)
- **+57% relative improvement**
- Approaching OpenAI's benchmark: 36.0% vs 43.6% (7.6 point gap)

### 2. Safety Tuning REQUIRES Detailed Policies
**Safeguard model performance by policy length:**
- Minimal (100-150 tok): 21.0% ❌ (WORST)
- Medium (300-500 tok): 30.3% (+9.3 points)
- Verbose (800-900 tok): **36.0%** (+15 points) 🏆

**Initial conclusion was WRONG**: Safeguard appeared worse (21.0% vs 23.0%) but this was due to minimal policies not leveraging safety tuning capacity.

### 3. Baseline Model Less Sensitive to Policy Length
**Baseline (gpt-oss:20b) performance:**
- Minimal: 23.0%
- Medium: 21.3% (slightly worse)
- Verbose: 26.7% (+3.7 points)

### 4. Medium Policies Don't Help (The "Valley of Death")
**Medium-length policies (300-500 tokens) underperform:**
- Baseline: 21.3% (worse than minimal 23.0%)
- Safeguard: 30.3% (better than minimal but worse than verbose)
- **Finding**: Either go minimal for speed or verbose for quality - medium is worst of both worlds

### 5. ⚠️ CRITICAL: Synthetic Data Doesn't Generalize
- Synthetic benchmarks: 56-59% multi-policy accuracy
- Real adversarial data (WildGuard): **23.0%** multi-policy accuracy
- **36-point drop** shows synthetic test data is NOT representative
- **Experiments 03-04 are INVALID** (self-created test data)

### 6. Throughput vs Quality Tradeoff
- Safeguard models: ~195-206 samples/hour (slower, more careful)
- Baseline with verbose: ~317 samples/hour (faster)
- **But**: Safeguard + verbose achieves +35% better accuracy despite being slower

### 7. Policy Length Impact is Model-Dependent
**Key interaction discovered:**
- Safety tuning creates capacity for detailed policy reasoning
- But minimal policies fail to activate this capacity
- Baseline models show smaller benefit from verbose policies (+16% vs +71% for safeguard)
