# Experiment Plan: Comprehensive Benchmark Evaluation

## Overview

Two planned experiments to complete the wizard101 evaluation:
1. Download and integrate SALAD-Bench
2. Run full benchmark evaluation across all datasets

---

## Experiment 1: SALAD-Bench Integration

### Dataset Info

**Source**: [OpenSafetyLab/Salad-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data)
**Paper**: [SALAD-Bench (ACL 2024)](https://arxiv.org/abs/2402.05044)
**GitHub**: [OpenSafetyLab/SALAD-BENCH](https://github.com/OpenSafetyLab/SALAD-BENCH)

### Dataset Structure

| Subset | Samples | Purpose |
|--------|---------|---------|
| base_set | 21,300 | Core harmful questions |
| attack_enhanced_set | 5,000 | Jailbreak-augmented questions |
| defense_enhanced_set | 200 | Defense-augmented questions |
| mcq_set | 3,840 | Multiple-choice format |

### Taxonomy (3-level hierarchy)

- **6 Domains** (Level 1)
- **16 Tasks** (Level 2)
- **66 Categories** (Level 3)

### Download Script

```python
# scripts/utils/download_salad_bench.py

from datasets import load_dataset
import json
from pathlib import Path

def download_salad_bench():
    """Download SALAD-Bench dataset from HuggingFace."""

    output_dir = Path("data/benchmark")

    # Download base set (primary evaluation)
    print("Downloading SALAD-Bench base_set...")
    base = load_dataset("OpenSafetyLab/Salad-Data", name="base_set", split="train")

    # Convert to our format
    samples = []
    for row in base:
        samples.append({
            "id": row.get("qid"),
            "prompt": row.get("question"),
            "source": row.get("source"),
            "category_1": row.get("1-category"),
            "category_2": row.get("2-category"),
            "category_3": row.get("3-category"),
            "label": "harmful"  # All SALAD-Bench questions are harmful
        })

    with open(output_dir / "salad_bench_base.json", "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to data/benchmark/salad_bench_base.json")

    # Download attack-enhanced set
    print("Downloading SALAD-Bench attack_enhanced_set...")
    attack = load_dataset("OpenSafetyLab/Salad-Data", name="attack_enhanced_set", split="train")

    attack_samples = []
    for row in attack:
        attack_samples.append({
            "id": row.get("aid"),
            "prompt": row.get("augq"),
            "method": row.get("method"),
            "label": "harmful"
        })

    with open(output_dir / "salad_bench_attack.json", "w") as f:
        json.dump(attack_samples, f, indent=2)

    print(f"Saved {len(attack_samples)} samples to data/benchmark/salad_bench_attack.json")

if __name__ == "__main__":
    download_salad_bench()
```

### Expected Output

- `data/benchmark/salad_bench_base.json` (~21K samples, ~10MB)
- `data/benchmark/salad_bench_attack.json` (~5K samples, ~3MB)

---

## Experiment 2: Full Benchmark Evaluation

### Current Benchmarks

| Dataset | Samples | Type | Status |
|---------|---------|------|--------|
| wildjailbreak | 88,444 | Jailbreak | Available |
| combined_benchmark | 10,384 | Mixed | Available |
| toxicchat_test | 5,083 | Toxicity | Available |
| beavertails_30k | 3,021 | Jailbreak | Available |
| openai_moderation | 1,680 | Moderation | Available |
| sgbench | 1,442 | Safety | Available |
| harmbench_test | 500 | Harm | Available |
| strongreject | 313 | Adversarial | Available |
| jailbreakbench | 200 | Jailbreak | Available |
| simplesafetytests | 100 | Sanity | Available |
| **salad_bench_base** | 21,300 | Comprehensive | **To Download** |
| **salad_bench_attack** | 5,000 | Jailbreak | **To Download** |

**Total**: ~137K samples across 12 benchmarks

### Evaluation Script

```python
# scripts/eval/evaluate_all_benchmarks.py

import json
from pathlib import Path
from cascade_inbound import SafetyCascade, CascadeConfig
from tqdm import tqdm

BENCHMARKS = [
    ("simplesafetytests.json", "SimpleSafetyTests"),
    ("jailbreakbench.json", "JailbreakBench"),
    ("strongreject.json", "StrongREJECT"),
    ("harmbench_test.json", "HarmBench"),
    ("sgbench.json", "SGBench"),
    ("openai_moderation.json", "OpenAI Moderation"),
    ("beavertails_30k.json", "BeaverTails"),
    ("toxicchat_test.json", "ToxicChat"),
    ("combined_benchmark.json", "Combined"),
    ("salad_bench_base.json", "SALAD-Bench Base"),
    ("salad_bench_attack.json", "SALAD-Bench Attack"),
    # ("wildjailbreak.json", "WildJailbreak"),  # Large - run separately
]

def evaluate_benchmark(cascade, benchmark_path, name):
    """Evaluate cascade on a single benchmark."""

    with open(benchmark_path) as f:
        data = json.load(f)

    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    layer_counts = {"L0": 0, "L1": 0, "L2": 0}

    for sample in tqdm(data, desc=name):
        prompt = sample.get("prompt") or sample.get("text") or sample.get("question")
        true_label = normalize_label(sample.get("label", "harmful"))

        result = cascade.classify(prompt)
        pred_label = result.label

        # Update confusion matrix
        if true_label == "harmful" and pred_label == "harmful":
            results["tp"] += 1
        elif true_label == "safe" and pred_label == "harmful":
            results["fp"] += 1
        elif true_label == "safe" and pred_label == "safe":
            results["tn"] += 1
        else:
            results["fn"] += 1

        # Track layer distribution
        layer_counts[result.stopped_at] += 1

    # Calculate metrics
    tp, fp, tn, fn = results["tp"], results["fp"], results["tn"], results["fn"]
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "name": name,
        "samples": len(data),
        "accuracy": round(accuracy * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1 * 100, 1),
        "layer_distribution": layer_counts,
        "confusion": results
    }

def normalize_label(label):
    if label.lower() in ['safe', 'benign', 'unharmful', 'harmless']:
        return 'safe'
    return 'harmful'

def main():
    config = CascadeConfig(l0_confidence_threshold=0.9, enable_l2=True)
    cascade = SafetyCascade(config)

    results = []
    benchmark_dir = Path("data/benchmark")

    for filename, name in BENCHMARKS:
        path = benchmark_dir / filename
        if path.exists():
            result = evaluate_benchmark(cascade, path, name)
            results.append(result)
            print(f"\n{name}: Acc={result['accuracy']}% F1={result['f1']}%")

    # Save results
    with open("experiments/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(f"{'Benchmark':<25} {'Samples':>8} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<25} {r['samples']:>8} {r['accuracy']:>5.1f}% {r['precision']:>5.1f}% {r['recall']:>5.1f}% {r['f1']:>5.1f}%")

if __name__ == "__main__":
    main()
```

### Expected Runtime

| Benchmark | Samples | Est. Time (L0 only) | Est. Time (Full) |
|-----------|---------|---------------------|------------------|
| simplesafetytests | 100 | <1s | ~1min |
| jailbreakbench | 200 | <1s | ~2min |
| strongreject | 313 | <1s | ~3min |
| harmbench_test | 500 | 1s | ~5min |
| sgbench | 1,442 | 3s | ~15min |
| openai_moderation | 1,680 | 3s | ~17min |
| beavertails_30k | 3,021 | 6s | ~30min |
| toxicchat_test | 5,083 | 10s | ~50min |
| salad_bench_attack | 5,000 | 10s | ~50min |
| combined_benchmark | 10,384 | 20s | ~2hr |
| salad_bench_base | 21,300 | 40s | ~4hr |
| wildjailbreak | 88,444 | 3min | ~15hr |

**Total (excluding wildjailbreak)**: ~50K samples, ~8 hours full cascade

---

## Execution Plan

### Phase 1: Download SALAD-Bench
```bash
cd /Users/vincent/development/wizard101
python scripts/utils/download_salad_bench.py
```

### Phase 2: Quick Sanity Check (L0 only)
```bash
python scripts/eval/evaluate_all_benchmarks.py --l0-only
```

### Phase 3: Full Evaluation (overnight)
```bash
python scripts/eval/evaluate_all_benchmarks.py --full
```

### Phase 4: Generate Report
```bash
python scripts/eval/generate_benchmark_report.py
```

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Overall Accuracy | >90% | Across all benchmarks |
| Recall on harmful | >95% | Safety-critical |
| False Positive Rate | <10% | Avoid over-refusal |
| SALAD-Bench Base | >85% | New comprehensive benchmark |
| SALAD-Bench Attack | >80% | Adversarial robustness |

---

## Output Files

After experiments:
```
experiments/
├── EXPERIMENT_PLAN.md          # This file
├── benchmark_results.json      # Raw results
├── BENCHMARK_REPORT.md         # Final report
└── plots/
    ├── accuracy_by_benchmark.png
    └── layer_distribution.png
```

---

*Created: 2025-11-24*

---

## Phase 5: Cascade Optimization (NEXT STEPS)

*Added: 2025-11-27*

### Problem Statement

Current eval results show:
- High recall (catching harmful content) but low precision (too many false positives)
- Expensive L3 Judge called too frequently
- At scale (50% of traffic = safety checks), cost explodes

### Current Results (11/12 benchmarks complete)

| Dataset | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| SimpleSafetyTests | 95.0% | 100.0% | 95.0% | 97.4% |
| JailbreakBench | 65.5% | 59.3% | 99.0% | 74.2% |
| StrongREJECT | 95.5% | 100.0% | 95.5% | 97.7% |
| HarmBench | 99.6% | 100.0% | 99.6% | 99.8% |
| SGBench | 89.6% | 100.0% | 89.6% | 94.5% |
| OpenAI Moderation | 74.1% | 55.0% | 90.8% | 68.5% |
| BeaverTails | 70.1% | 69.5% | 85.3% | 76.6% |
| ToxicChat | 91.5% | 44.8% | 81.5% | 57.8% |
| SALAD-Bench Attack | 90.8% | 100.0% | 90.8% | 95.2% |
| SALAD-Bench Base | 78.7% | 100.0% | 78.7% | 88.0% |
| OR-Bench | 🔄 running | - | - | - |
| Combined | pending | - | - | - |

**Key Finding**: Variance across datasets (65-99%) IS the story. Single accuracy numbers are meaningless.

### The Cost Problem

```
At 1M requests/day, 50% needing safety checks:
- Current: L3 runs on ~20% of traffic = K+/month  
- Target: L3 runs on <1% of traffic = K/month
```

### Optimization Strategy: Confidence-Based Fast Paths

```
Traffic Flow Target:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L0 Bouncer (fast, cheap)     → 85% resolved
  ├── Score > 0.95 harmful   → BLOCK immediately  
  ├── Score < 0.05 safe      → PASS immediately
  └── Score 0.05-0.95        → escalate

L1 Analyst (medium cost)     → 13% resolved
  ├── High confidence        → resolve locally
  └── Low confidence         → escalate

L2 Gauntlet (reduced)        → 1.9% resolved
  ├── 2-3 experts (not 6)
  └── Majority vote

L3 Judge (rare, expensive)   → 0.1% only
  └── True edge cases only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Implementation Tasks

#### Task 1: Analyze L0 Score Distribution
```bash
# Extract L0 confidence scores from eval run
python scripts/analysis/l0_score_distribution.py

# Questions to answer:
# - What % of samples fall in uncertain range (0.05-0.95)?
# - Where does L0 fail? (false positives vs false negatives)
# - Can we shift the distribution with retraining?
```

#### Task 2: Add Calibration to L0
```python
# Convert raw logits to calibrated probabilities
# Use temperature scaling or Platt scaling
# Goal: When L0 says 95% confident, it should BE 95% accurate
```

#### Task 3: Retrain L0 with High-Confidence Examples
```
Training data priorities:
1. Clear SAFE examples that current L0 escalates (reduce FP)
2. Borderline cases from OpenAI Mod, BeaverTails
3. Adversarial examples from JailbreakBench
```

#### Task 4: Benchmark L2 Experts
```bash
# Which experts actually change outcomes?
python scripts/analysis/l2_expert_ablation.py

# If expert_3 and expert_5 always agree with majority, remove them
```

#### Task 5: Add Confidence Thresholds
```python
# cascade_inbound/config.py updates:
class CascadeConfig:
    l0_block_threshold: float = 0.95   # Immediate block
    l0_pass_threshold: float = 0.05    # Immediate pass
    l1_confidence_threshold: float = 0.85
    l2_enabled_experts: List[str] = [toxicity, jailbreak, harm]
```

#### Task 6: Measure Layer Distribution
```
After optimization, track:
- % resolved at L0
- % resolved at L1
- % resolved at L2
- % reaching L3

Target: 85% / 13% / 1.9% / 0.1%
```

### Success Criteria (Post-Optimization)

| Metric | Current | Target |
|--------|---------|--------|
| L0 resolution rate | ~60%? | 85% |
| L3 call rate | ~20%? | <1% |
| Overall accuracy | 80% | >85% |
| Precision (borderline datasets) | 55% | >70% |
| Cost per 1K safety checks |  | /bin/zsh.10 |

### Files to Create

```
scripts/analysis/
├── l0_score_distribution.py     # Analyze L0 confidence spread
├── l2_expert_ablation.py        # Which experts matter?
├── layer_traffic_analysis.py    # Where does traffic stop?
└── calibration_analysis.py      # Is L0 well-calibrated?

cascade_inbound/
├── calibration.py               # Platt/temperature scaling
└── config.py                    # Add confidence thresholds
```

---

*Next: Complete OR-Bench + Combined eval, then start Task 1*
