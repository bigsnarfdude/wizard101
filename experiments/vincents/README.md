# Vincent's Cascade Evaluation Workspace

Self-contained folder for running cascade experiments. All outputs go to `./outputs/`.

## Quick Start

```bash
cd experiments/vincents
chmod +x *.sh

# Full cascade (L0 → L1 → L2)
./run_all.sh

# Or individual steps
./run_l2_standalone.sh    # L2 only (after L0/L1 done)
./run_blindspot_test.sh   # Test L1 on L0 misses
```

## Directory Structure

```
vincents/
├── README.md              # This file
├── eval_layered_batch.py  # Main evaluation code (self-contained copy)
├── run_all.sh             # Full cascade evaluation
├── run_l2_standalone.sh   # L2 only (gpt-oss:120b)
├── run_blindspot_test.sh  # Blindspot coverage test
├── run_cascade.py         # Python entry point
└── outputs/               # All results go here
    ├── l0_results.json
    ├── l1_results.json
    ├── l2_results.json
    ├── final_scores.json
    └── blindspot_test_results.json
```

## Data Locations

### Benchmarks (input)
```
../../data/benchmark/harmbench_test.json      (500)
../../data/benchmark/simplesafetytests.json   (100)
../../data/evaluation/xstest.json             (450)
```

### Models
```
# L0 - DeBERTa (local)
../cascade/models/l0_bouncer_full

# L1 - GuardReasoner-8B (HuggingFace)
yueliu1999/GuardReasoner-8B

# L2 - gpt-oss:120b (Ollama)
ollama list
```

## Manual Layer Commands

```bash
cd experiments/vincents
source ../guardreasoner/venv/bin/activate

# L0
python eval_layered_batch.py --layer l0 --benchmark all

# L1
python eval_layered_batch.py --layer l1

# L2
python eval_layered_batch.py --layer l2

# Combine
python eval_layered_batch.py --combine
```

All outputs automatically go to `./outputs/`.

## Check Results

```bash
# Quick summary
python -c "import json; d=json.load(open('outputs/final_scores.json')); m=d['metrics']['overall']; print(f\"Accuracy: {m['accuracy']*100:.1f}%, F1: {m['f1']*100:.1f}%\")"

# L2 accuracy
python -c "import json; d=json.load(open('outputs/final_scores.json')); print(f\"L2 Accuracy: {d['metrics']['l2_effectiveness']['accuracy']*100:.1f}%\")"
```

## Key Configuration

In `../eval_layered_batch.py`:
- Line 131: `confidence < 0.9` - L0 threshold
- Line 240: `max_new_tokens=512` - L1 speed
- Line 331: `L2_MODEL = "gpt-oss:120b"`

## Expected Results

| Layer | Accuracy | Speed |
|-------|----------|-------|
| L0 | 95.2% (when confident) | 195/sec |
| L1 | 88.5% | 8s each |
| L2 | 87.5% | 1.2s each |
| **Overall** | **94.9%** | ~9 min |
