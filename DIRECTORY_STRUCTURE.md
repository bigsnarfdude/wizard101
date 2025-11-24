# Wizard101 Directory Structure

## Current State (After Reorganization)

```
wizard101/                              # 43 active Python files, 13 active Markdown files
│
├── README.md                           # Main project documentation
├── DIRECTORY_STRUCTURE.md              # This file
├── LICENSE                             # MIT
├── wizard101.png                       # Architecture diagram
├── .gitignore
│
├── cascade_inbound/                    # LAYER 1: Safety Classification [FINAL]
│   ├── README.md                       #   Component documentation
│   ├── __init__.py                     #   Package exports
│   ├── cascade.py                      #   Main orchestrator
│   ├── l0_bouncer.py                   #   DeBERTa classifier (2ms)
│   ├── l1_analyst.py                   #   GuardReasoner-8B (8s)
│   ├── l2_gauntlet.py                  #   6-expert voting (0.18s)
│   ├── l3_judge.py                     #   Final authority (2-3s)
│   ├── example.py                      #   Usage example
│   ├── install.sh                      #   Setup script
│   ├── requirements.txt                #   Dependencies
│   └── models/                         #   Model weights (5.3GB)
│       └── guardreasoner-8b-4bit/
│
├── cascade_refusals/                   # LAYER 2: Refusal Generation [FINAL]
│   ├── README.md
│   ├── refusal_pipeline.py             #   Main pipeline
│   ├── refusal_generator.py            #   Response templates
│   ├── test_llama_guard.py             #   Llama Guard router
│   ├── api.py                          #   FastAPI wrapper
│   ├── compare_models.py               #   Model comparison
│   └── requirements.txt
│
├── cascade_quarantine/                 # LAYER 3: Input Sanitization [PHASE 1]
│   ├── README.md
│   ├── example.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── capture.py                  #   Edge case capture
│   │   ├── database.py                 #   SQLite storage
│   │   ├── models.py                   #   Data models
│   │   └── config.py                   #   Configuration
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_capture.py
│   │   └── test_database.py
│   └── logs/
│
├── cascade_dlp/                        # LAYER 5: Data Loss Prevention [FINAL]
│   ├── README.md                       #   Full documentation
│   ├── install.sh                      #   Installation script
│   ├── example.py                      #   Usage example
│   ├── requirements.txt
│   ├── cascade_dlp/                    #   Python package
│   │   ├── __init__.py
│   │   ├── cascade.py                  #   Main DLP pipeline
│   │   └── config.py
│   └── experiments/                    #   Benchmarks & tests
│       ├── eval/
│       └── tests/
│
├── scripts/                            # UTILITY SCRIPTS
│   ├── README.md                       #   Script documentation
│   ├── train/                          #   Training scripts
│   │   ├── train_l0_bouncer.py
│   │   ├── train_l0_mega.py            #   (recommended)
│   │   ├── train_l0_full.py
│   │   ├── train_l0_12k.py
│   │   ├── build_mega_dataset.py
│   │   ├── build_12k_dataset.py
│   │   └── build_test_set.py
│   ├── eval/                           #   Evaluation scripts
│   │   ├── evaluate_cascade.py         #   (recommended)
│   │   ├── evaluate_cascade_batch.py
│   │   ├── evaluate_heretic.py
│   │   ├── evaluate_heretic_full.py
│   │   ├── eval_l1_baseline.py
│   │   └── tune_l0_threshold.py
│   └── utils/                          #   Utilities
│       ├── quantize_guardreasoner.py
│       ├── download_guardreasoner_test.py
│       └── compare_methods.py
│
├── docs/                               # DOCUMENTATION HUB
│   └── README.md                       #   Links to all component docs
│
├── data/                               # TRAINING DATA
│   ├── raw/
│   ├── training/
│   ├── evaluation/
│   └── benchmark/
│
├── papers/                             # REFERENCE PAPERS
│
├── logs/                               # APPLICATION LOGS
│
└── archive/                            # ARCHIVED EXPERIMENTS (9.6GB)
    ├── README.md                       #   Explains what's archived
    ├── 2024-11-dlp-old/                #   Old DLP implementation
    │   ├── src/
    │   ├── docs/
    │   └── [exploration scripts]
    └── 2024-11-experiments/            #   All experimental code
        ├── cascade-development/        #   Early cascade (was duplicate)
        ├── guardreasoner-training/     #   LoRA training experiments
        ├── vincents-personal/          #   Personal experiments
        ├── benchmarks/                 #   JSON benchmark files
        ├── docs/                       #   Research notes
        └── misc/                       #   One-off scripts
```

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Active Python files | 189 | 43 |
| Active Markdown files | 105 | 13 |
| Root `experiments/` | 9.5 GB (messy) | Gone (archived) |
| DLP confusion | Yes (src vs dlp2) | No (dlp2 promoted) |
| Training scripts in cascade_inbound | 7 | 0 (moved to scripts/) |
| Duplicate code | Yes | No |

---

## Component Status

| Layer | Component | Status | Location |
|-------|-----------|--------|----------|
| 1 | Inbound Safety | Production | `cascade_inbound/` |
| 2 | Refusal Generation | Production | `cascade_refusals/` |
| 3 | Input Quarantine | Phase 1 | `cascade_quarantine/` |
| 4 | LLM | External | (user's model) |
| 5 | Data Loss Prevention | Production | `cascade_dlp/` |

---

## Quick Reference

### Run the cascade
```python
from cascade_inbound import SafetyCascade
cascade = SafetyCascade()
result = cascade.classify("user input here")
```

### Train L0 bouncer
```bash
python scripts/train/train_l0_mega.py
```

### Evaluate cascade
```bash
python scripts/eval/evaluate_cascade.py
```

### Check DLP
```bash
cd cascade_dlp
./install.sh
source venv/bin/activate
python example.py
```

```python
from cascade_dlp import DLPCascade
dlp = DLPCascade()
result = dlp.process("text with potential PII")
```
