# Wizard101 Directory Structure

## Current State (November 2025)

```
wizard101/
│
├── README.md                           # Main project documentation
├── DIRECTORY_STRUCTURE.md              # This file
├── LICENSE                             # MIT
├── wizard101.png                       # Architecture diagram
├── .gitignore
│
├── cascade_inbound/                    # LAYER 1: Safety Classification
│   ├── README.md                       #   Component documentation
│   ├── __init__.py                     #   Package exports
│   ├── cascade.py                      #   Main orchestrator
│   ├── l0_bouncer.py                   #   DeBERTa classifier (2ms)
│   ├── l1_analyst.py                   #   GuardReasoner-8B (8s)
│   ├── l2_gauntlet.py                  #   gpt-oss-safeguard:20b (0.18s)
│   ├── l3_judge.py                     #   Final authority
│   ├── example.py                      #   Usage example
│   ├── install.sh                      #   Setup script
│   ├── requirements.txt                #   Dependencies
│   └── models/                         #   Model weights
│       └── guardreasoner-8b-4bit/      #   4-bit quantized (5GB)
│           ├── config.json
│           ├── generation_config.json
│           ├── model.safetensors.index.json
│           ├── special_tokens_map.json
│           ├── tokenizer.json
│           └── tokenizer_config.json
│
├── cascade_refusals/                   # LAYER 2: Refusal Generation
│   ├── README.md
│   ├── refusal_pipeline.py             #   Main pipeline (Llama Guard 3)
│   ├── refusal_generator.py            #   Response templates
│   ├── test_llama_guard.py             #   Llama Guard router
│   ├── api.py                          #   FastAPI wrapper
│   ├── compare_models.py               #   Model comparison
│   └── requirements.txt
│
├── cascade_quarantine/                 # LAYER 3: Prompt Injection Defense
│   ├── README.md
│   ├── example.py                      #   Usage example
│   ├── src/
│   │   ├── __init__.py
│   │   ├── quarantine.py               #   Intent extraction (Qwen3:4b)
│   │   ├── classifier.py               #   ML injection classifier (99.2%)
│   │   ├── pipeline.py                 #   Full detection pipeline
│   │   ├── capture.py                  #   Low-confidence capture
│   │   ├── audit.py                    #   Audit logging
│   │   ├── database.py                 #   SQLite storage
│   │   ├── models.py                   #   Data models
│   │   └── config.py                   #   Configuration
│   ├── models/
│   │   └── injection_classifier.pkl    #   Trained TF-IDF + LogReg model
│   ├── experiments/
│   │   ├── benchmark.py                #   Classifier benchmarking
│   │   ├── benchmark_large.py          #   Large-scale eval
│   │   ├── smoke_test.py               #   Quick validation
│   │   └── dataset_xtram1.json         #   xTRam1 test data
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_quarantine.py
│   │   ├── test_quarantine_phase2.py
│   │   ├── test_classifier.py
│   │   └── test_phase4.py              #   Phase 4 integration tests
│   ├── data/
│   │   └── raw/                        #   Raw training data
│   └── logs/                           #   Capture logs
│
├── cascade_dlp/                        # LAYER 5: Data Loss Prevention
│   ├── README.md                       #   Full documentation
│   ├── install.sh                      #   Installation script
│   ├── example.py                      #   Usage example
│   ├── requirements.txt
│   ├── cascade_dlp/                    #   Python package
│   │   ├── __init__.py
│   │   ├── cascade.py                  #   Main DLP pipeline (Presidio)
│   │   └── config.py
│   └── experiments/
│       ├── eval/
│       │   └── benchmark.py            #   Benchmark runner
│       └── tests/
│
├── scripts/                            # UTILITY SCRIPTS
│   ├── README.md                       #   Script documentation
│   ├── normalize_sgbench.py            #   Dataset normalization
│   ├── download_wildjailbreak.py       #   Dataset download
│   ├── train/                          #   Training scripts
│   │   ├── train_l0_bouncer.py
│   │   ├── train_l0_mega.py            #   Recommended trainer
│   │   ├── train_l0_full.py
│   │   ├── train_l0_12k.py
│   │   ├── build_mega_dataset.py
│   │   ├── build_12k_dataset.py
│   │   └── build_test_set.py
│   ├── eval/                           #   Evaluation scripts
│   │   ├── evaluate_cascade.py         #   Main evaluator
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
├── data/                               # DATASETS
│   ├── DATA_README.md                  #   Dataset documentation
│   ├── DATASET_RESEARCH.md             #   Research notes
│   ├── raw/                            #   Raw source data
│   │   ├── harmful_behaviors.json
│   │   └── harmless_alpaca.json
│   ├── training/                       #   Training sets
│   │   ├── combined_train.json
│   │   ├── mega_train.json
│   │   └── train_12k.json
│   ├── evaluation/                     #   Eval/test sets
│   │   ├── combined_test.json
│   │   ├── guardreasoner_test_5k.json
│   │   ├── guardreasoner_test_10k.json
│   │   ├── wildguard_full_benchmark.json
│   │   └── xstest.json
│   └── benchmark/                      #   Public benchmarks
│       ├── beavertails_30k.json
│       ├── combined_benchmark.json
│       ├── harmbench_test.json
│       ├── jailbreakbench.json
│       ├── openai_moderation.json
│       ├── sgbench.json
│       ├── simplesafetytests.json
│       ├── strongreject.json
│       ├── toxicchat_test.json
│       └── wildjailbreak.json
│
├── docs/                               # DOCUMENTATION
│   └── README.md                       #   Links to component docs
│
├── papers/                             # REFERENCE PAPERS
│
├── logs/                               # APPLICATION LOGS
│
└── archive/                            # ARCHIVED EXPERIMENTS
    ├── README.md                       #   Archive index
    ├── 2024-11-dlp-old/                #   Old DLP implementation
    └── 2024-11-experiments/            #   R&D experiments
        ├── cascade-development/        #   Early cascade iterations
        ├── guardreasoner-training/     #   LoRA training experiments
        ├── vincents-personal/          #   Personal experiments
        ├── benchmarks/                 #   Benchmark runners
        ├── docs/                       #   Research notes
        └── misc/                       #   One-off scripts
```

---

## Component Status

| Layer | Component | Status | Location |
|-------|-----------|--------|----------|
| 1 | Inbound Safety | Production Ready | `cascade_inbound/` |
| 2 | Refusal Generation | Built | `cascade_refusals/` |
| 3 | Input Quarantine | Phases 1-3 Complete | `cascade_quarantine/` |
| 4 | Privileged LLM | External | (user's model) |
| 5 | Data Loss Prevention | Production Ready | `cascade_dlp/` |

---

## Quick Reference

### Run the Safety Cascade
```python
from cascade_inbound import SafetyCascade, CascadeConfig

config = CascadeConfig(l0_confidence_threshold=0.9, enable_l2=True)
cascade = SafetyCascade(config)
result = cascade.classify("user input here")
```

### Check for Prompt Injection
```python
from cascade_quarantine.src.quarantine import Quarantine

quarantine = Quarantine(model="qwen3:4b", use_classifier=True)
result = quarantine.extract_intent("untrusted input")
print(result.injection_detected)  # True/False
```

### Scan for PII/Secrets
```python
from cascade_dlp import DLPCascade

dlp = DLPCascade()
result = dlp.process("text with potential PII")
```

### Generate Refusals
```python
from cascade_refusals import RefusalPipeline

pipeline = RefusalPipeline()
result = pipeline.process("harmful request")
print(result["strategy"])   # HARD/SOFT/CONDITIONAL
print(result["response"])   # Appropriate refusal
```

### Train L0 Bouncer
```bash
python scripts/train/train_l0_mega.py
```

### Evaluate Cascade
```bash
python scripts/eval/evaluate_cascade.py
```

---

## Performance Summary

| Component | Key Metric | Value |
|-----------|------------|-------|
| cascade_inbound | Accuracy | 94.0% |
| cascade_inbound L0 | Latency | 2ms |
| cascade_quarantine | Accuracy | 99.2% |
| cascade_dlp | Precision | 100% (209K samples) |
| cascade_dlp | Latency | 3.7ms median |

---

*Last Updated: 2025-11-24*
