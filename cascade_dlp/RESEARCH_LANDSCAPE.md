# LLM Data Leakage Research Landscape

## Two Distinct Threat Models

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│   ADVERSARIAL EXFILTRATION      │    │   ACCIDENTAL MEMORIZATION       │
│   (Prompt Injection)            │    │   (Training Data Extraction)    │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│ • Intentional attacks           │    │ • Unintended statistical leak   │
│ • Offensive security methods    │    │ • Privacy/ML theory methods     │
│ • Attack Success Rate (ASR)     │    │ • Extraction probability        │
│ • OWASP #1 LLM risk             │    │ • Larger models = MORE vulner.  │
└─────────────────────────────────┘    └─────────────────────────────────┘
              │                                       │
              └───────────────┬───────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   RESEARCH GAP    │
                    │  <5% papers       │
                    │  address BOTH     │
                    └───────────────────┘
```

## Key Papers by Domain

### Adversarial Exfiltration

| Paper | Authors | Key Finding |
|-------|---------|-------------|
| "Not what you've signed up for" (2023) | Greshake et al. | Data theft via indirect injection, AISec Best Paper |
| HouYi Framework (2023) | Liu et al. | 86% attack success on 36 real apps |
| Self-replicating prompts (2024) | - | Prompts propagate between LLM instances |
| DefensiveTokens (2025) | - | 5 tokens reduced attack success to 0.24% |

### Accidental Memorization

| Paper | Authors | Key Finding |
|-------|---------|-------------|
| "Secret Sharer" (2019) | Carlini et al. | Introduced exposure metric, extracted CC numbers |
| "Extracting Training Data" (2021) | Carlini et al. | **Larger models MORE vulnerable** |
| "Quantifying Memorization" (2022) | Carlini et al. | Memorization increases with capacity, duplication, prompt length |
| "Scalable Extraction" (2023) | Carlini et al. | Divergence attack: 150x higher extraction from aligned models |

## Benchmarks

### For Adversarial Testing (Prompt Injection)

| Benchmark | Size | Source | Use |
|-----------|------|--------|-----|
| **PINT** | 4,314 samples | Lakera AI | 12 languages, embedded attacks |
| **LLMail-Inject** | 208,095 attacks | Microsoft MSRC | Email assistant scenario |
| **CyberSecEval 3** | 1,000 cases | Meta | Multimodal visual injections |
| **Securing AI Agents** | 847 cases | - | 5 sophistication levels |

### For Accidental Leakage (PII/Secrets)

| Benchmark | Size | Source | Use |
|-----------|------|--------|-----|
| **SecretBench** | 15,084 secrets | NC State/MSR | 8 categories, 49 languages |
| **PIILO** | 22,000 essays | Kaggle | 7 PII categories, 95%+ F1 achievable |
| **StarPII** | 20,961 secrets | BigCode | 31 programming languages |
| **BIG-Bench canary** | 1 GUID | Google | Benchmark contamination detection |

### For Code Leakage

| Benchmark | Size | Use |
|-----------|------|-----|
| **LessLeak-Bench** | 83 benchmarks | Python/Java/C++ leakage ratios |
| **BigCloneBench** | Standard | Clone detection (4 types) |
| **LBPP** | 161 prompts | Minimal contamination evaluation |

## Defense Approaches

### Adversarial Defenses

```python
# Input filtering
├── Microsoft Prompt Shields (ML-based)
├── DefensiveTokens (5 special tokens)
├── Multi-agent validation pipelines
└── Hierarchical prompt guardrails

# System-level
├── Data tagging (trusted sources)
├── Rate limiting
└── RAG-in-gateway architectures
```

### Accidental Leakage Defenses

```python
# Data-level
├── Deduplication (10x reduction, Lee et al.)
├── PII filtering (imperfect - style transfer bypasses)
└── Canary monitoring

# Training-level
├── Differential Privacy (DP-SGD)
├── Parameter-efficient fine-tuning + DP
└── VaultGemma (1B DP model, no memorization)

# Output-level
├── Perplexity anomaly detection
├── Presidio/NER PII filtering
└── Secret detection (TruffleHog, Gitleaks)
```

## Frontier Lab Architectures

### OpenAI GPT-4o

```
Pretraining → Model Safeguards → Output Classifiers → PII Filters
(data filter)  (deliberative     (real-time         (input
               alignment)        content filter)     filtering)
```

### Anthropic CBRN Filtering

```
Stage 1: Finetuned Constitutional classifier (scan all)
    │
    ▼
Stage 2: Prompted Constitutional classifier (rerank flagged)
    │
    ▼
Stage 3: Named entities matcher (parallel flag)

Result: 33% harm reduction, no utility loss
```

### DeepMind VaultGemma

- 1B parameter fully DP model
- Sequence-level DP (ε ≤ 2.0)
- No detectable memorization
- World's largest DP LLM

## Tool Recommendations

### PII Detection

| Tool | Approach | Performance |
|------|----------|-------------|
| **Presidio** | Rule + ML hybrid | Production-ready, extensible |
| **DeBERTa-v3** | Fine-tuned NER | 95%+ F1 on PIILO |
| **GLiNER** | Zero-shot | ~81% F1 multi-domain |

### Secret Detection

| Tool | Strength | Trade-off |
|------|----------|-----------|
| **GitHub Secret Scanner** | Best precision (75%) | Lower recall |
| **Gitleaks** | Best recall (88%) | More false positives |
| **TruffleHog** | 800+ secret types | Comprehensive but slower |

### Prompt Injection

| Tool | Source | Use |
|------|--------|-----|
| **Prompt Shields** | Microsoft | ML-based input filter |
| **Prompt Guard** | Meta | LLM-based detection |
| **Lakera Guard** | Lakera | Commercial solution |

## The Unified Framework Gap

**Current state**: Defenses optimize for ONE threat model, potentially weakening against the other.

**Missing research**:
1. Benchmarks testing BOTH threat vectors simultaneously
2. Analysis of defense trade-offs between domains
3. Unified metrics across ASR and extraction probability
4. Integrated lifecycle protection

**Research opportunity for DLP-Guard**:
- Build evaluation combining benchmarks from both fields
- Measure whether defenses conflict
- Develop unified framework addressing both threats
- Publish findings to advance field

## Practical Cascade Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│ Stage 1: Input  │  Fast models (86M-1B params)
│ Classification  │  Detect obvious threats
└────────┬────────┘
         │
    flagged?
         │
    ┌────┴────┐
    │         │
   yes        no
    │         │
    ▼         │
┌─────────┐   │
│ Stage 2 │   │  Larger models (7B-13B)
│ Context │   │  Rerank, reduce FPs
│ Analysis│   │
└────┬────┘   │
     │        │
     ▼        │
   block?     │
     │        │
 ┌───┴───┐    │
 │       │    │
yes      no   │
 │       │    │
 ▼       └────┤
BLOCK         │
              ▼
      [LLM Processing]
              │
              ▼
┌─────────────────┐
│ Stage 3: Output │  Specialized detectors
│ Validation      │  PII NER, secret regex,
│                 │  perplexity checks
└────────┬────────┘
         │
    leak detected?
         │
    ┌────┴────┐
   yes        no
    │         │
    ▼         ▼
 REDACT     ALLOW
              │
              ▼
┌─────────────────┐
│ Stage 4:       │  Pattern detection
│ Monitoring     │  Continuous improvement
└─────────────────┘
```

## Key Insights for DLP-Guard

1. **Larger models = MORE vulnerable** to memorization (counterintuitive)
2. **Deduplication is highly effective** - 10x reduction in memorized output
3. **PII filters can be bypassed** via style transfer prompts
4. **DefensiveTokens** achieved 0.24% attack success with just 5 tokens
5. **Combined defenses** reduced attack success from 73.2% → 8.7%
6. **No unified framework exists** - this is the research opportunity

## Evaluation Strategy

### For Adversarial Threats
- Attack Success Rate (ASR)
- False Positive Rate
- Task Performance Retention
- Use: PINT, LLMail-Inject, CyberSecEval 3

### For Accidental Leakage
- Extraction probability
- Exposure metric
- Perplexity ratios
- Use: SecretBench, PIILO, BIG-Bench canary

### Unified Evaluation (New)
- Test both on same system
- Measure defense conflicts
- Track security-utility Pareto frontier
- Custom benchmark combining both domains

## Strategic Recommendation

**For Tier 1 AI safety team**:

1. **Treat as separate but interconnected threats**
   - Specialized expertise in both domains
   - Don't assume one defense covers both

2. **Deploy layered defenses**
   - Input filtering (adversarial)
   - Output validation (both)
   - Perplexity checks (memorization)
   - PII/secret detection (both)

3. **Build comprehensive evaluation**
   - Combine benchmarks from both fields
   - Detect defense conflicts
   - Measure unified metrics

4. **Contribute to research gap**
   - Develop unified evaluation framework
   - Publish findings
   - Advance field understanding

---

*Source: Research compilation 2025-11-23*
*Status: Literature review complete, ready for implementation*
