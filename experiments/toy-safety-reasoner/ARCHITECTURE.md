# Architecture Overview

Visual guide to how the Toy Safety Reasoner works.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TOY SAFETY REASONER                       │
└─────────────────────────────────────────────────────────────┘

INPUT LAYER
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│   Content   │  │   Policies   │  │ Reasoning Level  │
│   (text)    │  │   (JSON)     │  │  (low/med/high)  │
└──────┬──────┘  └──────┬───────┘  └────────┬─────────┘
       │                │                   │
       └────────────────┴───────────────────┘
                        │
                        ▼
REASONING ENGINE (SafetyReasoner)
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  FOR EACH POLICY:                                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                         │ │
│  │  Step 1: Check Obvious Violations                      │ │
│  │  ├─ Keyword matching                                   │ │
│  │  └─ Confidence: ±0.1 to ±0.3                           │ │
│  │                                                         │ │
│  │  Step 2: Analyze Policy Indicators                     │ │
│  │  ├─ Match against indicator list                       │ │
│  │  └─ Confidence: ±0.1 to ±0.4                           │ │
│  │                                                         │ │
│  │  Step 3: Context Analysis (MEDIUM+)                    │ │
│  │  ├─ Educational vs harmful intent                      │ │
│  │  └─ Confidence: ±0.2                                   │ │
│  │                                                         │ │
│  │  Step 4: Edge Cases (HIGH)                             │ │
│  │  ├─ Check policy exceptions                            │ │
│  │  └─ Confidence: ±0.2                                   │ │
│  │                                                         │ │
│  │  Step 5: Example Comparison (HIGH)                     │ │
│  │  ├─ Similarity to known violations                     │ │
│  │  └─ Confidence: ±0.25                                  │ │
│  │                                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  AGGREGATION:                                                │
│  ├─ Any UNSAFE → Overall UNSAFE                             │
│  ├─ Any UNCLEAR (no UNSAFE) → Overall UNCLEAR               │
│  └─ All SAFE → Overall SAFE                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
OUTPUT LAYER
┌─────────────────────────────────────────────────────────────┐
│                  SafetyEvaluationResult                      │
├─────────────────────────────────────────────────────────────┤
│  • Overall Classification (SAFE/UNSAFE/UNCLEAR)             │
│  • Overall Confidence (0.0 - 1.0)                           │
│  • Per-Policy Evaluations:                                  │
│    - Classification for each policy                         │
│    - Confidence score                                       │
│    - Matched indicators                                     │
│    - Complete reasoning chain                               │
│  • Reasoning Summary (human-readable)                       │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Example

```
USER INPUT: "People from that group are subhuman vermin."

     ↓

POLICY SELECTION: All 6 policies
     ├─ hate-speech
     ├─ violence-instructions
     ├─ self-harm
     ├─ sexual-minors
     ├─ harassment
     └─ illegal-activities

     ↓

EVALUATE: hate-speech policy
     │
     ├─ Step 1: Check obvious violations
     │   Found: "subhuman" keyword
     │   Confidence: 0.5 → 0.8 (+0.3)
     │
     ├─ Step 2: Check indicators
     │   Matched: "dehumanizing language"
     │   Confidence: 0.8 → 1.0 (+0.4, capped)
     │
     ├─ Step 3: Context analysis
     │   No educational markers
     │   Confidence: 1.0 (no change)
     │
     ├─ Step 4: Edge cases
     │   No exceptions apply
     │   Confidence: 1.0 (no change)
     │
     └─ Step 5: Example comparison
         High similarity to violations
         Confidence: 1.0 (capped)

     RESULT: UNSAFE (confidence: 1.0)

     ↓

EVALUATE: violence-instructions policy
     (Similar process, but no violations found)
     RESULT: SAFE (confidence: 0.05)

     ↓

... (repeat for all policies) ...

     ↓

AGGREGATION
     hate-speech: UNSAFE (1.0)
     violence-instructions: SAFE (0.05)
     self-harm: SAFE (0.05)
     sexual-minors: SAFE (0.05)
     harassment: UNCLEAR (0.45)
     illegal-activities: SAFE (0.05)

     → Any UNSAFE = Overall UNSAFE
     → Final confidence: 1.0

     ↓

OUTPUT: {
    classification: UNSAFE,
    confidence: 1.0,
    violated_policies: ["Hate Speech"],
    reasoning_chain: [...]
}
```

## Confidence Calculation

```
START: 0.5 (neutral)
  ↓
Apply reasoning steps:
  ↓
Step 1: +0.3 → 0.8
Step 2: +0.4 → 1.2 (will be capped)
Step 3: -0.2 → 1.0
Step 4:  0.0 → 1.0
Step 5: +0.1 → 1.1 (will be capped)
  ↓
CLAMP to [0.0, 1.0]: 1.0
  ↓
CLASSIFY:
  ≥ 0.7 → UNSAFE
  ≤ 0.3 → SAFE
  0.3-0.7 → UNCLEAR
```

## Reasoning Levels Compared

```
┌──────────┬───────────┬──────────┬─────────────────────────┐
│  Level   │   Steps   │  Speed   │       Use Case          │
├──────────┼───────────┼──────────┼─────────────────────────┤
│   LOW    │    1-2    │  ⚡⚡⚡    │ Quick pre-filtering     │
│          │           │          │ High-volume screening   │
├──────────┼───────────┼──────────┼─────────────────────────┤
│  MEDIUM  │    3-4    │   ⚡⚡     │ Standard moderation     │
│          │           │          │ Balanced accuracy/speed │
├──────────┼───────────┼──────────┼─────────────────────────┤
│   HIGH   │    5+     │    ⚡     │ Edge cases & appeals    │
│          │           │          │ Maximum accuracy        │
└──────────┴───────────┴──────────┴─────────────────────────┘

LOW REASONING:
  Step 1: Obvious violations ✓
  Step 2: Indicators ✓

MEDIUM REASONING:
  Step 1: Obvious violations ✓
  Step 2: Indicators ✓
  Step 3: Context analysis ✓ NEW

HIGH REASONING:
  Step 1: Obvious violations ✓
  Step 2: Indicators ✓
  Step 3: Context analysis ✓
  Step 4: Edge cases ✓ NEW
  Step 5: Example comparison ✓ NEW
```

## Policy Structure

```
POLICY (JSON)
├── id: "hate-speech"
├── name: "Hate Speech"
├── description: "Content that demeans..."
├── severity: "high"
├── indicators: [
│     "dehumanizing language",
│     "calls for violence",
│     "slurs or epithets",
│     ...
│   ]
├── examples_violating: [
│     "People of [group] are subhuman...",
│     ...
│   ]
└── examples_allowed: [
      "Historical discussion of...",
      ...
    ]

USED FOR:
  indicators → Step 2 (matching)
  examples_violating → Step 5 (comparison)
  examples_allowed → Step 4 (exceptions)
  description → Human context
```

## Multi-Policy Aggregation Logic

```
SCENARIO 1: All Safe
  Policy A: SAFE (0.1)
  Policy B: SAFE (0.2)
  Policy C: SAFE (0.15)
  ────────────────────
  Overall: SAFE (avg confidence of safety)

SCENARIO 2: One Unsafe
  Policy A: SAFE (0.1)
  Policy B: UNSAFE (0.85)  ← Determines overall
  Policy C: SAFE (0.2)
  ────────────────────
  Overall: UNSAFE (0.85)

SCENARIO 3: Multiple Unsafe
  Policy A: UNSAFE (0.9)
  Policy B: UNSAFE (0.85)  ← Average these
  Policy C: SAFE (0.2)
  ────────────────────
  Overall: UNSAFE (0.875)

SCENARIO 4: Mixed with Unclear
  Policy A: SAFE (0.1)
  Policy B: UNCLEAR (0.5)  ← Determines overall
  Policy C: SAFE (0.2)
  ────────────────────
  Overall: UNCLEAR (0.5)

KEY RULE: Conservative approach
  UNSAFE > UNCLEAR > SAFE
```

## Code Structure

```
safety_reasoner.py
├── Classification (enum)
│   ├── SAFE
│   ├── UNSAFE
│   └── UNCLEAR
│
├── ReasoningLevel (enum)
│   ├── LOW
│   ├── MEDIUM
│   └── HIGH
│
├── ReasoningStep (dataclass)
│   ├── step_number
│   ├── description
│   ├── finding
│   └── confidence_impact
│
├── PolicyEvaluation (dataclass)
│   ├── policy_id
│   ├── classification
│   ├── confidence
│   └── reasoning_chain
│
├── SafetyEvaluationResult (dataclass)
│   ├── overall_classification
│   ├── overall_confidence
│   └── policy_evaluations
│
└── SafetyReasoner (class)
    ├── __init__(policies_path)
    ├── evaluate(content, policies, level)
    ├── _evaluate_single_policy()
    ├── _reason_step_1_obvious_violations()
    ├── _reason_step_2_indicators()
    ├── _reason_step_3_context()
    ├── _reason_step_4_edge_cases()
    ├── _reason_step_5_examples()
    └── _aggregate_evaluations()
```

## Comparison to Production Systems

```
TOY IMPLEMENTATION (This Project)
┌────────────────────────────────────────┐
│  Simple keyword matching               │
│  ~500 lines Python                     │
│  Single language                       │
│  ~60-70% accuracy                      │
│  Educational only                      │
└────────────────────────────────────────┘

        vs

GPT-OSS-SAFEGUARD (Production)
┌────────────────────────────────────────┐
│  20B-120B parameter LLM                │
│  Semantic understanding                │
│  14+ languages                         │
│  ~80-85% F1 score                      │
│  Production deployment                 │
│  Handles adversarial attacks           │
│  Extensive training data               │
└────────────────────────────────────────┘

SAME CONCEPTS, DIFFERENT SCALE
  ✓ Policy-based reasoning
  ✓ Chain-of-thought transparency
  ✓ Multi-policy evaluation
  ✓ Confidence scoring
  ✓ Reasoning effort levels
```

## Extension Points

Want to improve this system? Here are the key areas:

1. **Better Matching (Step 1-2)**
   - Replace keywords with embeddings
   - Use semantic similarity
   - Add fuzzy matching

2. **Smarter Context (Step 3)**
   - Integrate real LLM
   - Analyze tone and intent
   - Handle sarcasm/irony

3. **Learning System**
   - Track false positives/negatives
   - Update confidence weights
   - A/B test reasoning strategies

4. **Multi-language**
   - Add translation layer
   - Language-specific policies
   - Cross-lingual embeddings

5. **Adversarial Defense**
   - Jailbreak detection
   - Obfuscation handling
   - Instruction hierarchy

Each of these would move you closer to production-grade!
