# Sunday Experiments: Building the Defense Onion

**Goal:** Test each layer of the production pipeline architecture
**Server:** remote-server with gpt-oss:20b
**Time:** 2-3 hours on Sunday PST

---

## 🎯 What We're Building

A **production-grade layered defense pipeline** using the OpenAI architecture:

```
Layer 1 (Heuristic) → Layer 2 (Policy Gauntlet) → Layer 3 (Judge) → Layer 4 (Human)
     <1ms                 1-3s (parallel)           5-20s (async)      hours
```

**Sunday Focus:** Test and measure Layers 1-3 (skip Layer 4 human review)

---

## 🔬 Experiment 1: Layer Distribution Analysis (45 min)

### Goal
Understand what % of traffic each layer catches

### Setup
```python
# Test same 100 examples through complete pipeline
# Measure which layer makes final decision

test_set = load_test_data()  # 100 examples
results = {
    'layer1_blocked': 0,  # Heuristic catches
    'layer2_blocked': 0,  # Gauntlet catches
    'layer2_allowed': 0,  # Gauntlet says safe
    'layer3_needed': 0,   # Escalated to judge
}

for content in test_set:
    # Layer 1: Heuristic
    if heuristic.is_obvious_violation(content):
        results['layer1_blocked'] += 1
        continue

    # Layer 2: Policy gauntlet
    gauntlet_result = check_gauntlet(content)

    if gauntlet_result['confidence'] > 0.85 and gauntlet_result['violation']:
        results['layer2_blocked'] += 1
    elif gauntlet_result['confidence'] > 0.85 and not gauntlet_result['violation']:
        results['layer2_allowed'] += 1
    else:
        results['layer3_needed'] += 1
```

### Metrics
- **Layer 1 catch rate:** X% blocked instantly
- **Layer 2 decision rate:** Y% decided with confidence
- **Layer 3 escalation rate:** Z% need judge review

### Expected (Based on Your Data)
- Layer 1: 7-10% (your heuristic blocks 7.7%)
- Layer 2: 70-80% decided confidently
- Layer 3: 10-20% need escalation

**Insight:** Optimize thresholds to minimize expensive Layer 3 calls

---

## 🔬 Experiment 2: Gauntlet Architecture Test (60 min)

### Goal
Compare **3 approaches** to multi-policy classification

### Approach A: Sequential (Current Toy)
```python
# Check each policy one by one
violations = []
for policy in policies:
    result = check_policy(content, policy)
    if result['violation']:
        violations.append(policy)

# Issue: 6 sequential API calls = 9-15s latency!
```

### Approach B: Parallel Gauntlet (Production)
```python
# Check all policies in parallel
tasks = [
    check_policy_async(content, 'hate_speech'),
    check_policy_async(content, 'violence'),
    check_policy_async(content, 'self_harm'),
    check_policy_async(content, 'sexual_minors'),
    check_policy_async(content, 'harassment'),
    check_policy_async(content, 'illegal'),
]

results = await asyncio.gather(*tasks)
# Latency: ~1.5-2.5s (same as single call!)
```

### Approach C: Single Combined Policy
```python
# One model checks all policies at once
combined_policy = """
Check if content violates ANY of:
1. Hate speech
2. Violence
3. Self-harm
4. Sexual/minors
5. Harassment
6. Illegal activities

Return JSON: {"violations": ["policy1", "policy2"]}
"""

result = check_policy(content, combined_policy)
# Latency: ~1.5-2.5s
# Issue: Multi-policy accuracy only 43-46%!
```

### Test on Same 50 Examples
Measure:
- **Latency:** Sequential vs Parallel vs Combined
- **Accuracy:** Multi-policy exact match %
- **Cost:** Number of API calls

### Expected Results
| Approach | Latency | Accuracy | Cost/Content |
|----------|---------|----------|--------------|
| Sequential | 9-15s | 50-60% | 6× |
| **Parallel** | 1.5-2.5s | 55-65% | 6× |
| Combined | 1.5-2.5s | 40-50% | 1× |

**Winner:** Parallel gauntlet (same speed as combined, better accuracy)

---

## 🔬 Experiment 3: Escalation Threshold Optimization (45 min)

### Goal
Find optimal confidence threshold for Layer 2 → Layer 3 escalation

### Setup
```python
# Test different confidence thresholds
thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

for threshold in thresholds:
    layer2_accurate = 0
    layer3_needed = 0
    final_accurate = 0

    for content in test_set:
        layer2_result = gauntlet.check(content)

        if layer2_result['confidence'] >= threshold:
            # Trust Layer 2 decision
            if layer2_result['correct']:
                layer2_accurate += 1
                final_accurate += 1
        else:
            # Escalate to Layer 3
            layer3_needed += 1
            layer3_result = judge.review(content, layer2_result)
            if layer3_result['correct']:
                final_accurate += 1

    print(f"Threshold {threshold}:")
    print(f"  Layer 2 accuracy: {layer2_accurate/len(test_set)}")
    print(f"  Escalation rate: {layer3_needed/len(test_set)}")
    print(f"  Final accuracy: {final_accurate/len(test_set)}")
    print(f"  Cost factor: {1 + (layer3_needed/len(test_set)) * 10}")  # Judge is 10× more expensive
```

### Find Sweet Spot
Balance:
- **Too low threshold (0.5):** Escalate everything → expensive, slow
- **Too high threshold (0.9):** Trust Layer 2 too much → miss violations
- **Sweet spot (0.75-0.85?):** Best accuracy/cost trade-off

### Graph Results
```
Accuracy
   ↑
100|                           *Judge always
 95|                      *
 90|                 *
 85|            *        ← Sweet spot?
 80|       *
 75|  *                      ← Layer 2 only
   |________________________
      50  60  70  75  80  85  90
                Threshold →

Cost increases →
```

---

## 🔬 Experiment 4: Judge Specialization Test (30 min)

### Goal
Single general judge vs specialized judges?

### Approach A: Single General Judge (120B)
```python
# One 120B model reviews all escalated cases
judge_120b.review(content, context={
    'flagged_policies': ['hate_speech', 'harassment'],
    'layer2_confidence': 0.65,
    'layer2_reasoning': "..."
})

# Pros: Simple architecture
# Cons: 120B is expensive, slow
```

### Approach B: Specialized Policy Judges (20B)
```python
# Different 20B judges for each policy domain
if 'hate_speech' in flagged_policies:
    hate_judge_20b.review(content, layer2_reasoning)

if 'violence' in flagged_policies:
    violence_judge_20b.review(content, layer2_reasoning)

# Pros: Cheaper (20B), can run parallel
# Cons: More complex, need multiple models
```

### Test
- 30 escalated cases (uncertain from Layer 2)
- Compare accuracy and cost

### Expected
- **120B judge:** 80-85% accuracy, high cost
- **20B specialized:** 75-80% accuracy, lower cost
- **Trade-off:** Accuracy vs cost

---

## 🔬 Experiment 5: Policy Optimization for Gauntlet (30 min)

### Goal
Test if 400-600 token policies work for ALL policy types

### Setup
```python
# Create 3 versions of each policy:
policies_to_test = {
    'hate_speech': {
        'verbose': 850 tokens,
        'optimized': 520 tokens,  # OpenAI recommendation
        'minimal': 280 tokens
    },
    'violence': {
        'verbose': 780 tokens,
        'optimized': 480 tokens,
        'minimal': 250 tokens
    },
    # ... etc for all 6 policies
}

# Test same 20 examples per policy
for policy_type in policies_to_test:
    for version in ['verbose', 'optimized', 'minimal']:
        policy_text = policies_to_test[policy_type][version]
        results = test_policy(policy_text, test_cases)

        print(f"{policy_type} - {version}:")
        print(f"  Precision: {results['precision']}")
        print(f"  Recall: {results['recall']}")
        print(f"  F1: {results['f1']}")
        print(f"  Avg latency: {results['latency']}")
```

### Expected (Based on Your ToxicChat Work)
- **Verbose (800+):** High recall, slower, may confuse model
- **Optimized (400-600):** Best balance (your finding!)
- **Minimal (<300):** Fast but low recall

### Validate OpenAI's Recommendation
Does 400-600 tokens work for:
- ✅ Toxicity (your data: yes!)
- ✅ Jailbreaks (your data: yes!)
- ❓ Safety policies (test on Sunday)

---

## 📊 Expected Sunday Results

### Experiment 1: Layer Distribution
```
Layer 1 (Heuristic):    10% blocked
Layer 2 (Gauntlet):     75% decided
Layer 3 (Judge):        15% escalated
```

### Experiment 2: Gauntlet Architecture
```
Winner: Parallel Gauntlet
- Latency: 1.8s (vs 12s sequential, 1.9s combined)
- Accuracy: 58% (vs 52% sequential, 44% combined)
- Cost: 6× (same as sequential, but worth it!)
```

### Experiment 3: Optimal Threshold
```
Sweet Spot: 0.80 confidence
- Layer 2 catches: 70% with 72% accuracy
- Escalate: 30% to judge
- Final accuracy: 78%
- Cost: 4× baseline (acceptable)
```

### Experiment 4: Judge Specialization
```
Use 20B specialized judges:
- Cheaper than 120B
- Accuracy only 3-5% lower
- Can run parallel for multi-policy cases
```

### Experiment 5: Policy Optimization
```
400-600 tokens WORKS for all domains:
- Hate speech: 520 tokens optimal
- Violence: 480 tokens optimal
- Self-harm: 510 tokens optimal
- Avg F1 improvement: +4-6% vs verbose
```

---

## 🛠️ Code to Prepare Before Sunday

### File Structure
```bash
wizard101/experiments/sunday-pipeline/
├── 01_layer_distribution.py       # Experiment 1
├── 02_gauntlet_comparison.py      # Experiment 2
├── 03_threshold_optimization.py   # Experiment 3
├── 04_judge_specialization.py     # Experiment 4
├── 05_policy_optimization.py      # Experiment 5
│
├── policies/
│   ├── hate_speech_optimized.txt
│   ├── violence_optimized.txt
│   ├── self_harm_optimized.txt
│   ├── sexual_minors_optimized.txt
│   ├── harassment_optimized.txt
│   └── illegal_optimized.txt
│
├── test_data/
│   ├── layer_test_100.json        # 100 examples for distribution
│   ├── gauntlet_test_50.json      # 50 multi-policy examples
│   └── judge_test_30.json         # 30 escalated cases
│
└── utils/
    ├── ollama_client.py            # /api/chat wrapper
    ├── heuristic_filter.py         # Layer 1
    └── metrics.py                  # Logging and analysis
```

### Core Functions Needed
```python
# From your llm-abuse-patterns work:
async def check_policy_async(content, policy_text, model="gpt-oss:20b"):
    """
    Single policy check using /api/chat
    Returns: {violation, confidence, thinking}
    """
    # Your existing code from toxicchat experiments

async def check_gauntlet_parallel(content, policies):
    """
    Run all policies in parallel
    Returns: aggregated result
    """
    tasks = [check_policy_async(content, p) for p in policies]
    return await asyncio.gather(*tasks)

def heuristic_check(content):
    """
    Layer 1: Pattern matching
    From your pattern database
    """
    # Use your 01_safeguard_pattern_detector.py
```

---

## 🎓 What You'll Learn Sunday

### Production Architecture Insights
1. **Layer effectiveness:** What % each layer catches
2. **Optimal thresholds:** When to escalate vs decide
3. **Gauntlet design:** Parallel specialized vs single combined
4. **Judge strategy:** General 120B vs specialized 20B
5. **Policy optimization:** 400-600 tokens for all domains?

### Real-World Trade-offs
- **Accuracy vs Latency:** Can we get 80% accuracy in <2s?
- **Accuracy vs Cost:** Is 6× gauntlet cost worth accuracy gain?
- **Precision vs Recall:** Where to set thresholds?

### wizard101 as Training Manual
After Sunday, wizard101 becomes a **complete guide**:
1. ✅ Toy implementation (understand concepts)
2. ✅ Architecture guide (understand layers)
3. ✅ Experiment results (understand trade-offs)
4. ✅ Production patterns (build real systems)

---

Ready to build the experiment code?
