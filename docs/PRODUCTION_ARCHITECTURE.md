# Production-Grade Safety Pipeline Architecture

**Based on:** OpenAI gpt-oss-safeguard Technical Report
**Goal:** Training manual for building real-world content moderation systems
**Approach:** Layered "Defense Onion" using policy-aware reasoning

---

## 🎯 The Defense Onion Architecture

OpenAI's production system uses a **layered approach** where content passes through progressively more sophisticated (and expensive) checks:

```
User Content
    ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Fast Classifiers (Heuristics)                      │
│ - Latency: <1ms                                              │
│ - Cost: Near-zero                                            │
│ - Purpose: Block obvious violations instantly                │
│ - Blocks: ~10-40% of traffic (your data: 7.7%)              │
└─────────────────────────────────────────────────────────────┘
    ↓ (passes)
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Policy-Specific 20B Models (Synchronous)           │
│ - Latency: 1-3s                                              │
│ - Cost: Moderate                                             │
│ - Purpose: Real-time policy reasoning                        │
│ - Handles: 60-90% of remaining traffic                      │
│ - Multiple policies in parallel (gauntlet)                   │
└─────────────────────────────────────────────────────────────┘
    ↓ (uncertain/high-risk)
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: Judge/Reviewer (20B or 120B) (Asynchronous)        │
│ - Latency: 5-20s (background)                               │
│ - Cost: High                                                 │
│ - Purpose: Resolve conflicts, batch review                   │
│ - Handles: Edge cases, multi-policy conflicts                │
└─────────────────────────────────────────────────────────────┘
    ↓ (very uncertain/critical)
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4: Human Review (Batch)                               │
│ - Latency: Hours to days                                    │
│ - Cost: Very high                                            │
│ - Purpose: Appeals, training data, edge cases               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Layer 1: Fast Classifiers (Heuristic Filter)

### Purpose
Block **obvious violations** instantly with near-zero latency.

### Implementation
```python
class HeuristicFilter:
    """
    Layer 1: Fast pattern matching
    Based on your llm-abuse-patterns work: 80.8% precision, 26.1% recall
    """

    def __init__(self):
        # Load patterns from your pattern database
        self.patterns = {
            'hate_speech': ['subhuman', 'vermin', 'should be killed'],
            'violence': ['how to build a bomb', 'instructions to attack'],
            'self_harm': ['how to kill yourself', 'suicide methods'],
            'sexual_minors': ['sexual content', 'minor', 'child'],
            # ... more patterns
        }

    def check(self, content: str) -> dict:
        """
        Returns: {
            'blocked': bool,
            'policies_violated': list,
            'confidence': float,
            'latency_ms': float
        }
        """
        start = time.time()
        content_lower = content.lower()

        violated = []
        for policy, patterns in self.patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                violated.append(policy)

        latency = (time.time() - start) * 1000

        return {
            'blocked': len(violated) > 0,
            'policies_violated': violated,
            'confidence': 0.8 if violated else 0.95,  # High precision
            'latency_ms': latency
        }
```

### Performance (Your Data)
- **Latency:** <0.1ms
- **Precision:** 80.8% (few false positives)
- **Recall:** 26.1% (misses 73.9% of subtle cases)
- **Blocks:** 7.7% of traffic instantly

### When to Use
- ✅ First-layer filtering for ALL content
- ✅ Instant blocking of blatant violations
- ✅ Save compute for subtle cases

---

## 🧠 Layer 2: Policy-Specific Gauntlet (20B Models)

### Purpose
**Real-time policy reasoning** with multiple specialized models running in parallel.

### The "Gauntlet" Approach
Instead of one model checking all policies, run **multiple policy-specific models** in parallel:

```
Content → ┌─ Hate Speech Model (20B) ─┐
         ├─ Violence Model (20B) ──────┤
         ├─ Self-Harm Model (20B) ─────┤
         ├─ Sexual/Minors Model (20B) ─┤→ Aggregator → Decision
         ├─ Harassment Model (20B) ────┤
         └─ Illegal Acts Model (20B) ──┘
```

### Why Policy-Specific Models?

From the OpenAI report:
- **Multi-policy accuracy:** Only 43.6-46.3% when one model checks ALL policies
- **Specialized models:** Each focused on one domain = higher accuracy
- **Parallel execution:** Run simultaneously for same latency as single check

### Implementation
```python
class PolicyGauntlet:
    """
    Layer 2: Parallel policy-specific models
    Each model uses optimized 400-600 token policy
    """

    def __init__(self):
        # Load optimized policies (from your research)
        self.policies = {
            'hate_speech': load_policy('policies/optimized/hate_speech.txt'),
            'violence': load_policy('policies/optimized/violence.txt'),
            'self_harm': load_policy('policies/optimized/self_harm.txt'),
            'sexual_minors': load_policy('policies/optimized/sexual_minors.txt'),
            'harassment': load_policy('policies/optimized/harassment.txt'),
            'illegal': load_policy('policies/optimized/illegal.txt'),
        }

    async def check_parallel(self, content: str) -> dict:
        """
        Run all policy checks in parallel (async)
        Returns aggregate result from all models
        """
        # Create tasks for parallel execution
        tasks = []
        for policy_name, policy_text in self.policies.items():
            task = self.check_single_policy(content, policy_name, policy_text)
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        # Aggregate results
        return self.aggregate(results)

    async def check_single_policy(self, content, policy_name, policy_text):
        """
        Check content against one specific policy
        Uses /api/chat with optimized policy (400-600 tokens)
        """
        messages = [
            {"role": "system", "content": policy_text},
            {"role": "user", "content": f"Content to analyze: {content}"}
        ]

        response = await self.ollama_chat(
            model="gpt-oss:20b",  # Baseline, not safeguard!
            messages=messages
        )

        # Parse response (both thinking and content fields)
        classification = self.parse_response(response)

        return {
            'policy': policy_name,
            'violation': classification['violation'],
            'confidence': classification['confidence'],
            'reasoning': classification['thinking']
        }

    def aggregate(self, results: list) -> dict:
        """
        Aggregate results from all policy checks

        Decision logic:
        - Any HIGH confidence violation → BLOCK
        - Multiple MEDIUM confidence → ESCALATE to Judge
        - All SAFE → ALLOW
        """
        violations = [r for r in results if r['violation']]

        # High confidence violations
        high_conf = [v for v in violations if v['confidence'] > 0.85]
        if high_conf:
            return {
                'decision': 'BLOCK',
                'policies_violated': [v['policy'] for v in high_conf],
                'confidence': max(v['confidence'] for v in high_conf),
                'escalate': False
            }

        # Medium confidence violations - need Judge
        medium_conf = [v for v in violations if 0.5 < v['confidence'] <= 0.85]
        if len(medium_conf) >= 2:  # Multiple uncertain violations
            return {
                'decision': 'ESCALATE',
                'policies_violated': [v['policy'] for v in medium_conf],
                'confidence': 0.65,
                'escalate': True,
                'reason': 'Multiple medium-confidence violations'
            }

        # Single uncertain violation
        if medium_conf:
            return {
                'decision': 'ESCALATE',
                'policies_violated': [medium_conf[0]['policy']],
                'confidence': medium_conf[0]['confidence'],
                'escalate': True,
                'reason': 'Uncertain violation'
            }

        # All safe
        return {
            'decision': 'ALLOW',
            'policies_violated': [],
            'confidence': 0.95,
            'escalate': False
        }
```

### Performance Expectations
- **Latency:** 1.5-2.5s (parallel execution)
- **Accuracy:** 65-75% per policy (your baseline findings)
- **Multi-policy:** 50-60% (better than single model checking all)
- **Cost:** 6 API calls per content (parallel = same latency)

### Synchronous vs Asynchronous

**Synchronous (Real-time):**
```python
# User waits for result
content = user_input()
result = await policy_gauntlet.check_parallel(content)

if result['decision'] == 'BLOCK':
    return "Content violates policy"
elif result['decision'] == 'ESCALATE':
    # Queue for async review, allow content temporarily
    queue_for_review(content, result)
    return "Content posted (under review)"
else:
    return "Content posted"
```

**Asynchronous (Background):**
```python
# For non-critical content (e.g., DMs, comments)
content = user_input()
post_content_immediately(content)  # Low latency!

# Check in background
asyncio.create_task(
    review_content_async(content)
)

# If violation found later, remove and notify
```

---

## ⚖️ Layer 3: Judge/Reviewer (Conflict Resolution)

### Purpose
Resolve **multi-policy conflicts** and **uncertain cases** from Layer 2.

### When to Escalate to Judge

From Layer 2 gauntlet:
1. **Multiple violations** with conflicting signals
2. **Medium confidence** (0.5-0.85) violations
3. **Edge cases** that need deeper reasoning
4. **User appeals** of automated decisions

### Implementation Options

**Option A: Single Judge Model (120B)**
```python
class JudgeReviewer:
    """
    Layer 3: Use larger 120B model for complex cases
    Higher accuracy, slower, more expensive
    """

    async def review(self, content: str, gauntlet_results: dict) -> dict:
        """
        Review content with access to gauntlet reasoning
        """
        # Build context from Layer 2 results
        context = self.build_context(gauntlet_results)

        # Combined policy for holistic review
        combined_policy = self.get_combined_policy()

        messages = [
            {"role": "system", "content": combined_policy},
            {"role": "user", "content": f"""
Previous analysis found potential violations:
{context}

Please review this content with full context:
{content}

Provide final decision with confidence score.
            """}
        ]

        response = await self.ollama_chat(
            model="gpt-oss:120b",  # Larger model!
            messages=messages
        )

        return self.parse_final_decision(response)
```

**Option B: Committee of Specialized Judges**
```python
class JudgeCommittee:
    """
    Layer 3: Multiple specialized judge models vote

    Each judge specializes in one policy domain:
    - Hate speech expert
    - Violence expert
    - Self-harm expert
    etc.
    """

    async def review(self, content: str, flagged_policies: list) -> dict:
        """
        Only invoke judges for policies flagged by gauntlet
        """
        judge_tasks = []

        for policy in flagged_policies:
            # Use domain expert judge
            task = self.get_specialist_judge(policy).review(content)
            judge_tasks.append(task)

        # Collect votes
        votes = await asyncio.gather(*judge_tasks)

        # Majority vote wins
        return self.majority_vote(votes)
```

### Performance
- **Latency:** 5-20s (acceptable for async)
- **Accuracy:** 75-85% (120B performance)
- **Cost:** High (only for ~5-15% of content)

---

## 👤 Layer 4: Human Review (Final Safety Net)

### Purpose
Handle cases that automated systems can't resolve confidently.

### When to Escalate to Humans

1. **Very uncertain** (confidence < 0.5 from Judge)
2. **User appeals** of automated decisions
3. **New attack patterns** not in training data
4. **High-stakes content** (e.g., from verified users)
5. **Training data collection** for model improvement

### Implementation
```python
class HumanReviewQueue:
    """
    Layer 4: Queue for human moderators
    """

    def add_to_queue(self, content: str, context: dict):
        """
        Add to human review queue with full context
        """
        review_task = {
            'content': content,
            'automated_decision': context['decision'],
            'confidence': context['confidence'],
            'policies_flagged': context['policies_violated'],
            'layer1_result': context['heuristic_result'],
            'layer2_result': context['gauntlet_result'],
            'layer3_result': context['judge_result'],
            'reasoning_chains': context['all_reasoning'],
            'priority': self.calculate_priority(context),
            'created_at': datetime.now()
        }

        # Add to queue (Redis, database, etc.)
        self.queue.add(review_task)

        # Alert moderators if high priority
        if review_task['priority'] == 'CRITICAL':
            self.alert_moderators(review_task)
```

---

## 🔄 Complete Production Pipeline

### Full Flow Implementation

```python
class SafetyPipeline:
    """
    Complete production safety pipeline
    Implements all 4 layers
    """

    def __init__(self):
        self.layer1 = HeuristicFilter()
        self.layer2 = PolicyGauntlet()
        self.layer3 = JudgeReviewer()
        self.layer4 = HumanReviewQueue()

        self.metrics = MetricsCollector()

    async def check_content(
        self,
        content: str,
        sync_mode: bool = True
    ) -> dict:
        """
        Check content through layered pipeline

        Args:
            content: User-generated content
            sync_mode: True = wait for result, False = background check

        Returns:
            Decision with full context
        """
        start_time = time.time()

        # LAYER 1: Heuristic Filter (always synchronous)
        layer1_result = self.layer1.check(content)

        if layer1_result['blocked']:
            # Instant block for obvious violations
            self.metrics.record('layer1_block', layer1_result)
            return {
                'decision': 'BLOCK',
                'layer': 1,
                'policies_violated': layer1_result['policies_violated'],
                'confidence': layer1_result['confidence'],
                'latency_ms': layer1_result['latency_ms']
            }

        # LAYER 2: Policy Gauntlet
        if sync_mode:
            # Real-time check (wait for result)
            layer2_result = await self.layer2.check_parallel(content)
        else:
            # Background check (return immediately)
            asyncio.create_task(
                self.background_check(content, layer1_result)
            )
            return {
                'decision': 'ALLOW_PENDING',
                'layer': 2,
                'status': 'queued_for_review'
            }

        if layer2_result['decision'] == 'BLOCK':
            # High confidence violation
            self.metrics.record('layer2_block', layer2_result)
            return {
                'decision': 'BLOCK',
                'layer': 2,
                'policies_violated': layer2_result['policies_violated'],
                'confidence': layer2_result['confidence'],
                'latency_ms': (time.time() - start_time) * 1000
            }

        if layer2_result['decision'] == 'ALLOW':
            # All policies agree it's safe
            self.metrics.record('layer2_allow', layer2_result)
            return {
                'decision': 'ALLOW',
                'layer': 2,
                'confidence': layer2_result['confidence'],
                'latency_ms': (time.time() - start_time) * 1000
            }

        # LAYER 3: Judge Review (uncertain cases)
        # Always asynchronous - allow content, review in background
        asyncio.create_task(
            self.judge_review_async(content, layer1_result, layer2_result)
        )

        return {
            'decision': 'ALLOW_PENDING',
            'layer': 3,
            'status': 'escalated_to_judge',
            'policies_uncertain': layer2_result['policies_violated'],
            'latency_ms': (time.time() - start_time) * 1000
        }

    async def judge_review_async(self, content, layer1, layer2):
        """
        Asynchronous judge review
        """
        layer3_result = await self.layer3.review(content, layer2)

        if layer3_result['decision'] == 'BLOCK':
            # Remove content and notify user
            await self.remove_content(content)
            self.metrics.record('layer3_block', layer3_result)

        elif layer3_result['confidence'] < 0.5:
            # LAYER 4: Queue for human review
            self.layer4.add_to_queue(content, {
                'layer1': layer1,
                'layer2': layer2,
                'layer3': layer3_result
            })
            self.metrics.record('layer4_queue', layer3_result)

        else:
            # Judge says it's safe
            self.metrics.record('layer3_allow', layer3_result)
```

---

## 📊 Performance Characteristics

### Latency by Layer

| Layer | Latency | Cost | Handles | Accuracy |
|-------|---------|------|---------|----------|
| 1. Heuristic | <1ms | $0.000001 | 10-40% blocked | 80% precision, 26% recall |
| 2. Gauntlet (20B) | 1.5-2.5s | $0.01 | 60-90% decided | 65-75% per policy |
| 3. Judge (120B) | 5-20s | $0.10 | 5-15% reviewed | 75-85% accuracy |
| 4. Human | Hours | $1-10 | <1% escalated | 95%+ accuracy |

### Cost Optimization

**Example: 1M content items/day**

Without layering (all through 120B):
- Cost: 1M × $0.10 = $100,000/day

With layering:
- Layer 1 blocks: 200K × $0.000001 = $0.20
- Layer 2 checks: 800K × $0.01 = $8,000
- Layer 3 reviews: 100K × $0.10 = $10,000
- Layer 4 human: 10K × $5 = $50,000
- **Total: $68,000/day (32% savings)**

---

## 🎓 Sunday Experiments: Building the Pipeline

### Experiment 1: Measure Layer Effectiveness
```python
# Test what % of traffic each layer catches
results = {
    'layer1_blocked': 0,
    'layer2_blocked': 0,
    'layer3_blocked': 0,
    'layer4_needed': 0
}

for content in test_set:
    result = await pipeline.check_content(content)
    results[f"layer{result['layer']}_blocked"] += 1

# Understand the distribution
print(f"Layer 1 catches: {results['layer1_blocked']/len(test_set)*100}%")
print(f"Layer 2 catches: {results['layer2_blocked']/len(test_set)*100}%")
# etc.
```

### Experiment 2: Gauntlet vs Single Model
```python
# Compare parallel policy-specific vs single multi-policy model

# Approach A: Gauntlet (6 specialized models)
gauntlet_result = await policy_gauntlet.check_parallel(content)

# Approach B: Single model (all policies)
single_result = await single_model.check_all_policies(content)

# Measure:
# - Multi-policy exact match accuracy
# - Latency (should be similar if parallel)
# - Which approach catches more violations?
```

### Experiment 3: Optimal Escalation Thresholds
```python
# Find best confidence thresholds for escalation

thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

for threshold in thresholds:
    pipeline.layer2.escalation_threshold = threshold

    results = evaluate_pipeline(test_set)

    print(f"Threshold {threshold}:")
    print(f"  Layer 2 accuracy: {results['layer2_accuracy']}")
    print(f"  Escalation rate: {results['escalation_rate']}")
    print(f"  Final accuracy: {results['final_accuracy']}")
```

---

## 🚀 Production Deployment Checklist

### Infrastructure
- [ ] Ollama cluster for 20B models (Layer 2 gauntlet)
- [ ] Separate 120B instance for judge (Layer 3)
- [ ] Redis queue for async processing
- [ ] PostgreSQL for logging and metrics
- [ ] Monitoring dashboard (Grafana)

### Code
- [ ] Heuristic filter with pattern database
- [ ] Policy gauntlet with 6 specialized models
- [ ] Judge reviewer with conflict resolution
- [ ] Human review queue system
- [ ] Metrics and observability

### Policies
- [ ] Optimized 400-600 token policies (one per domain)
- [ ] Combined policy for judge
- [ ] Human moderator guidelines
- [ ] Appeal handling procedures

### Testing
- [ ] Layer-by-layer accuracy benchmarks
- [ ] Load testing (can handle traffic?)
- [ ] Cost modeling (actual spend vs budget)
- [ ] Latency SLAs (p50, p95, p99)

---

This is your **training manual** for building production-grade safety systems using the new policy-aware reasoning technology from OpenAI!

Want me to create the experiment code for Sunday to test each layer?
