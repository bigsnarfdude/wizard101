# What IS a Safety Reasoner? (From First Principles)

## 🤔 The Core Problem

**Traditional content moderation:**
```
User Content → [Black Box ML Model] → SAFE or UNSAFE
                      ↓
              "Why? ¯\_(ツ)_/¯"
```

**Problems:**
1. **Not adaptable** - Want new rule? Retrain entire model (expensive, slow)
2. **Not transparent** - Model says "unsafe" but can't explain why
3. **Not controllable** - Can't adjust for different contexts (school vs military base)
4. **Not trustable** - Makes mistakes, can't audit reasoning

---

## 💡 The Safety Reasoner Solution

**New paradigm:**
```
User Content + Written Policy → [Reasoning Model] → SAFE/UNSAFE + Explanation
                                        ↓
                          Chain-of-Thought Reasoning:
                          "1. Check for hate speech..."
                          "2. Analyze context..."
                          "3. Conclusion: Violates R2.a"
```

**Key Innovation:** The model READS and REASONS about the policy, rather than having rules baked into weights.

---

## 🔧 What IS the Technology?

### Traditional Approach (Pre-2024)
```python
# Trained classifier - rules are IMPLICIT in weights
model = train_classifier(
    data=[(content1, "toxic"), (content2, "safe"), ...],
    labels=["toxic", "safe"]
)

# Black box decision
prediction = model.predict("Some content")
# Output: "toxic" (but WHY?)
```

### Safety Reasoner Approach (NEW!)
```python
# Policy-aware reasoner - rules are EXPLICIT
policy = """
# Hate Speech Policy

## Violations:
- R1: Dehumanizing language (comparing people to animals)
- R2: Calls for violence against groups
- R3: Slurs targeting protected characteristics

## Safe:
- Political disagreement
- Historical discussion
"""

# Model READS the policy and REASONS
result = safety_reasoner.evaluate(
    content="Some content",
    policy=policy  # ← Policy is an INPUT, not baked in!
)

# Output: {
#   "classification": "UNSAFE",
#   "violated_rules": ["R1"],
#   "reasoning": "Content contains dehumanizing language comparing [group] to vermin, which violates R1",
#   "confidence": 0.92
# }
```

---

## 🎯 The Three Key Capabilities

### 1. Policy-Based Reasoning (The Core Innovation)

**What it means:**
- Policies are **written documents**, not trained weights
- Model reads policy like a human reads instructions
- Can change policy WITHOUT retraining

**Example:**
```python
# Day 1: Use corporate policy
corporate_policy = "Ban profanity and sexual content"
result1 = reasoner.evaluate(content, corporate_policy)

# Day 2: Switch to academic policy (SAME MODEL!)
academic_policy = "Allow profanity in quotes, allow sexual health education"
result2 = reasoner.evaluate(content, academic_policy)

# SAME content, DIFFERENT policies, NO RETRAINING!
```

**Why this matters:**
- ✅ Adapt to different contexts (schools, hospitals, forums)
- ✅ Update rules instantly (no 6-week retraining cycle)
- ✅ A/B test different policies
- ✅ Comply with changing regulations

---

### 2. Chain-of-Thought Transparency

**What it means:**
- Model shows its reasoning step-by-step
- Humans can validate each step
- Catch reasoning errors

**Example:**
```python
result = reasoner.evaluate(
    content="This research paper analyzes hate speech patterns online",
    policy=hate_speech_policy
)

# Output includes reasoning chain:
{
  "classification": "SAFE",
  "reasoning_steps": [
    {
      "step": 1,
      "description": "Check for dehumanizing language",
      "finding": "No dehumanizing terms found",
      "confidence_impact": -0.1
    },
    {
      "step": 2,
      "description": "Analyze context and intent",
      "finding": "Educational/research context detected with words 'research', 'analyzes'",
      "confidence_impact": -0.3
    },
    {
      "step": 3,
      "description": "Compare to allowed exceptions",
      "finding": "Matches exception: 'Academic analysis of hate speech'",
      "confidence_impact": -0.2
    }
  ],
  "final_confidence": 0.85,
  "conclusion": "SAFE - Academic research context"
}
```

**Why this matters:**
- ✅ Audit decisions (regulatory compliance)
- ✅ Improve policies (see where model struggles)
- ✅ Trust decisions (understand the "why")
- ✅ Debug errors (find where reasoning went wrong)

---

### 3. Multi-Policy Classification

**What it means:**
- Check content against MULTIPLE policies at once
- Identify ALL violations, not just one
- Handle complex real-world scenarios

**Example:**
```python
result = reasoner.evaluate(
    content="Here's how to make illegal drugs and sell them to kids at school",
    policies=[
        hate_speech_policy,
        violence_policy,
        illegal_activity_policy,
        sexual_minors_policy,
        self_harm_policy,
        harassment_policy
    ]
)

# Output identifies ALL violations:
{
  "classification": "UNSAFE",
  "policies_violated": [
    {
      "policy": "illegal_activity_policy",
      "rules_violated": ["R1: Drug manufacturing instructions"],
      "confidence": 0.95
    },
    {
      "policy": "sexual_minors_policy",
      "rules_violated": ["R4: Targeting minors for illegal activity"],
      "confidence": 0.88
    }
  ],
  "severity": "CRITICAL",
  "recommended_action": "Block and report to authorities"
}
```

**Why this matters:**
- ✅ Real content often violates multiple rules
- ✅ Different teams care about different violations (legal, HR, safety)
- ✅ Prioritize by severity (hate speech + violence = escalate)

---

## 🏗️ How Does It Actually Work?

### The Technical Implementation

**Step 1: Encode the Policy**
```python
# Policy becomes part of the prompt (system message)
system_prompt = f"""
You are a content safety analyzer.

{policy_text}

Analyze content against this policy and explain your reasoning.
"""
```

**Step 2: Present Content**
```python
# Content in user message
user_message = f"Content to analyze: {user_content}"
```

**Step 3: Model Reasons (using LLM)**
```python
# LLM reads policy, reads content, reasons about violation
response = llm.chat(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
)

# Model outputs:
# - Classification (SAFE/UNSAFE)
# - Which rules violated
# - Step-by-step reasoning
# - Confidence score
```

**The Magic:** The LLM's language understanding allows it to:
- Read and comprehend written policies (like a human would)
- Apply policies to new content it's never seen
- Explain its reasoning in natural language
- Handle ambiguity and edge cases

---

## 🆚 Comparison: Traditional vs Safety Reasoner

### Scenario: Ban "hate speech"

#### Traditional Classifier
```python
# Training phase (weeks/months)
train_data = [
    ("People of [group] are vermin", "hate_speech"),
    ("I disagree with their views", "safe"),
    # ... 100,000 more examples
]
model = train_classifier(train_data)

# Production
result = model.predict("New content")
# Output: "hate_speech"
# Why? ¯\_(ツ)_/¯

# Update policy?
# → Collect 10,000 new training examples
# → Retrain model (2-6 weeks)
# → Deploy new version
# → Hope it works
```

#### Safety Reasoner
```python
# No training phase needed!

# Define policy (minutes)
policy = """
# Hate Speech Policy
Violations:
- Dehumanizing language
- Calls for violence
- Slurs

Safe:
- Political disagreement
- Historical discussion
"""

# Production
result = reasoner.evaluate("New content", policy)
# Output: {
#   "classification": "hate_speech",
#   "rule": "Dehumanizing language",
#   "reasoning": "Compares [group] to vermin",
#   "confidence": 0.92
# }

# Update policy?
# → Edit policy text (5 minutes)
# → Deploy (instant)
# → Done!
```

---

## 📊 When to Use Each Approach

### Use Traditional Classifier When:
- ✅ Policy is **stable** (won't change often)
- ✅ Have **lots of training data** (100K+ examples)
- ✅ Need **maximum speed** (<10ms)
- ✅ Simple **binary classification** (toxic/safe)
- ✅ **Cost-sensitive** (cheaper inference)

**Examples:**
- Spam detection (gmail has billions of examples)
- NSFW image detection (stable policy)
- Profanity filter (simple rules)

### Use Safety Reasoner When:
- ✅ Policy **changes frequently** (new rules, contexts)
- ✅ **Limited training data** (new policy, rare violations)
- ✅ Need **explainability** (regulatory, trust, debugging)
- ✅ **Multi-policy** classification (many rules to check)
- ✅ **Context matters** (education vs abuse)
- ✅ **Customizable per customer** (SaaS, enterprise)

**Examples:**
- Corporate compliance (policies change per regulation)
- Community moderation (rules evolve with community)
- Content safety (context-dependent)
- Medical/legal screening (must explain decisions)

---

## 🔬 The wizard101 Project: Understanding Through Building

**What wizard101 teaches:**

### Part 1: The Toy (Understanding Concepts)
```python
# toy-safety-reasoner/
# Shows the CONCEPTS in simple code:
# - How policies work
# - How reasoning chains work
# - How multi-policy classification works
```

**Goal:** Understand the ideas with keyword matching (no LLM needed)

### Part 2: The Architecture (Production Patterns)
```python
# PRODUCTION_ARCHITECTURE.md
# Shows how OpenAI uses this in production:
# - Layer 1: Heuristics (fast filter)
# - Layer 2: Policy reasoner (real-time)
# - Layer 3: Judge (async review)
# - Layer 4: Human (appeals)
```

**Goal:** Understand how to DEPLOY this at scale

### Part 3: The Experiments (Sunday!)
```python
# experiments/sunday-pipeline/
# Test the architecture components:
# - Which layers catch what?
# - Parallel vs sequential policies?
# - Optimal escalation thresholds?
```

**Goal:** Learn the TRADE-OFFS for production

---

## 💡 The Big Picture: A New Defense Layer

### Before Safety Reasoners
```
Internet → Firewall → Rate Limiter → [Black Box Filter] → Users
                                            ↓
                                    Hope it works! 🤞
```

### After Safety Reasoners
```
Internet → Firewall → Rate Limiter → Defense Onion → Users
                                           ↓
                     ┌──────────────────────────────────┐
                     │ Layer 1: Heuristics (<1ms)      │
                     │ Layer 2: Policy Reasoner (1-3s) │
                     │ Layer 3: Judge Review (5-20s)   │
                     │ Layer 4: Human (hours)          │
                     └──────────────────────────────────┘
                                  ↓
                          Policy-aware ✅
                          Explainable ✅
                          Adaptable ✅
                          Multi-policy ✅
```

---

## 🎓 Simple Analogy

**Traditional Classifier = Security Guard with Training**
- Guard was trained: "Block anyone matching these descriptions"
- Fast reactions (pattern recognition)
- Can't explain decisions ("They look suspicious")
- Can't adapt to new rules without retraining
- Good for obvious cases

**Safety Reasoner = Security Guard with Handbook**
- Guard reads handbook: "Here are the rules and procedures"
- Slower but thoughtful (reads, reasons, decides)
- Can explain: "Violated section 3.2.a of the handbook"
- Update handbook anytime (no retraining)
- Handles edge cases and context

**Production System = Team of Guards**
- First guard (heuristic): "Obviously dangerous? Block instantly"
- Second team (reasoners): "Read handbook, discuss, decide"
- Supervisor (judge): "Resolve disagreements between guards"
- Manager (human): "Handle truly weird situations"

---

## 🔑 Key Takeaways

1. **Safety Reasoner = Policy-Aware LLM**
   - Reads written policies (not trained on them)
   - Reasons about content vs policy
   - Explains decisions transparently

2. **The Three Superpowers:**
   - Policy-based (adaptable without retraining)
   - Transparent (chain-of-thought reasoning)
   - Multi-policy (checks many rules at once)

3. **Not a Replacement, a New Layer:**
   - Use WITH traditional classifiers
   - Heuristics → Reasoner → Judge → Human
   - Each layer has different speed/accuracy trade-off

4. **wizard101 = Training Manual:**
   - Understand concepts (toy implementation)
   - Understand architecture (production patterns)
   - Understand trade-offs (experiments)
   - Build your own (apply to your domain)

---

## 🚀 What Sunday Experiments Will Teach

**Question 1:** How much traffic does each layer catch?
**Answer:** Optimize your pipeline (where to invest compute)

**Question 2:** Parallel policies vs single combined check?
**Answer:** Architecture decision (gauntlet vs monolith)

**Question 3:** When to escalate to judge?
**Answer:** Threshold tuning (cost vs accuracy)

**Question 4:** Do 400-600 token policies work universally?
**Answer:** Policy optimization (your llm-abuse-patterns findings → safety domain)

**Question 5:** How to aggregate multi-policy results?
**Answer:** Conflict resolution strategy (what if hate=safe but violence=unsafe?)

---

Does this clarify what safety reasoners are and why we're building wizard101 as a training manual?
