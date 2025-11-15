# Quick Start Guide

Get started with the Toy Safety Reasoner in 5 minutes!

## Installation

No installation needed! Just Python 3.6+

```bash
cd toy-safety-reasoner
```

## Try It Now

### 1. Interactive Demo (Recommended)

```bash
python3 demo.py
```

Follow the menu to:
- Evaluate your own content
- See example classifications
- Compare reasoning levels
- Learn about safety reasoners

### 2. Run Examples

```bash
# See all examples
python3 examples.py all

# Detailed example with full reasoning
python3 examples.py detailed

# Compare reasoning levels
python3 examples.py compare

# Multi-policy evaluation
python3 examples.py multi
```

### 3. Use as a Library

```python
from safety_reasoner import SafetyReasoner, ReasoningLevel

# Initialize
reasoner = SafetyReasoner()

# Evaluate content
result = reasoner.evaluate(
    content="Your text here",
    reasoning_level=ReasoningLevel.MEDIUM
)

# Check result
print(f"Classification: {result.overall_classification.value}")
print(f"Confidence: {result.overall_confidence:.1%}")

# See which policies were violated
for pe in result.policy_evaluations:
    if pe.classification.value == "unsafe":
        print(f"Violated: {pe.policy_name}")
```

## Example Session

```bash
$ python3 demo.py

Options:
  1. Evaluate custom content
  2. Run quick examples
  3. Compare reasoning levels
  4. Test specific policy
  5. View available policies
  6. Learn about safety reasoners
  q. Quit

Choose an option: 2

--- Safe: Political disagreement ---
Content: "I disagree with their tax policy proposals."
Result: ✓ SAFE (confidence: 98.3%)

--- Unsafe: Hate speech ---
Content: "People from that group are subhuman vermin."
Result: ✗ UNSAFE (confidence: 90.0%)
Violated: Hate Speech
```

## What to Try

### Test Safe Content
```
"I disagree with their political views."
"This historical analysis examines discrimination."
"Academic research on online harassment patterns."
```

### Test Unsafe Content
```
"Here's how to build explosives."
"People of [group] are subhuman."
"The best way to harm yourself is..."
```

### Test Edge Cases
```
"This research paper analyzes hate speech to improve detection."
"Self-defense techniques include strikes and holds."
"The novel contains graphic violence."
```

## Understanding Output

```
Classification: UNSAFE
Confidence: 85%

Violated Policies: Hate Speech

Reasoning Chain:
  1. Check obvious violations
     → Found keywords: "subhuman" (+0.30)
  2. Check indicators
     → Matched: dehumanizing language (+0.40)
  3. Analyze context
     → No educational framing (+0.15)
```

**Classification:**
- SAFE (confidence < 30%)
- UNCLEAR (confidence 30-70%)
- UNSAFE (confidence > 70%)

## Next Steps

1. Read `LEARNING_GUIDE.md` for deep dive
2. Modify `policies.json` to add your own policies
3. Experiment with different reasoning levels
4. Study `safety_reasoner.py` to understand the code

## Common Questions

**Q: Is this production-ready?**
A: No! This is educational only. Real systems use LLMs with billions of parameters.

**Q: Can I use this for my app?**
A: Not recommended. Check out OpenAI Moderation API or similar production services.

**Q: How accurate is it?**
A: ~60-70% for educational purposes. Real systems achieve 80-85%+ F1 scores.

**Q: Can I modify it?**
A: Yes! MIT license - learn, break, rebuild, experiment!

**Q: What if I want to build a real system?**
A: Start here to learn concepts, then integrate real LLMs (GPT-4, Claude, Llama, etc.)

## Troubleshooting

**Error: No module named 'safety_reasoner'**
```bash
# Make sure you're in the right directory
cd toy-safety-reasoner
python3 demo.py
```

**Error: File not found: policies.json**
```bash
# Make sure all files are present
ls -la
# Should see: policies.json, safety_reasoner.py, demo.py, examples.py
```

## Files Overview

- `README.md` - Project overview
- `QUICKSTART.md` - This file
- `LEARNING_GUIDE.md` - Deep educational content
- `policies.json` - Policy definitions
- `safety_reasoner.py` - Core implementation
- `examples.py` - Example test cases
- `demo.py` - Interactive demo

Happy learning!
