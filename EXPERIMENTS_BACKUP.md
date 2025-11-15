# Experiments Backup Summary

**Date:** 2025-11-15
**Purpose:** Document experiments structure for disaster recovery

## What Was Added

### Directory Structure
```
wizard101/
├── experiments/
│   ├── README.md                      # Complete implementation guide
│   ├── serial_gauntlet_simple.py      # Main implementation (261 lines)
│   └── policies/                      # 6 optimized policy files
│       ├── hate_speech.txt            # 591 tokens
│       ├── violence.txt               # 576 tokens
│       ├── self_harm.txt              # 579 tokens
│       ├── sexual_minors.txt          # 547 tokens
│       ├── harassment.txt             # 581 tokens
│       └── illegal.txt                # 592 tokens
│
├── LAYER2_GAUNTLET_EXPLAINED.md       # ONE model vs 6 models clarification
├── PRODUCTION_ARCHITECTURE.md         # Full 4-layer defense onion
├── WHAT_IS_SAFETY_REASONER.md         # Conceptual overview
├── SIMPLE_EXAMPLE.md                  # Examples comparing approaches
└── SUNDAY_PIPELINE_EXPERIMENTS.md     # Experiment plans
```

## Key Implementation Details

### Architecture
- **ONE model** (gpt-oss:20b, ~13GB VRAM)
- **SIX policy files** (400-600 tokens each)
- **Serial execution** (~12 seconds total)
- **Ollama /api/chat endpoint** with Harmony format

### Policy Files
Each policy follows this structure:
- INSTRUCTIONS (respond "VIOLATION" or "SAFE")
- DEFINITIONS (key terms)
- VIOLATES POLICY (R1-R5 rules)
- SAFE CONTENT (S1-S5 exceptions)
- EXAMPLES (5 examples with rationale)
- EDGE CASES (guidance for ambiguity)

### Serial Gauntlet Flow
```python
# Load all 6 policies
policies = load_policies()

# For each policy:
for policy_name, policy_text in policies.items():
    # Same model, different policy in system message
    messages = [
        {"role": "system", "content": policy_text},
        {"role": "user", "content": f"Content to analyze: {content}"}
    ]
    
    # Call Ollama
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "gpt-oss:20b", "messages": messages, "stream": False}
    )
    
    # Parse Harmony format (both thinking and content fields)
    result = parse_response(response)
    results.append(result)

# Aggregate results
violations = [r for r in results if r.violation]
```

## Testing on nigel.birs.ca

### Prerequisites
```bash
# Ensure Ollama is running
ollama list | grep gpt-oss:20b

# If not available
ollama pull gpt-oss:20b
```

### Running Experiments
```bash
cd ~/wizard101/experiments
python3 serial_gauntlet_simple.py
```

### Expected Output
- 3 test cases processed
- ~12 seconds per test (6 policies × 2s each)
- Detailed results showing which policies triggered

## Backup Locations

### GitHub
- Repository: https://github.com/bigsnarfdude/wizard101
- Commit: 45c2b22
- Branch: main

### Local
- Primary: /Users/vincent/development/wizard101/
- Experiments: /Users/vincent/development/wizard101/experiments/

### Related Research
- llm-abuse-patterns: /Users/vincent/development/llm-abuse-patterns/
- Baseline safeguard.py reference

## Recovery Instructions

If local machine fails:

1. **Clone from GitHub:**
```bash
git clone https://github.com/bigsnarfdude/wizard101.git
cd wizard101
```

2. **Verify files:**
```bash
ls -la experiments/policies/
# Should show all 6 .txt files
```

3. **Test on nigel:**
```bash
scp -r experiments vincent@nigel.birs.ca:~/wizard101/
ssh vincent@nigel.birs.ca "cd ~/wizard101/experiments && python3 serial_gauntlet_simple.py"
```

## Key Findings to Preserve

### From llm-abuse-patterns Research
- **Baseline > Safeguard** for 20B models on multi-policy
- **400-600 tokens** optimal policy length
- **Harmony format** requires parsing both thinking and content
- **DO NOT use** `"format": "json"` in Ollama calls (breaks Harmony)

### Architecture Clarification
- **NOT 6 separate models** (120GB VRAM) ❌
- **ONE model, 6 text files** (13GB VRAM) ✅
- Serial and parallel produce identical output, different timing

## Sunday Experiment Goals

1. **Layer Distribution** - How much does Layer 2 catch?
2. **Policy Effectiveness** - Which policies are most accurate?
3. **Multi-Policy Accuracy** - Compare to OpenAI's 46.3% baseline
4. **Token Optimization** - Validate 400-600 token range
5. **Aggregation Strategy** - Handle conflicting results

## Documentation Files

### Core Concepts
- `WHAT_IS_SAFETY_REASONER.md` - What is it and why it matters
- `SIMPLE_EXAMPLE.md` - Traditional vs reasoner comparison
- `LAYER2_GAUNTLET_EXPLAINED.md` - Architecture clarification

### Production Patterns
- `PRODUCTION_ARCHITECTURE.md` - Full 4-layer defense onion
- `SUNDAY_PIPELINE_EXPERIMENTS.md` - Experiment methodology

### Implementation
- `experiments/README.md` - Complete how-to guide
- `experiments/serial_gauntlet_simple.py` - Working code

## Version Control

```bash
# Latest commit
git log -1 --oneline
# 45c2b22 Add Serial Gauntlet experiments with 6 optimized policies

# Files changed
git show --name-only --oneline HEAD
```

## Contact Points

- **Local Development:** /Users/vincent/development/wizard101/
- **Experiment Server:** vincent@nigel.birs.ca:~/wizard101/
- **GitHub Repo:** https://github.com/bigsnarfdude/wizard101
- **Related Project:** ~/development/llm-abuse-patterns/

---

**Status:** ✅ All files committed and pushed to GitHub
**Last Updated:** 2025-11-15
**Backup Verified:** Yes
