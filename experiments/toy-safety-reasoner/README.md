# Toy Safety Reasoner

A learning implementation inspired by OpenAI's gpt-oss-safeguard models to understand how safety reasoners work.

## What is a Safety Reasoner?

A safety reasoner is an AI system that:
1. Takes a **policy** (rules about what content is allowed/disallowed)
2. Takes **content** to evaluate
3. Uses **chain-of-thought reasoning** to analyze the content
4. Returns a **classification** (safe/unsafe) with transparent reasoning

## Key Concepts from the Research

Based on OpenAI's technical report on gpt-oss-safeguard:

- **Multi-policy evaluation**: Can check content against multiple policies simultaneously
- **Chain-of-thought transparency**: Shows how it reasoned about the classification
- **Customizable policies**: Policies can be defined at runtime
- **Reasoning effort levels**: Can use low/medium/high reasoning depth

## Project Structure

```
toy-safety-reasoner/
├── README.md
├── policies.json          # Policy definitions
├── safety_reasoner.py     # Main reasoner implementation
├── examples.py            # Example content to classify
└── demo.py               # Interactive demonstration
```

## How It Works

```python
# 1. Define a policy
policy = {
    "name": "hate-speech",
    "description": "Content that attacks or dehumanizes people based on protected characteristics",
    "examples_violating": [...],
    "examples_allowed": [...]
}

# 2. Evaluate content
result = reasoner.evaluate(
    content="Your content here",
    policies=[policy],
    reasoning_effort="medium"
)

# 3. Get transparent results
print(result.classification)  # "safe" or "unsafe"
print(result.reasoning)       # Chain of thought explanation
print(result.confidence)      # 0.0 - 1.0
```

## Learning Objectives

1. **Policy representation**: How to encode safety rules
2. **Reasoning chains**: Breaking down complex decisions into steps
3. **Multi-policy handling**: Evaluating against multiple rules
4. **Confidence calibration**: Understanding certainty in classifications
5. **Transparency**: Making AI decisions interpretable

## Usage

```bash
# Run interactive demo
python demo.py

# Batch evaluate examples
python examples.py

# Use as library
from safety_reasoner import SafetyReasoner
reasoner = SafetyReasoner()
result = reasoner.evaluate(content, policies)
```

## Differences from Production Systems

This is a **toy implementation** for learning. Real systems like gpt-oss-safeguard:
- Use 20B-120B parameter LLMs (we use simple heuristics + small models)
- Handle 14+ languages (we focus on English)
- Achieve 80%+ F1 scores (we're educational, not production-grade)
- Process millions of requests (we're for learning)

## Educational Resources

- [OpenAI gpt-oss-safeguard Technical Report](https://arxiv.org/abs/2508.10925)
- [Instruction Hierarchy Paper](https://arxiv.org/abs/2404.13208)
- [StrongReject Jailbreak Benchmark](https://arxiv.org/abs/2402.10260)

## License

MIT License - Educational purposes
