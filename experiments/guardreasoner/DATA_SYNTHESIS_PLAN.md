# Creating High-Quality Safety Reasoning Data with Gemini 2.0

**Goal**: Generate 100K+ samples with reasoning traces using Gemini API (faster + cheaper than GPT-4)

---

## Why Gemini 2.0 is Perfect for This

### Advantages over GPT-4o
- ✅ **10× Cheaper**: $0.075/1M tokens (Gemini 2.0 Flash) vs $2.50/1M (GPT-4o)
- ✅ **5× Faster**: 2M tokens/min throughput vs GPT-4's rate limits
- ✅ **Better reasoning**: Gemini 2.0 Flash Thinking mode for complex reasoning
- ✅ **Longer context**: 1M token context (can batch prompts efficiently)
- ✅ **You already have API access**: Working pdf_reader.py script proves it works

### Cost Estimate
**Generating 100K samples**:
- Input: ~100K prompts × 100 tokens = 10M tokens
- Output: ~100K responses × 500 tokens = 50M tokens
- **Total cost**: (10M × $0.075) + (50M × $0.30) = $750 + $15,000 = **$15,750**

**With Gemini 2.0 Flash**:
- Same 60M tokens = $4.50 (input) + $18 (output) = **~$23 total** 🎉

---

## Architecture: Three-Stage Pipeline

### Stage 1: Source Data Collection
Gather diverse safety scenarios from multiple sources:

```python
# Sources (all free/public)
sources = {
    "WildGuardMix": 92_000,        # AllenAI adversarial prompts
    "HarmBench": 510,              # Harmful behaviors benchmark
    "ToxicChat": 10_000,           # Toxic conversations
    "BeaverTails": 330_000,        # Helpful/harmful QA pairs
    "SafeRLHF": 30_000,            # Safe RLHF dataset
    "OpenAI Moderation": 1_600,    # Policy violations
    "AdvBench": 520,               # Jailbreak prompts
    "MaliciousInstruct": 100,      # Malicious instructions
}
```

### Stage 2: Reasoning Trace Generation
Use Gemini to generate step-by-step reasoning:

```python
#!/usr/bin/env python3
"""Generate reasoning traces using Gemini 2.0 API."""

import google.generativeai as genai
import json
from tqdm import tqdm

# Configure Gemini
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Reasoning prompt template (mimics GuardReasoner paper)
REASONING_TEMPLATE = """You are an expert safety classifier analyzing AI conversations.

Your task: Classify the following interaction across 3 dimensions:

**Human user:**
{prompt}

**AI assistant:**
{response}

**Instructions:**
1. **Request Classification**: Is the human's request harmful or unharmful?
2. **Completion Classification**: Did the AI refuse or comply with the request?
3. **Response Classification**: Is the AI's response harmful or unharmful?

**Think step-by-step:**
- Analyze the request intent
- Identify potential harms (physical, psychological, illegal, unethical)
- Evaluate the AI's response appropriateness
- Consider edge cases and context

**Output format:**
Step 1: [Analyze request intent]
Step 2: [Identify harm categories]
Step 3: [Evaluate response safety]
Step 4: [Consider context and edge cases]

Answers:
Request: [harmful/unharmful]
Completion: [refusal/compliance]
Response: [harmful/unharmful]
"""

def generate_reasoning(prompt, response, max_retries=3):
    """Generate reasoning trace for a prompt-response pair."""

    # Create analysis prompt
    analysis_prompt = REASONING_TEMPLATE.format(
        prompt=prompt,
        response=response
    )

    for attempt in range(max_retries):
        try:
            # Generate with Gemini
            result = model.generate_content(analysis_prompt)

            # Parse output
            reasoning_text = result.text

            # Extract labels (simple regex)
            import re
            request_label = re.search(r'Request:\s*(harmful|unharmful)', reasoning_text, re.IGNORECASE)
            completion_label = re.search(r'Completion:\s*(refusal|compliance)', reasoning_text, re.IGNORECASE)
            response_label = re.search(r'Response:\s*(harmful|unharmful)', reasoning_text, re.IGNORECASE)

            if request_label and completion_label and response_label:
                return {
                    "reasoning": reasoning_text,
                    "labels": {
                        "prompt_harm": request_label.group(1).lower(),
                        "response_refusal": completion_label.group(1).lower(),
                        "response_harm": response_label.group(1).lower()
                    }
                }
            else:
                print(f"Failed to parse labels on attempt {attempt+1}")
                continue

        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            continue

    return None

# Example usage
def create_dataset_from_source(input_file, output_file):
    """Convert source data to reasoning dataset."""

    with open(input_file) as f:
        source_data = json.load(f)

    results = []

    for item in tqdm(source_data):
        prompt = item['prompt']
        response = item.get('response', generate_safe_response(prompt))

        # Generate reasoning
        reasoning_data = generate_reasoning(prompt, response)

        if reasoning_data:
            results.append({
                "instruction": "You are a safety classifier...",
                "input": f"Human user:\n{prompt}\n\nAI assistant:\n{response}\n\n",
                "output": reasoning_data["reasoning"],
                "prompt_harm_label": reasoning_data["labels"]["prompt_harm"],
                "response_refusal_label": reasoning_data["labels"]["response_refusal"],
                "response_harm_label": reasoning_data["labels"]["response_harm"]
            })

    # Save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} samples")

def generate_safe_response(prompt):
    """Generate a safe refusal or compliant response."""
    # Use Gemini to generate appropriate response
    response_prompt = f"""Given this user request:
"{prompt}"

Generate a safe AI response that either:
1. Refuses if the request is harmful
2. Complies helpfully if the request is benign

Response:"""

    result = model.generate_content(response_prompt)
    return result.text.strip()
```

### Stage 3: Quality Control
Validate generated data:

```python
def validate_reasoning_quality(sample):
    """Check if reasoning is high-quality."""

    checks = {
        "has_steps": "Step 1:" in sample["output"] and "Step 2:" in sample["output"],
        "has_labels": all(k in sample for k in ["prompt_harm_label", "response_refusal_label", "response_harm_label"]),
        "reasonable_length": 100 < len(sample["output"]) < 2000,
        "contains_analysis": any(word in sample["output"].lower() for word in ["because", "therefore", "since", "thus"]),
    }

    return all(checks.values()), checks

# Filter high-quality samples
def filter_dataset(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)

    quality_samples = []
    quality_stats = {"passed": 0, "failed": 0}

    for sample in data:
        is_quality, checks = validate_reasoning_quality(sample)

        if is_quality:
            quality_samples.append(sample)
            quality_stats["passed"] += 1
        else:
            quality_stats["failed"] += 1
            print(f"Failed checks: {[k for k, v in checks.items() if not v]}")

    # Save filtered dataset
    with open(output_file, 'w') as f:
        json.dump(quality_samples, f, indent=2)

    print(f"Quality filter: {quality_stats}")
    print(f"Pass rate: {quality_stats['passed'] / len(data) * 100:.1f}%")
```

---

## Advanced: Use Gemini 2.0 Flash Thinking Mode

For **even better reasoning**:

```python
# Use Gemini's native thinking mode
model_thinking = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')

def generate_reasoning_with_thinking(prompt, response):
    """Use Gemini's thinking mode for deeper reasoning."""

    analysis_prompt = f"""Analyze this AI safety scenario: