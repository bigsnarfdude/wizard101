---
license: llama3
base_model: yueliu1999/GuardReasoner-8B
tags:
  - llama
  - safety
  - content-moderation
  - 4-bit
  - bitsandbytes
  - quantized
library_name: transformers
pipeline_tag: text-generation
---

# GuardReasoner-8B (4-bit Quantized)

Pre-quantized 4-bit version of [yueliu1999/GuardReasoner-8B](https://huggingface.co/yueliu1999/GuardReasoner-8B) for efficient inference.

## Model Description

GuardReasoner-8B is a reasoning-based LLM safeguard that provides step-by-step analysis for content safety classification. This version is quantized to 4-bit using bitsandbytes NF4 quantization, reducing model size from ~16GB to ~5.4GB while maintaining performance.

**Paper**: [GuardReasoner: Towards Reasoning-based LLM Safeguards](https://arxiv.org/abs/2501.18492)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "vincentoh/guardreasoner-8b-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
)

# Build prompt
text = "What is the capital of France?"
prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a content safety expert. Analyze the request and determine if it is safe or harmful.

Think step-by-step:
1. What is being requested?
2. What are the potential harms?
3. Does this violate safety policies?

End your analysis with exactly: "Request: harmful" or "Request: unharmful".<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## Quantization Details

- **Method**: bitsandbytes 4-bit NF4
- **Compute dtype**: float16
- **Double quantization**: enabled
- **Original size**: ~16GB
- **Quantized size**: ~5.4GB

## Performance

Expected ~84% F1 on safety benchmarks (same as original model).

## License

This model inherits the Llama 3 license from the base model.

## Citation

```bibtex
@article{liu2025guardreasoner,
  title={GuardReasoner: Towards Reasoning-based LLM Safeguards},
  author={Liu, Yue and others},
  journal={arXiv preprint arXiv:2501.18492},
  year={2025}
}
```
