#!/usr/bin/env python3
"""
Normalize SG-Bench dataset to standard JSON format.

SG-Bench contains 1,442 prompts for evaluating LLM safety across various tasks and prompt types.
Source: https://github.com/MurrayTom/SG-Bench
"""

import json
import os

def normalize_sgbench(input_path: str, output_path: str) -> int:
    """
    Convert SG-Bench malicious_instruction.json to normalized format.

    Args:
        input_path: Path to source malicious_instruction.json
        output_path: Path to output normalized JSON

    Returns:
        Number of samples processed
    """
    # Load source data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize to standard format
    # All prompts in SG-Bench are malicious/harmful queries
    normalized = []
    for item in data:
        # Use the 'query' field as the prompt text
        prompt_text = item.get('query', '').strip()

        if prompt_text:
            normalized.append({
                "text": prompt_text,
                "label": "harmful"
            })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save normalized data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    return len(normalized)


def main():
    # Paths
    input_path = "/tmp/sg-bench-repo/datasets/SG-Bench/original_query/malicious_instruction.json"
    output_path = "/Users/vincent/development/wizard101/data/benchmark/sgbench.json"

    # Normalize
    count = normalize_sgbench(input_path, output_path)

    print(f"SG-Bench normalization complete!")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Total samples: {count}")
    print(f"  Labels: all 'harmful' (SG-Bench contains only malicious queries)")


if __name__ == "__main__":
    main()
