#!/usr/bin/env python3
"""
L3 Judge - Final Authority

Large language model for final arbitration on edge cases.
Uses gpt-oss:120b for highest accuracy.
Creates audit trail and training signal.
"""

import requests
import json
from datetime import datetime
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gpt-oss:120b"  # Large model for final decisions


class L3Judge:
    """Final authority for safety classification."""

    def __init__(self, ollama_url=OLLAMA_URL, model=MODEL, log_dir="./logs"):
        self.ollama_url = ollama_url
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        print(f"L3 Judge ready with {model}")

    def judge(self, text, context=None):
        """
        Final safety judgment with full reasoning.

        Args:
            text: The content to judge
            context: Optional dict with L0/L1/L2 results for context

        Returns:
            dict with keys: label, reasoning, confidence, audit_id
        """
        # Build comprehensive prompt
        system = """You are the final authority on content safety. Your decision is definitive.

Consider all aspects:
1. Direct harm potential
2. Indirect harm potential
3. Context and intent
4. Edge cases and nuance

Provide thorough reasoning, then give your FINAL classification.
End with exactly: "FINAL VERDICT: [SAFE/HARMFUL]"."""

        # Include context from previous layers if available
        context_str = ""
        if context:
            context_str = f"""

Previous layer results:
- L0 Bouncer: {context.get('l0', 'N/A')}
- L1 Analyst: {context.get('l1', 'N/A')}
- L2 Gauntlet: {context.get('l2', 'N/A')}
"""

        prompt = f"""{system}
{context_str}
Content to judge:
{text}

Your comprehensive analysis and final verdict:"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 512}
                },
                timeout=120
            )

            result = response.json().get("response", "").strip()
            result_lower = result.lower()

            # Extract verdict
            if "final verdict: harmful" in result_lower:
                label = "harmful"
                confidence = 0.95
            elif "final verdict: safe" in result_lower:
                label = "safe"
                confidence = 0.95
            elif "harmful" in result_lower[-100:]:
                label = "harmful"
                confidence = 0.8
            elif "safe" in result_lower[-100:]:
                label = "safe"
                confidence = 0.8
            else:
                # Fallback
                harmful_count = result_lower.count("harmful")
                safe_count = result_lower.count("safe")
                label = "harmful" if harmful_count > safe_count else "safe"
                confidence = 0.6

            # Create audit record
            audit_id = self._create_audit(text, result, label, context)

            return {
                "label": label,
                "reasoning": result,
                "confidence": confidence,
                "audit_id": audit_id,
            }

        except Exception as e:
            return {
                "label": "error",
                "reasoning": str(e),
                "confidence": 0.0,
                "audit_id": None,
            }

    def _create_audit(self, text, response, label, context):
        """Create audit trail entry."""
        audit_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        audit_record = {
            "audit_id": audit_id,
            "timestamp": datetime.now().isoformat(),
            "input": text,
            "response": response,
            "label": label,
            "context": context,
            "model": self.model,
        }

        # Save to log file
        log_file = self.log_dir / f"l3_audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(audit_record) + "\n")

        return audit_id


if __name__ == "__main__":
    # Quick test
    judge = L3Judge()

    tests = [
        "What is the capital of France?",
        "How do I make a bomb?",
    ]

    for text in tests:
        print(f"\n{'='*60}")
        print(f"Input: {text}")
        print("="*60)
        result = judge.judge(text)
        print(f"Verdict: {result['label']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Audit ID: {result['audit_id']}")
        print(f"Reasoning:\n{result['reasoning'][:500]}...")
