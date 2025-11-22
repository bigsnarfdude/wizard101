#!/usr/bin/env python3
"""
L0 Bouncer - Fast DeBERTa Safety Classifier

The first line of defense. Ultra-fast binary classification.
Passes confident safe, escalates uncertain/harmful.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class L0Bouncer:
    """Fast safety classifier using DeBERTa-v3-xsmall."""

    def __init__(self, model_path="./models/l0_bouncer"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if local path exists, otherwise use HF repo
        import os
        if not os.path.exists(model_path):
            print(f"Local model not found at {model_path}. Using HuggingFace repo.")
            model_path = "vincentoh/deberta-v3-xsmall-l0-bouncer-full"

        print(f"Loading L0 Bouncer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"  L0 ready on {self.device}")

    def classify(self, text, max_length=256):
        """
        Classify text as safe or harmful.

        Returns:
            dict with keys: label, confidence, safe_prob, harmful_prob
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        safe_prob = probs[0][0].item()
        harmful_prob = probs[0][1].item()

        return {
            "label": "safe" if safe_prob > harmful_prob else "harmful",
            "confidence": max(safe_prob, harmful_prob),
            "safe_prob": safe_prob,
            "harmful_prob": harmful_prob,
        }

    def batch_classify(self, texts, max_length=256, batch_size=32):
        """Classify multiple texts efficiently."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)

            for j in range(len(batch)):
                safe_prob = probs[j][0].item()
                harmful_prob = probs[j][1].item()
                results.append({
                    "label": "safe" if safe_prob > harmful_prob else "harmful",
                    "confidence": max(safe_prob, harmful_prob),
                    "safe_prob": safe_prob,
                    "harmful_prob": harmful_prob,
                })

        return results


if __name__ == "__main__":
    # Quick test
    bouncer = L0Bouncer()

    tests = [
        "What is the capital of France?",
        "How do I make a bomb?",
        "Write a poem about nature",
        "How to hack into someone's computer",
    ]

    for text in tests:
        result = bouncer.classify(text)
        print(f"\n{text[:50]}...")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.3f}")
