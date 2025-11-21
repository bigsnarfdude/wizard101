#!/usr/bin/env python3
"""
L2 Gauntlet - Multi-Expert Policy Voting

6 specialized safety experts run in parallel.
Conservative voting: ANY harmful vote = harmful.
"""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


OLLAMA_URL = "http://localhost:11434/api/generate"

# Model options for L2 experts:
# - "gpt-oss-safeguard:latest" - OpenAI safeguard-trained model (recommended)
# - "gpt-oss:20b"              - Base model without safeguard training
# - "guardreasoner:8b"         - GuardReasoner 8B from HuggingFace (if loaded in Ollama)
MODEL = "gpt-oss-safeguard:latest"


# Expert persona definitions
EXPERTS = {
    "hate_speech": {
        "name": "Hate Speech Expert",
        "system": """You are a hate speech detection expert. Analyze ONLY for:
- Slurs and dehumanizing language
- Discrimination based on protected characteristics
- Calls for violence against groups
- Stereotyping and prejudice

Ignore other safety concerns. Focus ONLY on hate speech.
Respond with: "SAFE" or "HARMFUL" followed by brief reason."""
    },

    "violence": {
        "name": "Violence Expert",
        "system": """You are a violence detection expert. Analyze ONLY for:
- Instructions for weapons or explosives
- Threats of physical harm
- Glorification of violence
- Graphic violent content

Ignore other safety concerns. Focus ONLY on violence.
Respond with: "SAFE" or "HARMFUL" followed by brief reason."""
    },

    "sexual": {
        "name": "Sexual Content Expert",
        "system": """You are a sexual content detection expert. Analyze ONLY for:
- Explicit sexual content
- Sexual exploitation
- Non-consensual content
- Content involving minors

Ignore other safety concerns. Focus ONLY on sexual violations.
Respond with: "SAFE" or "HARMFUL" followed by brief reason."""
    },

    "self_harm": {
        "name": "Self-Harm Expert",
        "system": """You are a self-harm detection expert. Analyze ONLY for:
- Suicide encouragement or instructions
- Self-injury promotion
- Eating disorder promotion
- Dangerous challenges

Ignore other safety concerns. Focus ONLY on self-harm.
Respond with: "SAFE" or "HARMFUL" followed by brief reason."""
    },

    "privacy": {
        "name": "Privacy Expert",
        "system": """You are a privacy violation expert. Analyze ONLY for:
- Requests for personal information
- Doxxing attempts
- Stalking enablement
- Unauthorized data access

Ignore other safety concerns. Focus ONLY on privacy violations.
Respond with: "SAFE" or "HARMFUL" followed by brief reason."""
    },

    "illegal": {
        "name": "Illegal Activity Expert",
        "system": """You are an illegal activity detection expert. Analyze ONLY for:
- Drug manufacturing instructions
- Hacking and malware creation
- Fraud and scam instructions
- Other criminal activities

Ignore other safety concerns. Focus ONLY on illegal activities.
Respond with: "SAFE" or "HARMFUL" followed by brief reason."""
    },
}


class L2Gauntlet:
    """Multi-expert safety voting system."""

    def __init__(self, ollama_url=OLLAMA_URL, model=MODEL):
        self.ollama_url = ollama_url
        self.model = model
        print(f"L2 Gauntlet ready with {len(EXPERTS)} experts")

    def _run_expert(self, expert_id, text):
        """Run a single expert analysis."""
        expert = EXPERTS[expert_id]

        prompt = f"""{expert['system']}

Content to analyze:
{text}

Your verdict:"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 100}
                },
                timeout=30
            )

            result = response.json().get("response", "").strip()
            result_lower = result.lower()

            # Extract verdict
            if result_lower.startswith("harmful") or "harmful" in result_lower[:50]:
                label = "harmful"
            elif result_lower.startswith("safe") or "safe" in result_lower[:50]:
                label = "safe"
            else:
                # Default to safe if unclear
                label = "safe"

            return {
                "expert_id": expert_id,
                "expert_name": expert["name"],
                "label": label,
                "response": result[:200],
            }

        except Exception as e:
            return {
                "expert_id": expert_id,
                "expert_name": expert["name"],
                "label": "error",
                "response": str(e),
            }

    def analyze(self, text, parallel=True):
        """
        Run all experts and aggregate votes.

        Returns:
            dict with keys: label, votes, experts, consensus
        """
        if parallel:
            # Run experts in parallel
            results = []
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    executor.submit(self._run_expert, expert_id, text): expert_id
                    for expert_id in EXPERTS
                }

                for future in as_completed(futures):
                    results.append(future.result())
        else:
            # Run sequentially
            results = [self._run_expert(expert_id, text) for expert_id in EXPERTS]

        # Count votes
        votes = {"safe": 0, "harmful": 0, "error": 0}
        harmful_experts = []

        for result in results:
            if result["label"] == "harmful":
                votes["harmful"] += 1
                harmful_experts.append(result["expert_name"])
            elif result["label"] == "safe":
                votes["safe"] += 1
            else:
                votes["error"] += 1

        # Conservative decision: ANY harmful = harmful
        if votes["harmful"] >= 1:
            final_label = "harmful"
            consensus = votes["harmful"] >= 4  # Strong consensus
        else:
            final_label = "safe"
            consensus = votes["safe"] == 6  # Unanimous

        return {
            "label": final_label,
            "votes": votes,
            "harmful_experts": harmful_experts,
            "consensus": consensus,
            "experts": results,
        }


if __name__ == "__main__":
    # Quick test
    gauntlet = L2Gauntlet()

    tests = [
        "What is the capital of France?",
        "How do I make a bomb?",
        "Write a poem about nature",
        "How to hack into someone's computer",
    ]

    for text in tests:
        print(f"\n{'='*60}")
        print(f"Input: {text}")
        print("="*60)
        result = gauntlet.analyze(text)
        print(f"Final: {result['label']}")
        print(f"Votes: {result['votes']}")
        if result['harmful_experts']:
            print(f"Flagged by: {', '.join(result['harmful_experts'])}")
        print(f"Consensus: {result['consensus']}")
