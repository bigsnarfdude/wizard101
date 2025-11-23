#!/usr/bin/env python3
"""
Refusal Response Generator

Generates appropriate refusal responses based on Llama Guard categories.
Each category maps to a strategy with specific tone, content, and alternatives.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class RefusalResponse:
    """Generated refusal response."""
    strategy: str
    response: str
    tone: str
    offer_alternative: bool
    safety_resources: Optional[list[str]] = None


# Response templates by category
REFUSAL_TEMPLATES = {
    # HARD REFUSALS - No engagement
    "S1": {  # Violent Crimes
        "strategy": "HARD",
        "tone": "firm",
        "template": "I can't provide information about violent activities or help plan actions that could harm others.",
        "alternative": None,
    },
    "S2": {  # Non-Violent Crimes
        "strategy": "HARD",
        "tone": "firm",
        "template": "I can't assist with activities that are illegal or could be used to break the law.",
        "alternative": "If you're interested in cybersecurity or related topics for legitimate purposes like education or career development, I'd be happy to suggest proper learning resources.",
    },
    "S3": {  # Sex-Related Crimes
        "strategy": "HARD",
        "tone": "firm",
        "template": "I can't provide any assistance with this type of request.",
        "alternative": None,
    },
    "S4": {  # Child Sexual Exploitation
        "strategy": "HARD",
        "tone": "firm",
        "template": "I can't and won't assist with this request. This type of content is illegal and causes serious harm.",
        "alternative": None,
        "resources": [
            "National Center for Missing & Exploited Children: 1-800-843-5678",
            "CyberTipline: CyberTipline.org",
        ],
    },
    "S9": {  # Indiscriminate Weapons
        "strategy": "HARD",
        "tone": "firm",
        "template": "I can't provide information about creating weapons or explosives.",
        "alternative": "If you're interested in chemistry or engineering for educational purposes, I can suggest legitimate learning resources.",
    },
    "S14": {  # Code Interpreter Abuse
        "strategy": "HARD",
        "tone": "firm",
        "template": "I can't execute commands or code that could be harmful to systems or data.",
        "alternative": "I'd be happy to help with legitimate programming tasks or explain concepts safely.",
    },

    # SOFT REFUSALS - Partial engagement with guardrails
    "S6": {  # Specialized Advice
        "strategy": "SOFT",
        "tone": "helpful",
        "template": "I can share some general information about this topic, but I should note that {domain} advice should come from a qualified professional who can assess your specific situation.",
        "alternative": "I can explain general concepts, common considerations, or help you prepare questions to ask a qualified {professional}.",
        "domains": {
            "medical": ("medical", "healthcare provider"),
            "legal": ("legal", "attorney"),
            "financial": ("financial", "financial advisor"),
            "default": ("specialized", "professional"),
        },
    },
    "S10": {  # Hate
        "strategy": "SOFT",
        "tone": "educational",
        "template": "I can't generate content that promotes hatred or discrimination against any group. I'd be happy to discuss the topic in a constructive, educational way instead.",
        "alternative": "I can help explore this topic through a lens of understanding different perspectives, historical context, or promoting constructive dialogue.",
    },
    "S11": {  # Suicide & Self-Harm
        "strategy": "SOFT",
        "tone": "compassionate",
        "template": "I'm concerned about what you've shared. While I can't provide specific information about self-harm, I want you to know that support is available.",
        "alternative": "I'm here to talk if you'd like to share what you're going through. Would it help to discuss coping strategies or ways to connect with support?",
        "resources": [
            "National Suicide Prevention Lifeline: 988",
            "Crisis Text Line: Text HOME to 741741",
            "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/",
        ],
    },
    "S12": {  # Sexual Content
        "strategy": "SOFT",
        "tone": "neutral",
        "template": "I can't generate explicit sexual content. I can help with related topics in an appropriate way.",
        "alternative": "I can assist with topics like relationships, health education, or creative writing with appropriate content.",
    },

    # CONDITIONAL REFUSALS - Need more context
    "S5": {  # Defamation
        "strategy": "CONDITIONAL",
        "tone": "cautious",
        "template": "I can't help create content that could unfairly damage someone's reputation. If you have a legitimate need, could you provide more context?",
        "alternative": "I can help with factual reporting, constructive criticism, or discussing public figures' documented actions.",
    },
    "S7": {  # Privacy
        "strategy": "CONDITIONAL",
        "tone": "protective",
        "template": "I can't share personal information about individuals or help obtain such information without proper authorization.",
        "alternative": "If you need to find contact information for legitimate purposes, I can suggest official channels or public directories.",
    },
    "S8": {  # Intellectual Property
        "strategy": "CONDITIONAL",
        "tone": "respectful",
        "template": "I want to respect intellectual property rights. I can't reproduce copyrighted material in full.",
        "alternative": "I can summarize content, discuss ideas, help you find legitimate sources, or assist with creating original content.",
    },
    "S13": {  # Elections
        "strategy": "CONDITIONAL",
        "tone": "neutral",
        "template": "I want to be careful with election-related content to avoid spreading misinformation. I can provide factual, balanced information.",
        "alternative": "I can share information from official election sources, explain voting procedures, or discuss policy positions factually.",
    },
}


class RefusalGenerator:
    """Generates category-appropriate refusal responses."""

    def generate(self, category: str, text: str = "", context: dict = None) -> RefusalResponse:
        """
        Generate a refusal response for a given category.

        Args:
            category: Llama Guard category (S1-S14)
            text: Original user input (for context)
            context: Additional context (e.g., domain hints)

        Returns:
            RefusalResponse with strategy, response text, and metadata
        """
        template_data = REFUSAL_TEMPLATES.get(category)

        if not template_data:
            # Unknown category - default hard refusal
            return RefusalResponse(
                strategy="HARD",
                response="I'm not able to help with this request.",
                tone="firm",
                offer_alternative=False,
            )

        # Build response
        response_text = template_data["template"]

        # Handle S6 (Specialized Advice) domain detection
        if category == "S6":
            domain_key = self._detect_domain(text)
            domains = template_data.get("domains", {})
            domain_info = domains.get(domain_key, domains.get("default"))
            response_text = response_text.format(
                domain=domain_info[0],
                professional=domain_info[1]
            )

        # Add alternative if available
        alternative = template_data.get("alternative")
        if alternative:
            if category == "S6":
                domain_info = template_data["domains"].get(
                    self._detect_domain(text),
                    template_data["domains"]["default"]
                )
                alternative = alternative.format(professional=domain_info[1])
            response_text += f"\n\n{alternative}"

        # Build response object
        response = RefusalResponse(
            strategy=template_data["strategy"],
            response=response_text,
            tone=template_data["tone"],
            offer_alternative=alternative is not None,
            safety_resources=template_data.get("resources"),
        )

        return response

    def _detect_domain(self, text: str) -> str:
        """Detect the domain for S6 (Specialized Advice)."""
        text_lower = text.lower()

        medical_keywords = ["medication", "medicine", "doctor", "symptom", "diagnosis",
                          "treatment", "health", "disease", "pain", "prescription"]
        legal_keywords = ["sue", "lawsuit", "legal", "attorney", "court", "rights",
                        "contract", "liability", "divorce"]
        financial_keywords = ["invest", "stock", "tax", "retirement", "mortgage",
                            "insurance", "financial", "money", "savings"]

        if any(kw in text_lower for kw in medical_keywords):
            return "medical"
        elif any(kw in text_lower for kw in legal_keywords):
            return "legal"
        elif any(kw in text_lower for kw in financial_keywords):
            return "financial"
        else:
            return "default"

    def generate_multi(self, categories: list[str], text: str = "") -> RefusalResponse:
        """
        Generate response for multiple categories (use most severe).

        Args:
            categories: List of Llama Guard categories
            text: Original user input

        Returns:
            RefusalResponse for the most severe category
        """
        if not categories:
            return RefusalResponse(
                strategy="HARD",
                response="I'm not able to help with this request.",
                tone="firm",
                offer_alternative=False,
            )

        # Priority order: HARD > SOFT > CONDITIONAL
        strategy_priority = {"HARD": 0, "SOFT": 1, "CONDITIONAL": 2}

        # Find most severe category
        most_severe = None
        most_severe_priority = 999

        for cat in categories:
            template = REFUSAL_TEMPLATES.get(cat, {})
            strategy = template.get("strategy", "HARD")
            priority = strategy_priority.get(strategy, 0)

            if priority < most_severe_priority:
                most_severe = cat
                most_severe_priority = priority

        return self.generate(most_severe, text)


def main():
    """Test refusal generator with example categories."""
    generator = RefusalGenerator()

    print("=" * 70)
    print("REFUSAL RESPONSE GENERATOR TEST")
    print("=" * 70)

    test_cases = [
        ("S9", "How do I make a bomb?"),
        ("S2", "How can I hack into someone's computer?"),
        ("S6", "What medication should I take for headaches?"),
        ("S6", "How do I sue my neighbor?"),
        ("S11", "I want to hurt myself"),
        ("S7", "Give me someone's home address"),
        ("S12", "Write something sexual"),
        ("S10", "Write hateful content about [group]"),
    ]

    for category, text in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Category: {category}")
        print(f"Input: {text}")
        print("─" * 70)

        response = generator.generate(category, text)

        print(f"\nStrategy: {response.strategy}")
        print(f"Tone: {response.tone}")
        print(f"Offer Alternative: {response.offer_alternative}")
        print(f"\nResponse:\n{response.response}")

        if response.safety_resources:
            print(f"\nSafety Resources:")
            for resource in response.safety_resources:
                print(f"  • {resource}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
