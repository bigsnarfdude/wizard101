#!/usr/bin/env python3
"""
Toy Safety Reasoner - Educational implementation inspired by gpt-oss-safeguard

This is a simplified version demonstrating the core concepts of safety reasoning:
1. Policy-based content evaluation
2. Chain-of-thought transparency
3. Multi-policy classification
4. Confidence scoring

NOT FOR PRODUCTION USE - Educational purposes only
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Classification(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNCLEAR = "unclear"


class ReasoningLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ReasoningStep:
    """A single step in the chain of thought"""
    step_number: int
    description: str
    finding: str
    confidence_impact: float  # -1.0 to 1.0


@dataclass
class PolicyEvaluation:
    """Result of evaluating content against a single policy"""
    policy_id: str
    policy_name: str
    classification: Classification
    confidence: float  # 0.0 to 1.0
    reasoning_chain: List[ReasoningStep]
    matched_indicators: List[str]


@dataclass
class SafetyEvaluationResult:
    """Complete evaluation result across all policies"""
    content: str
    overall_classification: Classification
    overall_confidence: float
    policy_evaluations: List[PolicyEvaluation]
    reasoning_summary: str
    reasoning_level: ReasoningLevel


class SafetyReasoner:
    """
    A toy implementation of a safety reasoner.

    In production systems like gpt-oss-safeguard, this would be a 20B-120B
    parameter LLM. Here we use simple heuristics to demonstrate concepts.
    """

    def __init__(self, policies_path: str = "policies.json"):
        """Initialize with policies from JSON file"""
        with open(policies_path, 'r') as f:
            self.policies_data = json.load(f)
        self.policies = self.policies_data['policies']

    def evaluate(
        self,
        content: str,
        policy_ids: Optional[List[str]] = None,
        reasoning_level: ReasoningLevel = ReasoningLevel.MEDIUM
    ) -> SafetyEvaluationResult:
        """
        Evaluate content against specified policies.

        Args:
            content: The text content to evaluate
            policy_ids: List of policy IDs to check (None = all policies)
            reasoning_level: Depth of reasoning (LOW, MEDIUM, HIGH)

        Returns:
            SafetyEvaluationResult with classifications and reasoning
        """
        # Select policies to evaluate
        if policy_ids is None:
            policies_to_check = self.policies
        else:
            policies_to_check = [p for p in self.policies if p['id'] in policy_ids]

        # Evaluate against each policy
        policy_evaluations = []
        for policy in policies_to_check:
            evaluation = self._evaluate_single_policy(content, policy, reasoning_level)
            policy_evaluations.append(evaluation)

        # Aggregate results
        overall_classification, overall_confidence = self._aggregate_evaluations(
            policy_evaluations
        )

        reasoning_summary = self._generate_summary(
            policy_evaluations,
            overall_classification,
            reasoning_level
        )

        return SafetyEvaluationResult(
            content=content,
            overall_classification=overall_classification,
            overall_confidence=overall_confidence,
            policy_evaluations=policy_evaluations,
            reasoning_summary=reasoning_summary,
            reasoning_level=reasoning_level
        )

    def _evaluate_single_policy(
        self,
        content: str,
        policy: Dict,
        reasoning_level: ReasoningLevel
    ) -> PolicyEvaluation:
        """Evaluate content against a single policy with chain-of-thought"""

        reasoning_chain = []
        confidence_score = 0.5  # Start neutral
        matched_indicators = []

        content_lower = content.lower()

        # Step 1: Check for obvious violations
        step1 = self._reason_step_1_obvious_violations(content, content_lower, policy)
        reasoning_chain.append(step1)
        confidence_score += step1.confidence_impact

        # Step 2: Check policy indicators
        step2, indicators = self._reason_step_2_indicators(content_lower, policy)
        reasoning_chain.append(step2)
        confidence_score += step2.confidence_impact
        matched_indicators.extend(indicators)

        # Step 3 (MEDIUM+): Context analysis
        if reasoning_level in [ReasoningLevel.MEDIUM, ReasoningLevel.HIGH]:
            step3 = self._reason_step_3_context(content, policy)
            reasoning_chain.append(step3)
            confidence_score += step3.confidence_impact

        # Step 4 (HIGH): Edge case consideration
        if reasoning_level == ReasoningLevel.HIGH:
            step4 = self._reason_step_4_edge_cases(content, policy)
            reasoning_chain.append(step4)
            confidence_score += step4.confidence_impact

        # Step 5 (HIGH): Example comparison
        if reasoning_level == ReasoningLevel.HIGH:
            step5 = self._reason_step_5_examples(content, content_lower, policy)
            reasoning_chain.append(step5)
            confidence_score += step5.confidence_impact

        # Determine classification
        confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]

        if confidence_score >= 0.7:
            classification = Classification.UNSAFE
        elif confidence_score <= 0.3:
            classification = Classification.SAFE
        else:
            classification = Classification.UNCLEAR

        return PolicyEvaluation(
            policy_id=policy['id'],
            policy_name=policy['name'],
            classification=classification,
            confidence=confidence_score,
            reasoning_chain=reasoning_chain,
            matched_indicators=matched_indicators
        )

    def _reason_step_1_obvious_violations(
        self, content: str, content_lower: str, policy: Dict
    ) -> ReasoningStep:
        """Check for obvious policy violations"""

        # Simple keyword matching (in production, this would be LLM reasoning)
        violation_keywords = {
            'hate-speech': ['subhuman', 'vermin', 'should be killed', 'deserve to die'],
            'violence-instructions': ['how to build a bomb', 'steps to attack', 'weapon tutorial'],
            'self-harm': ['how to kill yourself', 'suicide method', 'ways to end your life'],
            'sexual-minors': ['sexual content', 'minor', 'child'],
            'harassment': ['home address', 'dox', 'flood their inbox'],
            'illegal-activities': ['synthesize methamphetamine', 'credit card fraud', 'hacking tutorial']
        }

        policy_keywords = violation_keywords.get(policy['id'], [])
        found_violations = [kw for kw in policy_keywords if kw in content_lower]

        if found_violations:
            return ReasoningStep(
                step_number=1,
                description="Check for obvious policy violations",
                finding=f"Found potential violations: {', '.join(found_violations)}",
                confidence_impact=0.3
            )
        else:
            return ReasoningStep(
                step_number=1,
                description="Check for obvious policy violations",
                finding="No obvious violation keywords detected",
                confidence_impact=-0.1
            )

    def _reason_step_2_indicators(
        self, content_lower: str, policy: Dict
    ) -> Tuple[ReasoningStep, List[str]]:
        """Check for policy-specific indicators"""

        indicators = policy.get('indicators', [])
        matched = []

        for indicator in indicators:
            # Simple substring matching (production would use semantic understanding)
            indicator_words = indicator.lower().split()
            if any(word in content_lower for word in indicator_words):
                matched.append(indicator)

        if matched:
            impact = len(matched) * 0.15  # Each indicator adds risk
            finding = f"Matched {len(matched)} indicators: {', '.join(matched[:3])}"
        else:
            impact = -0.1
            finding = "No policy indicators matched"

        return ReasoningStep(
            step_number=2,
            description=f"Analyze against {policy['name']} indicators",
            finding=finding,
            confidence_impact=min(impact, 0.4)  # Cap impact
        ), matched

    def _reason_step_3_context(self, content: str, policy: Dict) -> ReasoningStep:
        """Analyze contextual factors (MEDIUM reasoning)"""

        # Check for educational/academic context
        educational_markers = [
            'research', 'study', 'academic', 'analysis', 'historical',
            'educational', 'learning', 'understanding', 'prevent'
        ]

        has_educational_context = any(marker in content.lower() for marker in educational_markers)

        # Check for harmful intent markers
        intent_markers = [
            'should', 'must', 'need to', 'have to', 'going to',
            'will', 'recommend', 'suggest', 'advise'
        ]

        has_intent = any(marker in content.lower() for marker in intent_markers)

        if has_educational_context and not has_intent:
            return ReasoningStep(
                step_number=3,
                description="Analyze contextual factors",
                finding="Content appears educational/analytical without harmful intent",
                confidence_impact=-0.2
            )
        elif has_intent:
            return ReasoningStep(
                step_number=3,
                description="Analyze contextual factors",
                finding="Content contains directive language suggesting actionable intent",
                confidence_impact=0.15
            )
        else:
            return ReasoningStep(
                step_number=3,
                description="Analyze contextual factors",
                finding="Neutral context without clear educational or harmful framing",
                confidence_impact=0.0
            )

    def _reason_step_4_edge_cases(self, content: str, policy: Dict) -> ReasoningStep:
        """Consider edge cases and exceptions (HIGH reasoning)"""

        # Check for allowed exceptions from policy
        allowed_examples = policy.get('examples_allowed', [])

        # Simple similarity check (production would use embeddings)
        content_words = set(content.lower().split())

        for allowed in allowed_examples:
            allowed_words = set(allowed.lower().split())
            overlap = len(content_words & allowed_words)
            if overlap > len(allowed_words) * 0.3:  # 30% word overlap
                return ReasoningStep(
                    step_number=4,
                    description="Consider edge cases and policy exceptions",
                    finding=f"Content similar to allowed exception: '{allowed[:50]}...'",
                    confidence_impact=-0.2
                )

        return ReasoningStep(
            step_number=4,
            description="Consider edge cases and policy exceptions",
            finding="No clear exceptions apply",
            confidence_impact=0.0
        )

    def _reason_step_5_examples(
        self, content: str, content_lower: str, policy: Dict
    ) -> ReasoningStep:
        """Compare against violating examples (HIGH reasoning)"""

        violating_examples = policy.get('examples_violating', [])

        content_words = set(content_lower.split())
        max_similarity = 0.0
        most_similar = None

        for example in violating_examples:
            example_words = set(example.lower().split())
            if len(example_words) > 0:
                similarity = len(content_words & example_words) / len(example_words)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar = example

        if max_similarity > 0.4:  # High similarity to violation example
            return ReasoningStep(
                step_number=5,
                description="Compare to known violation patterns",
                finding=f"High similarity ({max_similarity:.0%}) to violation example",
                confidence_impact=0.25
            )
        elif max_similarity > 0.2:
            return ReasoningStep(
                step_number=5,
                description="Compare to known violation patterns",
                finding=f"Moderate similarity ({max_similarity:.0%}) to violation patterns",
                confidence_impact=0.1
            )
        else:
            return ReasoningStep(
                step_number=5,
                description="Compare to known violation patterns",
                finding="Low similarity to known violations",
                confidence_impact=-0.05
            )

    def _aggregate_evaluations(
        self, policy_evaluations: List[PolicyEvaluation]
    ) -> Tuple[Classification, float]:
        """
        Aggregate multiple policy evaluations into overall classification.

        Mimics the multi-policy accuracy approach from gpt-oss-safeguard.
        """

        if not policy_evaluations:
            return Classification.SAFE, 1.0

        # Count classifications
        unsafe_count = sum(1 for pe in policy_evaluations
                          if pe.classification == Classification.UNSAFE)
        unclear_count = sum(1 for pe in policy_evaluations
                           if pe.classification == Classification.UNCLEAR)

        # Any unsafe = overall unsafe (conservative approach)
        if unsafe_count > 0:
            # Average confidence of unsafe classifications
            unsafe_confidences = [pe.confidence for pe in policy_evaluations
                                 if pe.classification == Classification.UNSAFE]
            avg_confidence = sum(unsafe_confidences) / len(unsafe_confidences)
            return Classification.UNSAFE, avg_confidence

        # Any unclear without unsafe = overall unclear
        if unclear_count > 0:
            unclear_confidences = [pe.confidence for pe in policy_evaluations
                                  if pe.classification == Classification.UNCLEAR]
            avg_confidence = sum(unclear_confidences) / len(unclear_confidences)
            return Classification.UNCLEAR, avg_confidence

        # All safe = overall safe
        safe_confidences = [1.0 - pe.confidence for pe in policy_evaluations]
        avg_confidence = sum(safe_confidences) / len(safe_confidences)
        return Classification.SAFE, avg_confidence

    def _generate_summary(
        self,
        policy_evaluations: List[PolicyEvaluation],
        overall_classification: Classification,
        reasoning_level: ReasoningLevel
    ) -> str:
        """Generate human-readable summary of reasoning"""

        summary_parts = []
        summary_parts.append(f"=== Safety Evaluation Summary ({reasoning_level.value} reasoning) ===\n")
        summary_parts.append(f"Overall Classification: {overall_classification.value.upper()}\n")

        for pe in policy_evaluations:
            summary_parts.append(f"\n--- {pe.policy_name} ---")
            summary_parts.append(f"Classification: {pe.classification.value}")
            summary_parts.append(f"Confidence: {pe.confidence:.2f}")

            if pe.matched_indicators:
                summary_parts.append(f"Matched Indicators: {', '.join(pe.matched_indicators[:3])}")

            summary_parts.append("\nReasoning Chain:")
            for step in pe.reasoning_chain:
                impact_str = f"+{step.confidence_impact:.2f}" if step.confidence_impact >= 0 else f"{step.confidence_impact:.2f}"
                summary_parts.append(f"  {step.step_number}. {step.description}")
                summary_parts.append(f"     → {step.finding} (impact: {impact_str})")

        return "\n".join(summary_parts)


def format_result(result: SafetyEvaluationResult) -> str:
    """Format evaluation result for display"""

    output = []
    output.append("=" * 80)
    output.append("SAFETY EVALUATION RESULT")
    output.append("=" * 80)
    output.append(f"\nContent: \"{result.content[:100]}{'...' if len(result.content) > 100 else ''}\"")
    output.append(f"\nOverall Classification: {result.overall_classification.value.upper()}")
    output.append(f"Overall Confidence: {result.overall_confidence:.2%}")
    output.append(f"Reasoning Level: {result.reasoning_level.value}")
    output.append(f"\nPolicies Evaluated: {len(result.policy_evaluations)}")

    output.append("\n" + "-" * 80)
    output.append("DETAILED REASONING")
    output.append("-" * 80)
    output.append(result.reasoning_summary)
    output.append("\n" + "=" * 80)

    return "\n".join(output)


if __name__ == "__main__":
    # Simple test
    reasoner = SafetyReasoner()

    test_cases = [
        "I disagree with their political views on taxation.",
        "People of that group are subhuman and should be removed.",
        "This is a historical analysis of discrimination in the 1960s."
    ]

    for test in test_cases:
        print(f"\n\nTesting: {test}")
        result = reasoner.evaluate(test, reasoning_level=ReasoningLevel.HIGH)
        print(f"Result: {result.overall_classification.value} (confidence: {result.overall_confidence:.2f})")
