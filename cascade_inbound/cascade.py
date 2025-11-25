#!/usr/bin/env python3
"""
Safety Cascade - Full Pipeline

Cascata Fiduciae Fundata (Cascade of Founded Trust)

L0 → L1 → L2 → L3 with intelligent routing.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from .l0_bouncer import L0Bouncer
from .l1_analyst import L1Analyst
from .l2_gauntlet import L2Gauntlet
from .l3_judge import L3Judge


@dataclass
class CascadeConfig:
    """Configuration for cascade routing."""
    # L0 thresholds
    l0_confidence_threshold: float = 0.7
    l0_safe_threshold: float = 0.6

    # L1 thresholds
    l1_confidence_threshold: float = 0.7

    # Whether to use L2 and L3
    enable_l2: bool = True
    enable_l3: bool = True


@dataclass
class CascadeResult:
    """Result from cascade classification."""
    label: str
    stopped_at: str
    confidence: float
    total_latency_ms: float
    layers: list = field(default_factory=list)
    reasoning: Optional[str] = None
    audit_id: Optional[str] = None


class SafetyCascade:
    """
    Multi-tier safety classification cascade.

    Flow:
    1. L0 Bouncer (fast) - Pass confident safe
    2. L1 Analyst (medium) - Reason through uncertain cases
    3. L2 Gauntlet (slow) - Multi-expert voting for edge cases
    4. L3 Judge (slowest) - Final authority
    """

    def __init__(self, config: CascadeConfig = None):
        self.config = config or CascadeConfig()

        print("=" * 60)
        print("INITIALIZING SAFETY CASCADE")
        print("=" * 60)

        # Load models
        self.l0 = L0Bouncer()
        self.l1 = L1Analyst()

        if self.config.enable_l2:
            self.l2 = L2Gauntlet()
        else:
            self.l2 = None

        if self.config.enable_l3:
            self.l3 = L3Judge()
        else:
            self.l3 = None

        print("=" * 60)
        print("CASCADE READY")
        print("=" * 60)

    def classify(self, text: str) -> CascadeResult:
        """
        Classify text through the cascade.

        Returns CascadeResult with final label and journey details.
        """
        layers = []
        total_start = time.time()

        # === L0: Bouncer ===
        start = time.time()
        l0_result = self.l0.classify(text)
        l0_time = (time.time() - start) * 1000

        layers.append({
            "level": "L0",
            "label": l0_result["label"],
            "confidence": l0_result["confidence"],
            "safe_prob": l0_result["safe_prob"],
            "latency_ms": l0_time,
        })

        # Check if L0 is confident (either direction)
        # High confidence = stop, low confidence = escalate for verification
        if l0_result["confidence"] >= self.config.l0_confidence_threshold:
            # L0 is confident - trust the decision
            return CascadeResult(
                label=l0_result["label"],
                stopped_at="L0",
                confidence=l0_result["confidence"],
                total_latency_ms=(time.time() - total_start) * 1000,
                layers=layers,
            )

        # L0 is uncertain - escalate to L1 for verification

        # === L1: Analyst ===
        start = time.time()
        l1_result = self.l1.analyze(text)
        l1_time = (time.time() - start) * 1000

        layers.append({
            "level": "L1",
            "label": l1_result["label"],
            "confidence": l1_result["confidence"],
            "latency_ms": l1_time,
        })

        # Check if L1 is confident
        if l1_result["label"] in ["safe", "harmful"]:
            if l1_result["confidence"] >= self.config.l1_confidence_threshold:
                return CascadeResult(
                    label=l1_result["label"],
                    stopped_at="L1",
                    confidence=l1_result["confidence"],
                    total_latency_ms=(time.time() - total_start) * 1000,
                    layers=layers,
                    reasoning=l1_result["reasoning"],
                )

        # Skip L2/L3 if not enabled
        if not self.config.enable_l2:
            return CascadeResult(
                label=l1_result["label"],
                stopped_at="L1",
                confidence=l1_result["confidence"],
                total_latency_ms=(time.time() - total_start) * 1000,
                layers=layers,
                reasoning=l1_result["reasoning"],
            )

        # === L2: Gauntlet ===
        start = time.time()
        l2_result = self.l2.analyze(text)
        l2_time = (time.time() - start) * 1000

        layers.append({
            "level": "L2",
            "label": l2_result["label"],
            "votes": l2_result["votes"],
            "consensus": l2_result["consensus"],
            "latency_ms": l2_time,
        })

        # Check for consensus
        if l2_result["consensus"]:
            return CascadeResult(
                label=l2_result["label"],
                stopped_at="L2",
                confidence=0.9 if l2_result["consensus"] else 0.7,
                total_latency_ms=(time.time() - total_start) * 1000,
                layers=layers,
            )

        # Skip L3 if not enabled
        if not self.config.enable_l3:
            return CascadeResult(
                label=l2_result["label"],
                stopped_at="L2",
                confidence=0.7,
                total_latency_ms=(time.time() - total_start) * 1000,
                layers=layers,
            )

        # === L3: Judge ===
        context = {
            "l0": f"{l0_result['label']} ({l0_result['confidence']:.2f})",
            "l1": f"{l1_result['label']} ({l1_result['confidence']:.2f})",
            "l2": f"{l2_result['label']} (votes: {l2_result['votes']})",
        }

        start = time.time()
        l3_result = self.l3.judge(text, context)
        l3_time = (time.time() - start) * 1000

        layers.append({
            "level": "L3",
            "label": l3_result["label"],
            "confidence": l3_result["confidence"],
            "audit_id": l3_result["audit_id"],
            "latency_ms": l3_time,
        })

        return CascadeResult(
            label=l3_result["label"],
            stopped_at="L3",
            confidence=l3_result["confidence"],
            total_latency_ms=(time.time() - total_start) * 1000,
            layers=layers,
            reasoning=l3_result["reasoning"],
            audit_id=l3_result["audit_id"],
        )


def main():
    """Demo the cascade."""
    # Initialize with L2 disabled for faster testing
    config = CascadeConfig(enable_l2=False, enable_l3=False)
    cascade = SafetyCascade(config)

    tests = [
        "What is the capital of France?",
        "How do I make a bomb?",
        "Write a poem about nature",
        "How to hack into someone's computer",
        "Explain photosynthesis",
        "How to pick a lock",
    ]

    print("\n" + "=" * 60)
    print("CASCADE DEMO")
    print("=" * 60)

    for text in tests:
        result = cascade.classify(text)

        print(f"\nInput: {text[:50]}...")
        print(f"  Final: {result.label}")
        print(f"  Stopped at: {result.stopped_at}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Latency: {result.total_latency_ms:.1f}ms")
        print(f"  Journey: {' → '.join(l['level'] for l in result.layers)}")


if __name__ == "__main__":
    main()
