"""
Cascade Orchestrator for DLP.

Multi-stage filtering with early exit and timing instrumentation.
"""

import time
from typing import List, Optional
from dataclasses import dataclass, field

# Import detectors
from detectors.secret_detector import SecretDetector, Detection

# Try Presidio
try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


@dataclass
class StageResult:
    """Result from a cascade stage."""
    stage_name: str
    detections: List[Detection]
    latency_ms: float
    should_block: bool


@dataclass
class CascadeResult:
    """Result from full cascade."""
    input_text: str
    stages: List[StageResult]
    total_detections: List[Detection]
    blocked: bool
    block_reason: Optional[str]
    total_latency_ms: float

    @property
    def detection_count(self) -> int:
        return len(self.total_detections)


class DLPCascade:
    """
    Multi-stage DLP cascade.

    Stage 1: Secret detection (regex, fast)
    Stage 2: PII detection (Presidio NER)
    Stage 3: Policy evaluation
    """

    def __init__(
        self,
        secret_threshold: float = 0.7,
        pii_threshold: float = 0.7,
        block_on_high_confidence: bool = True
    ):
        self.secret_detector = SecretDetector(confidence_threshold=secret_threshold)
        self.pii_threshold = pii_threshold
        self.block_on_high_confidence = block_on_high_confidence

        # Initialize Presidio if available
        if PRESIDIO_AVAILABLE:
            self.presidio = AnalyzerEngine()
        else:
            self.presidio = None

    def run(self, text: str) -> CascadeResult:
        """
        Run text through the full cascade.

        Returns CascadeResult with all detections and timing.
        """
        stages = []
        all_detections = []
        blocked = False
        block_reason = None
        start_time = time.time()

        # Stage 1: Secret Detection (fast regex)
        stage1_start = time.time()
        secret_detections = self.secret_detector.detect(text)
        stage1_time = (time.time() - stage1_start) * 1000

        # Convert to common Detection format
        stage1_detections = secret_detections
        all_detections.extend(stage1_detections)

        # Check for high-confidence secrets (immediate block)
        high_conf_secrets = [d for d in stage1_detections if d.confidence >= 0.9]
        stage1_block = len(high_conf_secrets) > 0 and self.block_on_high_confidence

        if stage1_block:
            blocked = True
            block_reason = f"High-confidence secret detected: {high_conf_secrets[0].entity_type}"

        stages.append(StageResult(
            stage_name="SecretDetector",
            detections=stage1_detections,
            latency_ms=stage1_time,
            should_block=stage1_block
        ))

        # Stage 2: PII Detection (Presidio)
        if self.presidio and not blocked:
            stage2_start = time.time()
            presidio_results = self.presidio.analyze(text=text, language="en")
            stage2_time = (time.time() - stage2_start) * 1000

            # Convert Presidio results to Detection format
            stage2_detections = []
            for r in presidio_results:
                if r.score >= self.pii_threshold:
                    detection = Detection(
                        detector_name="Presidio",
                        entity_type=r.entity_type,
                        text=text[r.start:r.end],
                        start=r.start,
                        end=r.end,
                        confidence=r.score,
                        metadata={"recognizer": r.recognition_metadata}
                    )
                    stage2_detections.append(detection)

            all_detections.extend(stage2_detections)

            # Check for high-confidence PII
            high_conf_pii = [d for d in stage2_detections if d.confidence >= 0.9]
            stage2_block = len(high_conf_pii) > 0 and self.block_on_high_confidence

            if stage2_block and not blocked:
                blocked = True
                block_reason = f"High-confidence PII detected: {high_conf_pii[0].entity_type}"

            stages.append(StageResult(
                stage_name="Presidio",
                detections=stage2_detections,
                latency_ms=stage2_time,
                should_block=stage2_block
            ))

        # Stage 3: Policy evaluation (placeholder for now)
        # This would check provenance, allowed readers, etc.

        total_time = (time.time() - start_time) * 1000

        return CascadeResult(
            input_text=text,
            stages=stages,
            total_detections=all_detections,
            blocked=blocked,
            block_reason=block_reason,
            total_latency_ms=total_time
        )


def main():
    """Test the DLP cascade."""
    cascade = DLPCascade()

    test_cases = [
        # Should block - secrets
        ("My AWS key is AKIAIOSFODNN7EXAMPLE", "AWS key"),
        ("ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "GitHub token"),
        ("postgres://admin:secretpass@db.example.com:5432/prod", "DB connection"),

        # Should detect - PII
        ("Contact John Smith at john.smith@example.com", "PII"),
        ("Call me at 555-123-4567", "Phone"),
        ("SSN: 123-45-6789", "SSN"),

        # Should pass - clean
        ("Hello, this is a normal message.", "Clean"),
        ("The weather today is sunny.", "Clean"),
    ]

    print("=" * 70)
    print("DLP CASCADE TEST")
    print("=" * 70)

    for text, description in test_cases:
        result = cascade.run(text)

        print(f"\n{'─' * 70}")
        print(f"Input ({description}): {text[:50]}...")
        print(f"{'─' * 70}")

        print(f"\nBlocked: {'YES - ' + result.block_reason if result.blocked else 'No'}")
        print(f"Total detections: {result.detection_count}")
        print(f"Total latency: {result.total_latency_ms:.1f}ms")

        print(f"\nStages:")
        for stage in result.stages:
            status = "🚫 BLOCK" if stage.should_block else "✓"
            print(f"  {stage.stage_name}: {len(stage.detections)} detections, {stage.latency_ms:.1f}ms {status}")

            for d in stage.detections[:3]:  # Show first 3
                print(f"    • {d.entity_type}: '{d.text[:30]}' ({d.confidence:.0%})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    blocked_count = sum(1 for text, _ in test_cases if cascade.run(text).blocked)
    print(f"\nBlocked: {blocked_count}/{len(test_cases)}")
    print(f"Passed: {len(test_cases) - blocked_count}/{len(test_cases)}")


if __name__ == "__main__":
    main()
