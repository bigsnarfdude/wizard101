"""
Response router for cascade_dlp.

Wires together detection → policy → redaction → response.
"""

import time
from typing import List, Optional
from dataclasses import dataclass, field

from cascade import DLPCascade
from context import RequestContext, create_external_context
from policies import PolicyEngine, PolicyDecision, Action
from redactor import Redactor, RedactionStrategy
from detectors.secret_detector import Detection


@dataclass
class RouterResult:
    """Result from the full DLP pipeline."""
    # Original and processed text
    original_text: str
    processed_text: str

    # Overall action taken
    action: Action

    # Detections and decisions
    detections: List[Detection]
    decisions: List[PolicyDecision]

    # Timing
    detection_time_ms: float
    policy_time_ms: float
    redaction_time_ms: float
    total_time_ms: float

    # Metadata
    blocked: bool
    redacted: bool
    block_reason: Optional[str] = None

    @property
    def detection_count(self) -> int:
        return len(self.detections)


class OutboundRouter:
    """
    Full outbound DLP pipeline.

    Detect → Policy → Redact → Respond
    """

    # Standard block messages
    BLOCK_MESSAGES = {
        "credential": "I cannot share credentials or API keys.",
        "private_key": "I cannot share private keys.",
        "pii": "I've removed sensitive personal information from my response.",
        "default": "I've removed some sensitive information from my response.",
    }

    def __init__(
        self,
        secret_threshold: float = 0.7,
        pii_threshold: float = 0.7,
        redaction_strategy: RedactionStrategy = RedactionStrategy.TYPE_AWARE
    ):
        self.cascade = DLPCascade(
            secret_threshold=secret_threshold,
            pii_threshold=pii_threshold,
            block_on_high_confidence=False  # We handle blocking via policy
        )
        self.policy_engine = PolicyEngine()
        self.redactor = Redactor(strategy=redaction_strategy)

    def process(
        self,
        text: str,
        context: Optional[RequestContext] = None
    ) -> RouterResult:
        """
        Process text through the full DLP pipeline.

        Args:
            text: Model response to process
            context: Request context for policy decisions

        Returns:
            RouterResult with processed text and metadata
        """
        if context is None:
            context = create_external_context()

        start_time = time.time()

        # Step 1: Detect
        detect_start = time.time()
        cascade_result = self.cascade.run(text)
        detections = cascade_result.total_detections
        detection_time = (time.time() - detect_start) * 1000

        # Step 2: Policy evaluation
        policy_start = time.time()
        decisions = self.policy_engine.evaluate_all(detections, context)
        policy_time = (time.time() - policy_start) * 1000

        # Step 3: Determine overall action
        overall_action, block_reason = self._determine_overall_action(decisions)

        # Step 4: Apply action
        redact_start = time.time()
        if overall_action == Action.BLOCK:
            processed_text = self._get_block_message(decisions)
            blocked = True
            redacted = False
        elif overall_action == Action.REDACT:
            # Get detections that need redaction
            redact_detections = [
                d.detection for d in decisions
                if d.action == Action.REDACT
            ]
            processed_text = self.redactor.redact(text, redact_detections)
            blocked = False
            redacted = True
        else:
            processed_text = text
            blocked = False
            redacted = False

        redaction_time = (time.time() - redact_start) * 1000
        total_time = (time.time() - start_time) * 1000

        return RouterResult(
            original_text=text,
            processed_text=processed_text,
            action=overall_action,
            detections=detections,
            decisions=decisions,
            detection_time_ms=detection_time,
            policy_time_ms=policy_time,
            redaction_time_ms=redaction_time,
            total_time_ms=total_time,
            blocked=blocked,
            redacted=redacted,
            block_reason=block_reason,
        )

    def _determine_overall_action(
        self,
        decisions: List[PolicyDecision]
    ) -> tuple[Action, Optional[str]]:
        """Determine the overall action from all decisions."""
        if not decisions:
            return Action.ALLOW, None

        # Priority: BLOCK > ALERT > REVIEW > REDACT > ALLOW
        has_block = any(d.action == Action.BLOCK for d in decisions)
        has_alert = any(d.action == Action.ALERT for d in decisions)
        has_redact = any(d.action == Action.REDACT for d in decisions)

        if has_block:
            block_decision = next(d for d in decisions if d.action == Action.BLOCK)
            return Action.BLOCK, block_decision.reason

        if has_alert:
            return Action.ALERT, "Alert triggered"

        if has_redact:
            return Action.REDACT, None

        return Action.ALLOW, None

    def _get_block_message(self, decisions: List[PolicyDecision]) -> str:
        """Get appropriate block message based on what was blocked."""
        block_decisions = [d for d in decisions if d.action == Action.BLOCK]

        if not block_decisions:
            return self.BLOCK_MESSAGES["default"]

        # Check what type of detection triggered block
        entity_type = block_decisions[0].detection.entity_type.upper()

        if "KEY" in entity_type or "TOKEN" in entity_type or "SECRET" in entity_type:
            return self.BLOCK_MESSAGES["credential"]
        elif "PRIVATE" in entity_type:
            return self.BLOCK_MESSAGES["private_key"]
        elif entity_type in ("PERSON", "EMAIL", "PHONE", "SSN"):
            return self.BLOCK_MESSAGES["pii"]

        return self.BLOCK_MESSAGES["default"]


def main():
    """Test the full router."""
    from context import create_external_context, create_internal_context, create_admin_context

    router = OutboundRouter()

    test_cases = [
        # Should block - credentials
        "Here's my AWS key: AKIAIOSFODNN7EXAMPLE",

        # Should redact - PII in external
        "Contact John Smith at john@example.com or call 555-123-4567",

        # Should allow - clean text
        "The weather today is sunny with a high of 75 degrees.",

        # Should block - multiple sensitive items
        "DB connection: postgres://admin:secret@db.com:5432/prod, token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    ]

    contexts = [
        ("External", create_external_context("user123")),
        ("Internal", create_internal_context("staff456")),
        ("Admin", create_admin_context("admin789")),
    ]

    print("=" * 80)
    print("OUTBOUND ROUTER TEST")
    print("=" * 80)

    for text in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Input: {text[:60]}...")
        print(f"{'─' * 80}")

        for context_name, context in contexts:
            result = router.process(text, context)

            action_icon = {
                Action.ALLOW: "✓",
                Action.BLOCK: "🚫",
                Action.REDACT: "✂️",
                Action.ALERT: "⚠️",
            }.get(result.action, "?")

            print(f"\n  {context_name}:")
            print(f"    Action: {action_icon} {result.action.value}")
            print(f"    Detections: {result.detection_count}")
            print(f"    Latency: {result.total_time_ms:.1f}ms")

            if result.blocked:
                print(f"    Output: {result.processed_text}")
            elif result.redacted:
                print(f"    Output: {result.processed_text[:60]}...")
            else:
                print(f"    Output: [unchanged]")

    # Summary
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
