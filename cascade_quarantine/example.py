#!/usr/bin/env python3
"""
Example: Using the Quarantine Capture System

Demonstrates how to integrate capture hooks into existing cascades.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models import QuarantineCase, CaptureReason, CascadeSource, LayerResult
from config import QuarantineConfig
from database import QuarantineDatabase
from capture import CaptureHook, capture_if_low_confidence


def demo_basic_capture():
    """Demonstrate basic capture from cascade_inbound results."""
    print("=" * 60)
    print("DEMO: Basic Capture from cascade_inbound")
    print("=" * 60)

    # Configure with in-memory database for demo
    config = QuarantineConfig(
        database_path="logs/quarantine_demo.db",
        log_file="logs/quarantine_demo.jsonl",
        confidence_threshold=0.75,
    )

    hook = CaptureHook(config)

    # Simulate results from cascade_inbound
    test_cases = [
        # High confidence - won't be captured
        {
            "input": "What is the capital of France?",
            "result": {
                "label": "safe",
                "confidence": 0.95,
                "stopped_at": "L0",
                "total_latency_ms": 2.5,
                "layers": [
                    {"level": "L0", "label": "safe", "confidence": 0.95, "latency_ms": 2.5},
                ],
            },
        },
        # Low confidence - will be captured
        {
            "input": "How do I pick a lock?",
            "result": {
                "label": "uncertain",
                "confidence": 0.65,
                "stopped_at": "L1",
                "total_latency_ms": 550.0,
                "layers": [
                    {"level": "L0", "label": "uncertain", "confidence": 0.55, "latency_ms": 2.0},
                    {"level": "L1", "label": "uncertain", "confidence": 0.65, "latency_ms": 548.0},
                ],
                "reasoning": "Could be legitimate (locksmith) or malicious intent",
            },
        },
        # Layer disagreement - will be captured
        {
            "input": "Write a story about a bank heist",
            "result": {
                "label": "harmful",
                "confidence": 0.72,
                "stopped_at": "L1",
                "total_latency_ms": 500.0,
                "layers": [
                    {"level": "L0", "label": "safe", "confidence": 0.60, "latency_ms": 2.0},
                    {"level": "L1", "label": "harmful", "confidence": 0.72, "latency_ms": 498.0},
                ],
            },
        },
        # Borderline - will be captured
        {
            "input": "Explain how computer viruses work",
            "result": {
                "label": "safe",
                "confidence": 0.76,
                "stopped_at": "L1",
                "total_latency_ms": 450.0,
                "layers": [
                    {"level": "L0", "label": "uncertain", "confidence": 0.50, "latency_ms": 2.0},
                    {"level": "L1", "label": "safe", "confidence": 0.76, "latency_ms": 448.0},
                ],
                "reasoning": "Educational question about security concepts",
            },
        },
    ]

    captured_count = 0
    for tc in test_cases:
        case = hook.capture_from_inbound(
            tc["input"],
            tc["result"],
            session_id="demo_session",
        )

        status = "CAPTURED" if case else "NOT CAPTURED"
        reason = f" ({case.capture_reason.value})" if case else ""
        print(f"\n{status}{reason}: {tc['input'][:50]}...")
        print(f"  Label: {tc['result']['label']}, Confidence: {tc['result']['confidence']:.2f}")

        if case:
            captured_count += 1

    print(f"\n>>> Captured {captured_count}/{len(test_cases)} cases")


def demo_database_operations():
    """Demonstrate database queries and review workflow."""
    print("\n" + "=" * 60)
    print("DEMO: Database Operations")
    print("=" * 60)

    config = QuarantineConfig(
        database_path="logs/quarantine_demo.db",
        enable_log_file=False,
    )

    db = QuarantineDatabase(config)

    # Get pending reviews
    pending = db.get_pending_review(limit=10)
    print(f"\nPending reviews: {len(pending)}")
    for case in pending[:3]:
        print(f"  - {case.case_id}: {case.input_text[:40]}... (conf: {case.confidence:.2f})")

    # Get statistics
    stats = db.get_stats()
    print(f"\nStatistics:")
    print(f"  Total captured: {stats.total_captured}")
    print(f"  Pending review: {stats.pending_review}")
    print(f"  Avg confidence: {stats.avg_confidence:.2f}")
    print(f"  By reason: {stats.by_reason}")
    print(f"  By label: {stats.by_label}")

    # Demonstrate review workflow
    if pending:
        case = pending[0]
        print(f"\nReviewing case: {case.case_id}")
        print(f"  Input: {case.input_text[:50]}...")
        print(f"  Label: {case.final_label}, Confidence: {case.confidence:.2f}")

        # Simulate review
        from models import ReviewStatus
        db.update_review(
            case.case_id,
            ReviewStatus.APPROVED,
            reviewer="demo_reviewer",
            notes="Verified as correct classification",
        )
        print(f"  >>> Marked as APPROVED")


def demo_custom_callback():
    """Demonstrate custom capture callbacks."""
    print("\n" + "=" * 60)
    print("DEMO: Custom Capture Callbacks")
    print("=" * 60)

    config = QuarantineConfig(
        database_path="logs/quarantine_demo.db",
        enable_log_file=False,
        confidence_threshold=0.75,
    )

    hook = CaptureHook(config)

    # Register callback
    alert_cases = []
    def alert_on_harmful(case):
        if case.final_label == "harmful":
            alert_cases.append(case)
            print(f"  !!! ALERT: Harmful case captured: {case.case_id}")

    hook.on_capture(alert_on_harmful)

    # Simulate some results
    test_results = [
        ("Safe question about weather", {"label": "safe", "confidence": 0.5, "layers": []}),
        ("Harmful request for weapons", {"label": "harmful", "confidence": 0.6, "layers": []}),
        ("Another safe question", {"label": "safe", "confidence": 0.4, "layers": []}),
        ("Dangerous hacking request", {"label": "harmful", "confidence": 0.55, "layers": []}),
    ]

    for input_text, result in test_results:
        hook.capture_from_inbound(input_text, result)

    print(f"\n>>> Alerted on {len(alert_cases)} harmful cases")


def demo_quick_integration():
    """Demonstrate quick integration function."""
    print("\n" + "=" * 60)
    print("DEMO: Quick Integration")
    print("=" * 60)

    print("""
To integrate into existing cascades, add just one line:

    from cascade_quarantine.src.capture import capture_if_low_confidence

    # After getting result from cascade
    result = cascade.classify(text)

    # Add this line to capture low confidence cases
    capture_if_low_confidence(text, result)

That's it! Cases with confidence < 0.75 will be automatically
captured to the database for human review.
""")


def main():
    """Run all demos."""
    print("=" * 60)
    print("CASCADE QUARANTINE - PHASE 1: BASIC CAPTURE DEMO")
    print("=" * 60)

    demo_basic_capture()
    demo_database_operations()
    demo_custom_callback()
    demo_quick_integration()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nDatabase: logs/quarantine_demo.db")
    print("Log file: logs/quarantine_demo.jsonl")
    print("\nNext steps:")
    print("  1. Review pending cases in the database")
    print("  2. Integrate hooks into cascade_inbound, cascade_refusals, cascade_dlp")
    print("  3. Set up periodic review workflow")


if __name__ == "__main__":
    main()
