"""
Evaluation tests for cascade_dlp.

Tests the full pipeline: detection → policy → redaction → response.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from router import OutboundRouter
from context import (
    create_external_context,
    create_internal_context,
    create_admin_context,
    RequestContext,
    Destination,
    UserRole,
)
from policies import Action
from redactor import Redactor, RedactionStrategy
from detectors.secret_detector import SecretDetector


def test_secret_detection():
    """Test that all secret patterns are detected."""
    detector = SecretDetector()

    test_cases = [
        # (input, expected_type, should_detect)
        ("AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY", True),
        ("ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "GITHUB_TOKEN", True),
        ("sk_live_EXAMPLE_KEY_HERE_REPLACE", "STRIPE_KEY", True),
        ("postgres://user:pass@host:5432/db", "POSTGRES_URI", True),
        ("-----BEGIN RSA PRIVATE KEY-----\ndata\n-----END RSA PRIVATE KEY-----", "RSA_PRIVATE_KEY", True),
        ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.xxx", "JWT", True),
        ("Hello world", None, False),
        ("Just some normal text", None, False),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("SECRET DETECTION TESTS")
    print("=" * 60)

    for text, expected_type, should_detect in test_cases:
        detections = detector.detect(text)
        detected = len(detections) > 0

        if should_detect:
            if detected:
                found_types = [d.entity_type for d in detections]
                if expected_type in found_types:
                    print(f"✓ {expected_type}: detected")
                    passed += 1
                else:
                    print(f"✗ {expected_type}: wrong type (got {found_types})")
                    failed += 1
            else:
                print(f"✗ {expected_type}: not detected")
                failed += 1
        else:
            if not detected:
                print(f"✓ Clean text: no detection")
                passed += 1
            else:
                print(f"✗ Clean text: false positive ({[d.entity_type for d in detections]})")
                failed += 1

    print(f"\nPassed: {passed}/{passed+failed}")
    return failed == 0


def test_policy_matrix():
    """Test policy decisions for different contexts."""
    router = OutboundRouter()

    # Test matrix: (text, context_factory, expected_action)
    test_cases = [
        # Credentials - external should block
        ("AWS key: AKIAIOSFODNN7EXAMPLE", create_external_context, Action.BLOCK),
        ("Token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", create_external_context, Action.BLOCK),

        # Credentials - internal should allow
        ("AWS key: AKIAIOSFODNN7EXAMPLE", create_internal_context, Action.ALLOW),
        ("Token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", create_internal_context, Action.ALLOW),

        # Credentials - admin should allow
        ("AWS key: AKIAIOSFODNN7EXAMPLE", create_admin_context, Action.ALLOW),

        # PII - external should redact
        ("Email: john@example.com", create_external_context, Action.REDACT),
        ("Name: John Smith", create_external_context, Action.REDACT),

        # PII - internal should allow
        ("Email: john@example.com", create_internal_context, Action.ALLOW),

        # Clean - all should allow
        ("Hello world", create_external_context, Action.ALLOW),
        ("Normal text", create_internal_context, Action.ALLOW),
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("POLICY MATRIX TESTS")
    print("=" * 60)

    for text, context_factory, expected_action in test_cases:
        context = context_factory("test_user", "test_req")
        result = router.process(text, context)

        context_name = {
            Destination.EXTERNAL: "External",
            Destination.INTERNAL: "Internal",
        }.get(context.destination, "Unknown")

        if context.user_role == UserRole.ADMIN:
            context_name = "Admin"

        if result.action == expected_action:
            print(f"✓ {context_name}: {text[:30]}... → {result.action.value}")
            passed += 1
        else:
            print(f"✗ {context_name}: {text[:30]}... → {result.action.value} (expected {expected_action.value})")
            failed += 1

    print(f"\nPassed: {passed}/{passed+failed}")
    return failed == 0


def test_redaction():
    """Test redaction strategies."""

    test_cases = [
        # (strategy, text, detections_info, expected_contains)
        (RedactionStrategy.TYPE_AWARE, "Email: john@example.com",
         [("EMAIL_ADDRESS", "john@example.com", 7, 23)], "[EMAIL_ADDRESS]"),
        (RedactionStrategy.GENERIC, "SSN: 123-45-6789",
         [("US_SSN", "123-45-6789", 5, 16)], "[REDACTED]"),
        (RedactionStrategy.PARTIAL, "Card: 4532-1234-5678-9012",
         [("CREDIT_CARD", "4532-1234-5678-9012", 6, 25)], "9012"),
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("REDACTION TESTS")
    print("=" * 60)

    from detectors.secret_detector import Detection

    for strategy, text, detection_info, expected_contains in test_cases:
        redactor = Redactor(strategy=strategy)

        detections = []
        for entity_type, matched_text, start, end in detection_info:
            detections.append(Detection(
                detector_name="test",
                entity_type=entity_type,
                text=matched_text,
                start=start,
                end=end,
                confidence=1.0,
                metadata={}
            ))

        result = redactor.redact(text, detections)

        if expected_contains in result:
            print(f"✓ {strategy.value}: {result}")
            passed += 1
        else:
            print(f"✗ {strategy.value}: {result} (expected to contain '{expected_contains}')")
            failed += 1

    print(f"\nPassed: {passed}/{passed+failed}")
    return failed == 0


def test_false_positives():
    """Test that false positives are minimized."""
    router = OutboundRouter()
    context = create_external_context("test_user")

    # These should NOT be blocked
    false_positive_cases = [
        "Visit example.com for more info",
        "Use YOUR_API_KEY_HERE as a placeholder",
        "The function returns error code 123",
        "AWS_ACCESS_KEY_ID is an environment variable name",
        "Call 1-800-FLOWERS for delivery",
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("FALSE POSITIVE TESTS")
    print("=" * 60)

    for text in false_positive_cases:
        result = router.process(text, context)

        # These should ideally be ALLOW, but REDACT is acceptable
        # BLOCK would be a false positive
        if result.action != Action.BLOCK:
            print(f"✓ {text[:40]}... → {result.action.value}")
            passed += 1
        else:
            print(f"✗ {text[:40]}... → BLOCKED (false positive)")
            failed += 1

    print(f"\nPassed: {passed}/{passed+failed}")
    return failed == 0


def test_performance():
    """Test latency performance."""
    import time

    router = OutboundRouter()
    context = create_external_context("test_user")

    # Test cases of varying complexity
    test_cases = [
        "Short text",
        "Email: john@example.com and phone: 555-123-4567",
        "AWS: AKIAIOSFODNN7EXAMPLE, Token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, " * 5,
        "Normal text without any sensitive data. " * 100,
    ]

    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)

    all_latencies = []

    for text in test_cases:
        latencies = []
        for _ in range(10):
            start = time.time()
            router.process(text, context)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[5]
        p99 = sorted(latencies)[9]
        all_latencies.extend(latencies)

        print(f"Text ({len(text)} chars): avg={avg:.1f}ms, p50={p50:.1f}ms, p99={p99:.1f}ms")

    overall_avg = sum(all_latencies) / len(all_latencies)
    overall_p99 = sorted(all_latencies)[int(len(all_latencies) * 0.99)]

    print(f"\nOverall: avg={overall_avg:.1f}ms, p99={overall_p99:.1f}ms")

    # Pass if p99 < 100ms
    passed = overall_p99 < 100
    if passed:
        print("✓ Performance target met (p99 < 100ms)")
    else:
        print("✗ Performance target missed (p99 >= 100ms)")

    return passed


def test_full_pipeline():
    """End-to-end pipeline test."""
    from audit import AuditLogger

    router = OutboundRouter()
    logger = AuditLogger()

    test_cases = [
        # (description, text, context_factory, expected_action, expected_blocked)
        ("Block external credential",
         "My AWS key is AKIAIOSFODNN7EXAMPLE",
         create_external_context, Action.BLOCK, True),
        ("Redact external PII",
         "Contact john@example.com",
         create_external_context, Action.REDACT, False),
        ("Allow internal credential",
         "Key: AKIAIOSFODNN7EXAMPLE",
         create_internal_context, Action.ALLOW, False),
        ("Allow clean text",
         "Hello world",
         create_external_context, Action.ALLOW, False),
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("FULL PIPELINE TESTS")
    print("=" * 60)

    for desc, text, context_factory, expected_action, expected_blocked in test_cases:
        context = context_factory("test_user", "test_req")
        result = router.process(text, context)
        entry = logger.log(result, context)

        action_ok = result.action == expected_action
        blocked_ok = result.blocked == expected_blocked
        audit_ok = entry.action == expected_action.value

        if action_ok and blocked_ok and audit_ok:
            print(f"✓ {desc}")
            passed += 1
        else:
            issues = []
            if not action_ok:
                issues.append(f"action={result.action.value}")
            if not blocked_ok:
                issues.append(f"blocked={result.blocked}")
            if not audit_ok:
                issues.append(f"audit={entry.action}")
            print(f"✗ {desc}: {', '.join(issues)}")
            failed += 1

    # Check audit summary
    summary = logger.get_summary()
    print(f"\nAudit Summary: {summary['total']} total, {summary['blocked']} blocked, {summary['redacted']} redacted")

    print(f"\nPassed: {passed}/{passed+failed}")
    return failed == 0


def main():
    """Run all tests."""
    print("=" * 60)
    print("CASCADE DLP EVALUATION TESTS")
    print("=" * 60)

    results = {
        "Secret Detection": test_secret_detection(),
        "Policy Matrix": test_policy_matrix(),
        "Redaction": test_redaction(),
        "False Positives": test_false_positives(),
        "Performance": test_performance(),
        "Full Pipeline": test_full_pipeline(),
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
