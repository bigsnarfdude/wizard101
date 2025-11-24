"""
Redactor for cascade_dlp.

Masks detected sensitive data in text with configurable strategies.
"""

from typing import List
from enum import Enum
from detectors.secret_detector import Detection


class RedactionStrategy(Enum):
    """Redaction mask strategies."""
    GENERIC = "generic"      # [REDACTED]
    TYPE_AWARE = "type"      # [EMAIL], [SSN]
    PARTIAL = "partial"      # j***@***.com
    HASH = "hash"            # [sha256:abc123...]


class Redactor:
    """Redact sensitive data from text."""

    def __init__(self, strategy: RedactionStrategy = RedactionStrategy.TYPE_AWARE):
        self.strategy = strategy

    def redact(self, text: str, detections: List[Detection]) -> str:
        """
        Replace detected spans with redaction masks.

        Args:
            text: Original text
            detections: List of Detection objects with start/end positions

        Returns:
            Redacted text
        """
        if not detections:
            return text

        # Sort detections by start position (reverse to preserve indices)
        sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)

        result = text
        for detection in sorted_detections:
            mask = self._get_mask(detection)
            result = result[:detection.start] + mask + result[detection.end:]

        return result

    def _get_mask(self, detection: Detection) -> str:
        """Get redaction mask based on strategy."""
        if self.strategy == RedactionStrategy.GENERIC:
            return "[REDACTED]"

        elif self.strategy == RedactionStrategy.TYPE_AWARE:
            # Use entity type as mask
            entity = detection.entity_type.upper()
            return f"[{entity}]"

        elif self.strategy == RedactionStrategy.PARTIAL:
            # Partial mask preserving format
            return self._partial_mask(detection.text, detection.entity_type)

        elif self.strategy == RedactionStrategy.HASH:
            # Hash for correlation without exposure
            import hashlib
            hash_val = hashlib.sha256(detection.text.encode()).hexdigest()[:12]
            return f"[hash:{hash_val}]"

        return "[REDACTED]"

    def _partial_mask(self, text: str, entity_type: str) -> str:
        """Create partial mask preserving some format."""
        if "EMAIL" in entity_type:
            # j***@***.com
            if "@" in text:
                local, domain = text.split("@", 1)
                return f"{local[0]}***@***.{domain.split('.')[-1]}"
            return text[0] + "***"

        elif "PHONE" in entity_type:
            # ***-***-1234
            if len(text) >= 4:
                return "***-***-" + text[-4:]
            return "***"

        elif "SSN" in entity_type:
            # ***-**-6789
            if len(text) >= 4:
                return "***-**-" + text[-4:]
            return "***"

        elif "CREDIT" in entity_type:
            # ****-****-****-1234
            if len(text) >= 4:
                return "****-****-****-" + text[-4:]
            return "****"

        else:
            # Generic partial: first char + *** + last char
            if len(text) > 2:
                return text[0] + "***" + text[-1]
            return "***"

    def redact_for_display(self, text: str, detections: List[Detection]) -> str:
        """
        Redact with visual indicators for debugging.

        Returns text with detections highlighted.
        """
        if not detections:
            return text

        sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)

        result = text
        for detection in sorted_detections:
            mask = f"<<{detection.entity_type}:{detection.text[:10]}...>>"
            result = result[:detection.start] + mask + result[detection.end:]

        return result


def main():
    """Test redactor."""
    from detectors.secret_detector import SecretDetector

    detector = SecretDetector()
    redactor = Redactor(strategy=RedactionStrategy.TYPE_AWARE)

    test_cases = [
        "Contact john@example.com for help",
        "AWS key: AKIAIOSFODNN7EXAMPLE",
        "Database: postgres://admin:secret@db.com:5432/prod",
        "Token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    ]

    print("=" * 60)
    print("REDACTOR TEST")
    print("=" * 60)

    for text in test_cases:
        detections = detector.detect(text)
        redacted = redactor.redact(text, detections)

        print(f"\nOriginal: {text}")
        print(f"Redacted: {redacted}")
        print(f"Detections: {len(detections)}")

    # Test different strategies
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)

    text = "Email: john@company.com, SSN: 123-45-6789"

    # Create fake detections for testing
    detections = [
        Detection(
            detector_name="test",
            entity_type="EMAIL_ADDRESS",
            text="john@company.com",
            start=7,
            end=23,
            confidence=1.0,
            metadata={}
        ),
        Detection(
            detector_name="test",
            entity_type="US_SSN",
            text="123-45-6789",
            start=30,
            end=41,
            confidence=1.0,
            metadata={}
        ),
    ]

    for strategy in RedactionStrategy:
        r = Redactor(strategy=strategy)
        result = r.redact(text, detections)
        print(f"\n{strategy.value:12}: {result}")


if __name__ == "__main__":
    main()
