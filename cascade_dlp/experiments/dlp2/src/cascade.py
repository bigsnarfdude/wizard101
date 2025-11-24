"""
Cascade DLP v2 - GLiNER2-based PII detection pipeline.
"""

import re
import hashlib
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from gliner2 import GLiNER2
from . import config


@dataclass
class Detection:
    """PII detection result."""
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    stage: str  # "secrets" or "gliner2"


@dataclass
class ProcessResult:
    """Result from processing text through the pipeline."""
    original_text: str
    redacted_text: str
    detections: List[Detection]
    action: str  # "ALLOW", "REDACT", "BLOCK"
    processing_time_ms: float
    audit_id: str


class DLPCascade:
    """
    DLP pipeline using GLiNER2 for semantic PII detection.

    Architecture:
        Stage 0: Secret patterns (regex) - <1ms
        Stage 1: GLiNER2 (semantic) - ~80ms
        Stage 2: Policy engine - <1ms
        Stage 3: Redactor - <1ms
        Stage 4: Audit - <1ms
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        pii_types: List[str] = None,
    ):
        """
        Initialize the DLP cascade.

        Args:
            model_name: GLiNER2 model to use
            pii_types: List of PII types to detect
        """
        self.pii_types = pii_types or config.PII_TYPES

        # Compile secret patterns
        self.secret_patterns = {}
        for name, cfg in config.SECRET_PATTERNS.items():
            self.secret_patterns[name] = {
                "regex": re.compile(cfg["pattern"]),
                "confidence": cfg["confidence"],
            }

        # Load GLiNER2 model
        print(f"Loading GLiNER2 model: {model_name}")
        self.model = GLiNER2.from_pretrained(model_name)
        print(f"Configured {len(self.pii_types)} PII types")

    def process(self, text: str) -> ProcessResult:
        """
        Process text through the full DLP pipeline.

        Args:
            text: Input text to scan

        Returns:
            ProcessResult with detections and redacted text
        """
        import time
        start_time = time.time()

        detections = []

        # Stage 0: Secret patterns (deterministic)
        detections.extend(self._detect_secrets(text))

        # Stage 1: GLiNER2 (semantic)
        detections.extend(self._detect_gliner2(text))

        # Deduplicate overlapping detections
        detections = self._deduplicate(detections)

        # Stage 2: Policy
        action = self._apply_policy(detections)

        # Stage 3: Redact
        if action == "BLOCK":
            redacted_text = "[BLOCKED - SENSITIVE CONTENT DETECTED]"
        else:
            redacted_text = self._redact(text, detections)

        # Stage 4: Audit
        audit_id = self._audit(text, detections, action)

        processing_time = (time.time() - start_time) * 1000

        return ProcessResult(
            original_text=text,
            redacted_text=redacted_text,
            detections=detections,
            action=action,
            processing_time_ms=processing_time,
            audit_id=audit_id,
        )

    def _detect_secrets(self, text: str) -> List[Detection]:
        """Stage 0: Detect secrets using regex patterns."""
        detections = []

        for name, cfg in self.secret_patterns.items():
            for match in cfg["regex"].finditer(text):
                detection = Detection(
                    entity_type=name,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=cfg["confidence"],
                    stage="secrets",
                )
                detections.append(detection)

        return detections

    def _detect_gliner2(self, text: str) -> List[Detection]:
        """Stage 1: Detect PII using GLiNER2."""
        detections = []

        try:
            entities = self.model.extract_entities(text, self.pii_types)

            for entity in entities:
                detection = Detection(
                    entity_type=entity["label"],
                    text=entity["text"],
                    start=entity["start"],
                    end=entity["end"],
                    confidence=entity.get("score", 0.9),
                    stage="gliner2",
                )
                detections.append(detection)

        except Exception as e:
            print(f"GLiNER2 detection error: {e}")

        return detections

    def _deduplicate(self, detections: List[Detection]) -> List[Detection]:
        """Remove overlapping detections, prefer higher confidence."""
        if not detections:
            return []

        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda d: -d.confidence)

        # Keep non-overlapping
        result = []
        for detection in sorted_detections:
            overlaps = False
            for kept in result:
                if (detection.start < kept.end and detection.end > kept.start):
                    overlaps = True
                    break
            if not overlaps:
                result.append(detection)

        # Sort by position
        return sorted(result, key=lambda d: d.start)

    def _apply_policy(self, detections: List[Detection]) -> str:
        """Stage 2: Apply policy rules to determine action."""
        if not detections:
            return "ALLOW"

        for detection in detections:
            entity_type = detection.entity_type.lower()
            action = config.POLICY_ACTIONS.get(entity_type, config.DEFAULT_ACTION)

            if action == "BLOCK":
                return "BLOCK"

        return "REDACT"

    def _redact(self, text: str, detections: List[Detection]) -> str:
        """Stage 3: Redact detected PII from text."""
        if not detections:
            return text

        # Sort by position (descending) to replace from end
        sorted_detections = sorted(detections, key=lambda d: -d.start)

        result = text
        for detection in sorted_detections:
            if config.REDACTION_STYLE == "type":
                replacement = f"[{detection.entity_type.upper().replace(' ', '_')}]"
            elif config.REDACTION_STYLE == "hash":
                hash_val = hashlib.sha256(detection.text.encode()).hexdigest()[:8]
                replacement = f"[{hash_val}]"
            elif config.REDACTION_STYLE == "partial":
                if len(detection.text) > 4:
                    replacement = detection.text[:2] + "*" * (len(detection.text) - 4) + detection.text[-2:]
                else:
                    replacement = "*" * len(detection.text)
            else:  # generic
                replacement = "[REDACTED]"

            result = result[:detection.start] + replacement + result[detection.end:]

        return result

    def _audit(self, text: str, detections: List[Detection], action: str) -> str:
        """Stage 4: Create audit log entry."""
        audit_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}{text[:100]}".encode()
        ).hexdigest()[:16]

        # In production, write to audit log
        # For now, just return the ID
        return audit_id


def main():
    """Example usage."""
    dlp = DLPCascade()

    test_cases = [
        "My password is E5_N8G2xW and email is john@example.com",
        "Call me at 555-123-4567, I'm John Smith from New York",
        "AWS key: AKIAIOSFODNN7EXAMPLE",
        "Your SSN is 123-45-6789",
        "Normal text without any PII",
    ]

    print("=" * 60)
    print("CASCADE DLP v2 - Demo")
    print("=" * 60)

    for text in test_cases:
        result = dlp.process(text)

        print(f"\nInput: {text}")
        print(f"Action: {result.action}")
        print(f"Output: {result.redacted_text}")
        print(f"Detections: {len(result.detections)}")
        for d in result.detections:
            print(f"  - {d.entity_type}: {d.text}")
        print(f"Time: {result.processing_time_ms:.1f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
