"""
Cascade DLP v2 - GLiNER2-based PII detection pipeline.
"""

import re
import hashlib
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from gliner2 import GLiNER2
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from . import config


@dataclass
class Detection:
    """PII detection result."""
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    stage: str  # "secrets", "presidio", or "gliner2"


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
    DLP pipeline using Presidio + GLiNER2 hybrid approach.

    Architecture:
        Stage 0: Secret patterns (regex) - <1ms
        Stage 1a: Presidio (span detection) - ~5ms
        Stage 1b: GLiNER2 (semantic validation) - ~80ms
        Stage 2: Policy engine - <1ms
        Stage 3: Redactor - <1ms
        Stage 4: Audit - <1ms
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        pii_types: List[str] = None,
        use_presidio: bool = True,
        use_gliner: bool = True,
    ):
        """
        Initialize the DLP cascade.

        Args:
            model_name: GLiNER2 model to use
            pii_types: List of PII types to detect
            use_presidio: Enable Presidio for span detection
            use_gliner: Enable GLiNER2 for semantic validation
        """
        self.pii_types = pii_types or config.PII_TYPES
        self.use_presidio = use_presidio
        self.use_gliner = use_gliner

        # Compile secret patterns
        self.secret_patterns = {}
        for name, cfg in config.SECRET_PATTERNS.items():
            self.secret_patterns[name] = {
                "regex": re.compile(cfg["pattern"]),
                "confidence": cfg["confidence"],
            }

        # Initialize Presidio
        if self.use_presidio:
            print("Loading Presidio analyzer...")
            # Use spaCy small model for speed
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()
            self.presidio = AnalyzerEngine(nlp_engine=nlp_engine)
            print("Presidio loaded")

        # Initialize GLiNER2
        if self.use_gliner:
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

        # Stage 1: Hybrid Presidio + GLiNER2
        detections.extend(self._detect_hybrid(text))

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

    def _detect_hybrid(self, text: str) -> List[Detection]:
        """
        Stage 1: Hybrid Presidio + GLiNER2 detection.
        
        Presidio provides accurate spans, GLiNER2 validates/enriches.
        """
        detections = []

        # Step 1: Presidio finds spans
        if self.use_presidio:
            presidio_results = self._detect_presidio(text)
            detections.extend(presidio_results)

        # Step 2: GLiNER2 validates and finds additional entities
        if self.use_gliner:
            gliner_results = self._detect_gliner2_validation(text, detections)
            detections.extend(gliner_results)

        return detections

    def _detect_presidio(self, text: str) -> List[Detection]:
        """Use Presidio for accurate span detection."""
        detections = []

        try:
            # Map our PII types to Presidio entities
            presidio_entities = self._map_to_presidio_entities(self.pii_types)
            
            # Analyze with Presidio
            results = self.presidio.analyze(
                text=text,
                entities=presidio_entities,
                language="en"
            )

            for result in results:
                detection = Detection(
                    entity_type=result.entity_type.lower().replace("_", " "),
                    text=text[result.start:result.end],
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                    stage="presidio",
                )
                detections.append(detection)

        except Exception as e:
            print(f"Presidio detection error: {e}")

        return detections

    def _detect_gliner2_validation(self, text: str, existing_detections: List[Detection]) -> List[Detection]:
        """
        Use GLiNER2 to validate Presidio results and find additional entities.
        """
        detections = []

        try:
            # Get GLiNER2 predictions
            results = self.model.extract_entities(text, self.pii_types)
            
            if "entities" in results:
                for pii_type, entity_list in results["entities"].items():
                    for entity_text in entity_list:
                        # Find position in text
                        pos = text.find(entity_text)
                        if pos == -1:
                            continue
                        
                        # Check if this overlaps with existing Presidio detection
                        overlaps_presidio = False
                        for existing in existing_detections:
                            if not (pos + len(entity_text) <= existing.start or pos >= existing.end):
                                overlaps_presidio = True
                                break
                        
                        # If it doesn't overlap, it's a new detection GLiNER2 found
                        if not overlaps_presidio:
                            detection = Detection(
                                entity_type=pii_type,
                                text=entity_text,
                                start=pos,
                                end=pos + len(entity_text),
                                confidence=0.85,  # GLiNER2 semantic confidence
                                stage="gliner2",
                            )
                            detections.append(detection)

        except Exception as e:
            print(f"GLiNER2 validation error: {e}")

        return detections

    def _map_to_presidio_entities(self, pii_types: List[str]) -> List[str]:
        """Map our PII types to Presidio's entity names."""
        mapping = {
            "person name": "PERSON",
            "first name": "PERSON",
            "last name": "PERSON",
            "email address": "EMAIL_ADDRESS",
            "phone number": "PHONE_NUMBER",
            "social security number": "US_SSN",
            "credit card number": "CREDIT_CARD",
            "ip address": "IP_ADDRESS",
            "date of birth": "DATE_TIME",
            "street address": "LOCATION",
            "city": "LOCATION",
            "medical record number": "MEDICAL_LICENSE",
            "driver license number": "US_DRIVER_LICENSE",
            "bank account number": "US_BANK_NUMBER",
        }
        
        presidio_entities = set()
        for pii_type in pii_types:
            if pii_type in mapping:
                presidio_entities.add(mapping[pii_type])
        
        return list(presidio_entities) if presidio_entities else None

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
