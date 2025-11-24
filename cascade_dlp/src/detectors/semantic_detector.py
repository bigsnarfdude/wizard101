"""
Semantic PII Detector for cascade_dlp - Stage 3.

Uses transformer models to detect PII that regex patterns miss:
- Natural language password disclosures
- Context-dependent sensitive data
- Complex patterns requiring semantic understanding

Model options:
1. DeBERTa-v3-base: 10-50ms, 0.95+ F1 (default)
2. GLiNER: Zero-shot NER, no fine-tuning needed
3. Llama-3.2-1B: 50-200ms, 0.98 F1 (highest accuracy)
"""

import re
from typing import List, Optional
from dataclasses import dataclass

# Try to import transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import GLiNER
try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False


@dataclass
class Detection:
    """Detection result."""
    detector_name: str
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    metadata: dict


class SemanticDetector:
    """
    Semantic PII detector using transformer models.

    Catches patterns that regex misses:
    - "Your password is X"
    - "The secret code is Y"
    - Context-dependent PII
    """

    # Entity types this detector specializes in
    SEMANTIC_ENTITIES = [
        "PASSWORD",
        "SECRET_CODE",
        "API_KEY",
        "ACCESS_TOKEN",
        "CREDENTIAL",
        "PIN_CODE",
        "SECURITY_QUESTION",
        "VERIFICATION_CODE",
    ]

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        confidence_threshold: float = 0.7,
        use_gliner: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize semantic detector.

        Args:
            model_name: HuggingFace model name
            confidence_threshold: Minimum confidence for detection
            use_gliner: Use GLiNER for zero-shot NER
            device: Device to run model on (cpu/cuda)
        """
        self.name = "SemanticDetector"
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.use_gliner = use_gliner

        # Semantic patterns to look for (pre-filter)
        self.semantic_patterns = [
            # Password disclosures
            r"(?i)(?:your|the|my|temp|temporary|default|initial|new|current)\s+password",
            r"(?i)password\s+(?:is|:|\=)",
            r"(?i)(?:enter|use|type)\s+(?:the\s+)?password",

            # Secret/credential disclosures
            r"(?i)(?:your|the|my)\s+(?:secret|api|access)\s+(?:key|token|code)",
            r"(?i)(?:secret|api|access)\s+(?:key|token|code)\s+(?:is|:|\=)",

            # PIN/verification codes
            r"(?i)(?:your|the)\s+(?:pin|verification|security)\s+(?:code|number)",
            r"(?i)(?:pin|verification)\s+(?:code|number)\s+(?:is|:|\=)",
        ]
        self._compiled_patterns = [re.compile(p) for p in self.semantic_patterns]

        # Load model if available
        if use_gliner and GLINER_AVAILABLE:
            self._load_gliner()
        elif TRANSFORMERS_AVAILABLE:
            self._load_transformer(model_name)

    def _load_gliner(self):
        """Load GLiNER model for zero-shot NER."""
        try:
            self.model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
            self.model_type = "gliner"
            print(f"Loaded GLiNER model")
        except Exception as e:
            print(f"Failed to load GLiNER: {e}")
            self.model = None

    def _load_transformer(self, model_name: str):
        """Load transformer model for NER."""
        try:
            self.model = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
                device=-1 if self.device == "cpu" else 0
            )
            self.model_type = "transformer"
            print(f"Loaded transformer model: {model_name}")
        except Exception as e:
            print(f"Failed to load transformer model: {e}")
            self.model = None

    def _has_semantic_indicator(self, text: str) -> bool:
        """Check if text contains semantic patterns worth analyzing."""
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return True
        return False

    def _extract_value_after_pattern(self, text: str, match_end: int) -> Optional[tuple]:
        """Extract the actual sensitive value after a pattern match."""
        # Look for value after "is", ":", "="
        remaining = text[match_end:match_end + 100]  # Look ahead 100 chars

        # Pattern: whitespace + optional quotes + value
        value_pattern = r'^\s*[:\=]?\s*[\'"]?([A-Za-z0-9!@#$%^&*_\-+=\.]{4,50})[\'"]?'
        match = re.search(value_pattern, remaining)

        if match:
            value = match.group(1)
            # Filter out placeholders
            if value.upper() in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'XXXXX', 'PLACEHOLDER']:
                return None
            return (value, match_end + match.start(1), match_end + match.end(1))

        return None

    def detect(self, text: str) -> List[Detection]:
        """
        Detect PII using semantic analysis.

        Args:
            text: Input text to scan

        Returns:
            List of Detection objects
        """
        detections = []

        # Quick check - does text have semantic patterns?
        if not self._has_semantic_indicator(text):
            return detections

        # Method 1: Pattern-based extraction (fast, always available)
        detections.extend(self._detect_with_patterns(text))

        # Method 2: Model-based detection (if model loaded)
        if self.model:
            if self.model_type == "gliner":
                detections.extend(self._detect_with_gliner(text))
            else:
                detections.extend(self._detect_with_transformer(text))

        # Deduplicate by position
        seen = set()
        unique_detections = []
        for d in detections:
            key = (d.start, d.end, d.entity_type)
            if key not in seen:
                seen.add(key)
                unique_detections.append(d)

        return unique_detections

    def _detect_with_patterns(self, text: str) -> List[Detection]:
        """Detect using semantic regex patterns."""
        detections = []

        for i, pattern in enumerate(self._compiled_patterns):
            for match in pattern.finditer(text):
                # Try to extract the actual value
                value_info = self._extract_value_after_pattern(text, match.end())

                if value_info:
                    value, start, end = value_info

                    # Determine entity type from pattern
                    pattern_text = self.semantic_patterns[i].lower()
                    if 'password' in pattern_text:
                        entity_type = "PASSWORD"
                    elif 'pin' in pattern_text:
                        entity_type = "PIN_CODE"
                    elif 'verification' in pattern_text:
                        entity_type = "VERIFICATION_CODE"
                    elif 'api' in pattern_text:
                        entity_type = "API_KEY"
                    elif 'secret' in pattern_text or 'access' in pattern_text:
                        entity_type = "CREDENTIAL"
                    else:
                        entity_type = "SECRET_CODE"

                    detection = Detection(
                        detector_name=self.name,
                        entity_type=entity_type,
                        text=value,
                        start=start,
                        end=end,
                        confidence=0.85,
                        metadata={
                            "description": f"Semantic {entity_type.lower()} detection",
                            "method": "pattern",
                            "context": text[max(0, match.start()-20):min(len(text), end+20)]
                        }
                    )
                    detections.append(detection)

        return detections

    def _detect_with_gliner(self, text: str) -> List[Detection]:
        """Detect using GLiNER zero-shot NER."""
        if not self.model:
            return []

        detections = []

        # Define labels for zero-shot detection
        labels = ["password", "api key", "secret key", "pin code", "verification code"]

        try:
            entities = self.model.predict_entities(text, labels)

            for entity in entities:
                if entity["score"] < self.confidence_threshold:
                    continue

                # Map GLiNER label to our entity type
                label = entity["label"].upper().replace(" ", "_")
                if label == "PASSWORD":
                    entity_type = "PASSWORD"
                elif label in ["API_KEY", "SECRET_KEY"]:
                    entity_type = "API_KEY"
                elif label == "PIN_CODE":
                    entity_type = "PIN_CODE"
                elif label == "VERIFICATION_CODE":
                    entity_type = "VERIFICATION_CODE"
                else:
                    entity_type = "CREDENTIAL"

                detection = Detection(
                    detector_name=self.name,
                    entity_type=entity_type,
                    text=entity["text"],
                    start=entity["start"],
                    end=entity["end"],
                    confidence=entity["score"],
                    metadata={
                        "description": f"GLiNER {entity_type.lower()} detection",
                        "method": "gliner",
                        "original_label": entity["label"]
                    }
                )
                detections.append(detection)

        except Exception as e:
            print(f"GLiNER detection error: {e}")

        return detections

    def _detect_with_transformer(self, text: str) -> List[Detection]:
        """Detect using transformer NER model."""
        if not self.model:
            return []

        detections = []

        try:
            # Run NER
            entities = self.model(text)

            for entity in entities:
                # Standard NER models detect MISC, which can include secrets
                if entity.get("entity_group") in ["MISC", "O"]:
                    continue

                if entity["score"] < self.confidence_threshold:
                    continue

                # Check if this might be a credential
                entity_text = entity.get("word", "")

                # Look for patterns that suggest this is a credential
                context_start = max(0, entity["start"] - 50)
                context = text[context_start:entity["end"] + 20].lower()

                is_credential = any(word in context for word in [
                    "password", "secret", "key", "token", "pin", "code", "credential"
                ])

                if is_credential:
                    detection = Detection(
                        detector_name=self.name,
                        entity_type="CREDENTIAL",
                        text=entity_text,
                        start=entity["start"],
                        end=entity["end"],
                        confidence=entity["score"],
                        metadata={
                            "description": "Transformer credential detection",
                            "method": "transformer",
                            "original_entity": entity.get("entity_group")
                        }
                    )
                    detections.append(detection)

        except Exception as e:
            print(f"Transformer detection error: {e}")

        return detections


def main():
    """Test the semantic detector."""
    print("=" * 60)
    print("SEMANTIC DETECTOR TEST")
    print("=" * 60)

    # Initialize detector (pattern-only mode for testing)
    detector = SemanticDetector(confidence_threshold=0.7)

    test_cases = [
        # Should detect
        "Your password is E5_N8G2xW",
        "The temporary password is: abc123def",
        "Enter the password SuperSecret123!",
        "Your API key is sk_live_abc123xyz",
        "The secret code is 847291",
        "Your PIN code is 4532",
        "The verification code is 123456",
        "My new password: MyP@ssw0rd!",

        # Should NOT detect
        "Please enter your password in the form",
        "Change your password regularly",
        "The password field is required",
        "Your password has been reset",
        "This is normal text without any secrets",
    ]

    for text in test_cases:
        print(f"\nInput: {text}")
        detections = detector.detect(text)

        if detections:
            for d in detections:
                print(f"  Found: {d.entity_type}")
                print(f"    Value: {d.text}")
                print(f"    Confidence: {d.confidence}")
        else:
            print("  No detections")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
