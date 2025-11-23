"""
Core types for cascade_dlp.

DataProvenance: Track data lineage and sensitivity
Detection: Result from a detector
PolicyResult: Outcome of policy evaluation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Set, List


class SensitivityLevel(str, Enum):
    """Data sensitivity classification."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataType(str, Enum):
    """Types of sensitive data."""
    PII = "pii"
    CREDENTIAL = "credential"
    CODE = "code"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    TEXT = "text"


class Action(str, Enum):
    """Policy actions."""
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    ALERT = "alert"
    AUDIT = "audit"
    REVIEW = "review"


@dataclass
class DataProvenance:
    """Track metadata for data flowing through the system."""

    sources: Set[str] = field(default_factory=set)
    allowed_readers: Set[str] = field(default_factory=set)
    sensitivity_level: SensitivityLevel = SensitivityLevel.INTERNAL
    data_types: Set[DataType] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def can_flow_to(self, reader: str) -> bool:
        """Check if data can flow to a specific reader."""
        if not self.allowed_readers:
            return True  # No restrictions
        return reader in self.allowed_readers


@dataclass
class Detection:
    """Result from a detector."""

    detector_name: str
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    metadata: dict = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8

    @property
    def is_medium_confidence(self) -> bool:
        return 0.5 <= self.confidence < 0.8


@dataclass
class PolicyResult:
    """Outcome of policy evaluation."""

    action: Action
    reason: str
    detections: List[Detection] = field(default_factory=list)
    requires_review: bool = False
    reviewer: Optional[str] = None

    @property
    def is_blocked(self) -> bool:
        return self.action in (Action.BLOCK, Action.REDACT)


@dataclass
class CascadeResult:
    """Result from full cascade evaluation."""

    input_text: str
    output_text: str  # After any redaction
    policy_result: PolicyResult
    stages_run: List[str]
    total_latency_ms: float
    stage_latencies_ms: dict = field(default_factory=dict)

    @property
    def was_modified(self) -> bool:
        return self.input_text != self.output_text


@dataclass
class RequestContext:
    """Context for a request being evaluated."""

    user_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance: Optional[DataProvenance] = None
    metadata: dict = field(default_factory=dict)
