"""
Core data models for cascade_quarantine.

QuarantineCase: Captured case for review when confidence is low.
"""

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import json


class CaptureReason(str, Enum):
    """Why the case was captured for review."""
    LOW_CONFIDENCE = "low_confidence"        # confidence < threshold
    LAYER_DISAGREEMENT = "layer_disagreement"  # L0 and L1 disagree
    SUSPICIOUS_PATTERN = "suspicious_pattern"  # Detected injection attempt
    BORDERLINE_CASE = "borderline_case"       # Near decision boundary
    MANUAL_FLAG = "manual_flag"               # Flagged by operator
    AUDIT_SAMPLE = "audit_sample"             # Random audit sampling


class CascadeSource(str, Enum):
    """Which cascade the case came from."""
    INBOUND = "cascade_inbound"       # Safety classification (L0-L3)
    REFUSALS = "cascade_refusals"     # Refusal generation
    QUARANTINE = "cascade_quarantine" # This layer (intent extraction)
    DLP = "cascade_dlp"               # Data loss prevention


class ReviewStatus(str, Enum):
    """Current status of human review."""
    PENDING = "pending"       # Awaiting review
    IN_REVIEW = "in_review"   # Currently being reviewed
    APPROVED = "approved"     # Decision was correct
    CORRECTED = "corrected"   # Decision was wrong, corrected
    ESCALATED = "escalated"   # Needs senior review
    DISMISSED = "dismissed"   # Not actionable


@dataclass
class LayerResult:
    """Result from a single cascade layer."""
    layer: str                      # L0, L1, L2, L3, etc.
    label: str                      # safe, harmful, uncertain, etc.
    confidence: float               # 0.0 to 1.0
    latency_ms: float               # Processing time
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuarantineCase:
    """
    A captured case for human review.

    Captured when any cascade layer has low confidence (< 0.75).
    """
    # Identification
    case_id: str                                # Unique case ID (qua_XXXXX)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Input data
    input_text: str = ""                        # Original untrusted input
    input_hash: str = ""                        # SHA256 hash for dedup
    input_length: int = 0                       # Character count

    # Capture metadata
    capture_reason: CaptureReason = CaptureReason.LOW_CONFIDENCE
    cascade_source: CascadeSource = CascadeSource.INBOUND
    confidence: float = 0.0                     # Confidence that triggered capture
    threshold: float = 0.75                     # Threshold that was violated

    # Classification result
    final_label: str = ""                       # safe, harmful, uncertain
    stopped_at: str = ""                        # Which layer made final decision

    # Layer journey
    layer_results: List[LayerResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    # Review tracking
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer: Optional[str] = None              # Who reviewed
    review_timestamp: Optional[str] = None      # When reviewed
    corrected_label: Optional[str] = None       # If corrected, new label
    review_notes: Optional[str] = None          # Reviewer comments

    # Additional context
    session_id: Optional[str] = None            # For tracking user sessions
    request_id: Optional[str] = None            # Original request ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived fields."""
        if self.input_text and not self.input_hash:
            self.input_hash = self._hash_content(self.input_text)
        if self.input_text and not self.input_length:
            self.input_length = len(self.input_text)

    @staticmethod
    def _hash_content(content: str) -> str:
        """Create SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def generate_case_id() -> str:
        """Generate unique case ID."""
        import uuid
        return f"qua_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enums to strings
        result["capture_reason"] = self.capture_reason.value
        result["cascade_source"] = self.cascade_source.value
        result["review_status"] = self.review_status.value
        # Convert layer results
        result["layer_results"] = [lr.to_dict() if isinstance(lr, LayerResult) else lr
                                    for lr in self.layer_results]
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuarantineCase":
        """Create from dictionary."""
        # Convert string enums back
        if isinstance(data.get("capture_reason"), str):
            data["capture_reason"] = CaptureReason(data["capture_reason"])
        if isinstance(data.get("cascade_source"), str):
            data["cascade_source"] = CascadeSource(data["cascade_source"])
        if isinstance(data.get("review_status"), str):
            data["review_status"] = ReviewStatus(data["review_status"])
        # Convert layer results
        if "layer_results" in data:
            data["layer_results"] = [
                LayerResult(**lr) if isinstance(lr, dict) else lr
                for lr in data["layer_results"]
            ]
        return cls(**data)

    @property
    def is_low_confidence(self) -> bool:
        """Check if this is a low confidence case."""
        return self.confidence < self.threshold

    @property
    def needs_review(self) -> bool:
        """Check if case needs human review."""
        return self.review_status == ReviewStatus.PENDING

    @property
    def was_corrected(self) -> bool:
        """Check if the decision was corrected."""
        return self.review_status == ReviewStatus.CORRECTED


@dataclass
class CaptureStats:
    """Statistics about captured cases."""
    total_captured: int = 0
    pending_review: int = 0
    reviewed: int = 0
    corrected: int = 0

    # By reason
    by_reason: Dict[str, int] = field(default_factory=dict)

    # By source
    by_source: Dict[str, int] = field(default_factory=dict)

    # By label
    by_label: Dict[str, int] = field(default_factory=dict)

    # Confidence distribution
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0

    # Time range
    earliest: Optional[str] = None
    latest: Optional[str] = None
