"""
Base detector class for cascade_dlp.
"""

from abc import ABC, abstractmethod
from typing import List
import time
from dataclasses import dataclass


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


class BaseDetector(ABC):
    """Abstract base class for all detectors."""

    def __init__(self, name: str, confidence_threshold: float = 0.5):
        self.name = name
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect(self, text: str) -> List[Detection]:
        """
        Detect sensitive data in text.

        Args:
            text: Input text to scan

        Returns:
            List of Detection objects
        """
        pass

    def detect_with_timing(self, text: str) -> tuple[List[Detection], float]:
        """Detect and return timing in milliseconds."""
        start = time.time()
        detections = self.detect(text)
        elapsed_ms = (time.time() - start) * 1000
        return detections, elapsed_ms

    def filter_by_confidence(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections below confidence threshold."""
        return [d for d in detections if d.confidence >= self.confidence_threshold]
