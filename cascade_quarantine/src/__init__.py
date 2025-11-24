"""
Cascade Quarantine - Sanitizes untrusted input before privileged LLM execution.

Phase 1: Basic Capture System
- QuarantineCase dataclass for tracking cases
- SQLite database for persistence
- Capture hooks for low-confidence classifications
"""

from .models import QuarantineCase, CaptureReason, CascadeSource
from .config import QuarantineConfig
from .database import QuarantineDatabase
from .capture import CaptureHook

__all__ = [
    "QuarantineCase",
    "CaptureReason",
    "CascadeSource",
    "QuarantineConfig",
    "QuarantineDatabase",
    "CaptureHook",
]
