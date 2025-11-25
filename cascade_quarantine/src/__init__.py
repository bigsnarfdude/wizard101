"""
Cascade Quarantine - Sanitizes untrusted input before privileged LLM execution.

Based on Simon Willison's Dual LLM Pattern.

Phases:
- Phase 1: Basic Capture System (SQLite storage)
- Phase 2: Intent Extraction (Qwen3:4b via Ollama)
- Phase 3: ML Injection Classifier (99%+ accuracy)
- Phase 4: Pipeline Integration (audit logging)
"""

# Core models
from .models import QuarantineCase, CaptureReason, CascadeSource

# Configuration
from .config import QuarantineConfig

# Database
from .database import QuarantineDatabase

# Capture hooks
from .capture import CaptureHook

# Phase 2: Intent extraction
from .quarantine import Quarantine, ExtractedIntent, SuspicionLevel

# Phase 3: ML classifier
from .classifier import InjectionClassifier

# Phase 4: Pipeline integration
from .audit import AuditLogger
from .pipeline import SafeLLMPipeline, PipelineResult, quarantine_input

__all__ = [
    # Core
    "QuarantineCase",
    "CaptureReason",
    "CascadeSource",
    "QuarantineConfig",
    "QuarantineDatabase",
    "CaptureHook",
    # Phase 2
    "Quarantine",
    "ExtractedIntent",
    "SuspicionLevel",
    # Phase 3
    "InjectionClassifier",
    # Phase 4
    "AuditLogger",
    "SafeLLMPipeline",
    "PipelineResult",
    "quarantine_input",
]
