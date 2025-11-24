"""
Capture hooks for quarantine system.

Captures low-confidence cases from all cascades for human review.
"""

import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import asdict

try:
    from .models import (
        QuarantineCase,
        LayerResult,
        CaptureReason,
        CascadeSource,
        ReviewStatus,
    )
    from .config import QuarantineConfig, DEFAULT_CONFIG
    from .database import QuarantineDatabase
except ImportError:
    from models import (
        QuarantineCase,
        LayerResult,
        CaptureReason,
        CascadeSource,
        ReviewStatus,
    )
    from config import QuarantineConfig, DEFAULT_CONFIG
    from database import QuarantineDatabase


class CaptureHook:
    """
    Hook to capture low-confidence cases from any cascade.

    Usage:
        hook = CaptureHook(config)

        # In cascade_inbound
        result = cascade.classify(text)
        hook.capture_from_inbound(text, result)

        # In cascade_refusals
        result = refusal_pipeline.process(text, category)
        hook.capture_from_refusals(text, result, category)

        # In cascade_dlp
        result = dlp.process(text, context)
        hook.capture_from_dlp(text, result, context)
    """

    def __init__(self, config: QuarantineConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.db = QuarantineDatabase(self.config) if self.config.enable_database else None
        self._log_file = None

        # Callbacks for custom processing
        self._on_capture: List[Callable[[QuarantineCase], None]] = []

    def capture_from_inbound(
        self,
        input_text: str,
        result: Dict[str, Any],
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[QuarantineCase]:
        """
        Capture from cascade_inbound (safety classification).

        Args:
            input_text: Original user input
            result: CascadeResult dict with label, confidence, layers, etc.
            session_id: Optional session tracking
            request_id: Optional request tracking

        Returns:
            QuarantineCase if captured, None otherwise
        """
        if not self.config.capture_from_inbound:
            return None

        confidence = result.get("confidence", 1.0)
        capture_reason = self._determine_capture_reason(result, confidence)

        if capture_reason is None:
            # Check for audit sampling
            if random.random() < self.config.audit_sample_rate:
                capture_reason = CaptureReason.AUDIT_SAMPLE
            else:
                return None

        # Build layer results
        layer_results = []
        for layer_data in result.get("layers", []):
            layer_results.append(LayerResult(
                layer=layer_data.get("level", "unknown"),
                label=layer_data.get("label", "unknown"),
                confidence=layer_data.get("confidence", 0.0),
                latency_ms=layer_data.get("latency_ms", 0.0),
                metadata={k: v for k, v in layer_data.items()
                         if k not in ["level", "label", "confidence", "latency_ms"]},
            ))

        case = QuarantineCase(
            case_id=QuarantineCase.generate_case_id(),
            input_text=input_text,
            capture_reason=capture_reason,
            cascade_source=CascadeSource.INBOUND,
            confidence=confidence,
            threshold=self.config.confidence_threshold,
            final_label=result.get("label", "unknown"),
            stopped_at=result.get("stopped_at", "unknown"),
            layer_results=layer_results,
            total_latency_ms=result.get("total_latency_ms", 0.0),
            session_id=session_id,
            request_id=request_id,
            metadata={
                "reasoning": result.get("reasoning"),
                "audit_id": result.get("audit_id"),
            },
        )

        return self._save_case(case)

    def capture_from_refusals(
        self,
        input_text: str,
        result: Dict[str, Any],
        category: str,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[QuarantineCase]:
        """
        Capture from cascade_refusals (refusal generation).

        Args:
            input_text: Original user input
            result: RefusalResponse dict
            category: Llama Guard category
            session_id: Optional session tracking
            request_id: Optional request tracking
        """
        if not self.config.capture_from_refusals:
            return None

        # Refusals don't have confidence scores, capture based on category ambiguity
        # For now, capture a sample for training data
        if random.random() >= self.config.audit_sample_rate:
            return None

        case = QuarantineCase(
            case_id=QuarantineCase.generate_case_id(),
            input_text=input_text,
            capture_reason=CaptureReason.AUDIT_SAMPLE,
            cascade_source=CascadeSource.REFUSALS,
            confidence=0.0,  # No confidence for refusals
            threshold=self.config.confidence_threshold,
            final_label=category,
            stopped_at="refusal_generator",
            layer_results=[],
            session_id=session_id,
            request_id=request_id,
            metadata={
                "strategy": result.get("strategy"),
                "tone": result.get("tone"),
                "offer_alternative": result.get("offer_alternative"),
            },
        )

        return self._save_case(case)

    def capture_from_dlp(
        self,
        input_text: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[QuarantineCase]:
        """
        Capture from cascade_dlp (data loss prevention).

        Args:
            input_text: Text that was processed
            result: RouterResult dict
            context: RequestContext dict
            session_id: Optional session tracking
            request_id: Optional request tracking
        """
        if not self.config.capture_from_dlp:
            return None

        # DLP captures when there's a detection with low confidence
        detections = result.get("detections", [])
        if not detections:
            # Audit sample for clean passes
            if random.random() >= self.config.audit_sample_rate:
                return None
            capture_reason = CaptureReason.AUDIT_SAMPLE
            confidence = 1.0
        else:
            # Find lowest confidence detection
            min_conf = min(d.get("confidence", 1.0) for d in detections)
            if min_conf >= self.config.confidence_threshold:
                if random.random() >= self.config.audit_sample_rate:
                    return None
                capture_reason = CaptureReason.AUDIT_SAMPLE
            else:
                capture_reason = CaptureReason.LOW_CONFIDENCE
            confidence = min_conf

        # Build layer results from detections
        layer_results = []
        for detection in detections:
            layer_results.append(LayerResult(
                layer=detection.get("detector_name", "unknown"),
                label=detection.get("entity_type", "unknown"),
                confidence=detection.get("confidence", 0.0),
                latency_ms=0.0,
                metadata={
                    "text": detection.get("text", ""),
                    "start": detection.get("start"),
                    "end": detection.get("end"),
                },
            ))

        case = QuarantineCase(
            case_id=QuarantineCase.generate_case_id(),
            input_text=input_text,
            capture_reason=capture_reason,
            cascade_source=CascadeSource.DLP,
            confidence=confidence,
            threshold=self.config.confidence_threshold,
            final_label=result.get("action", "unknown"),
            stopped_at="dlp_router",
            layer_results=layer_results,
            total_latency_ms=result.get("total_time_ms", 0.0),
            session_id=session_id,
            request_id=request_id or (context.get("request_id") if context else None),
            metadata={
                "blocked": result.get("blocked"),
                "redacted": result.get("redacted"),
                "detection_count": result.get("detection_count", 0),
                "user_id": context.get("user_id") if context else None,
                "user_role": context.get("user_role") if context else None,
            },
        )

        return self._save_case(case)

    def capture_generic(
        self,
        input_text: str,
        source: CascadeSource,
        label: str,
        confidence: float,
        reason: CaptureReason = CaptureReason.LOW_CONFIDENCE,
        layer_results: Optional[List[LayerResult]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[QuarantineCase]:
        """
        Generic capture for any source.

        Useful for custom integrations or testing.
        """
        case = QuarantineCase(
            case_id=QuarantineCase.generate_case_id(),
            input_text=input_text,
            capture_reason=reason,
            cascade_source=source,
            confidence=confidence,
            threshold=self.config.confidence_threshold,
            final_label=label,
            stopped_at="generic",
            layer_results=layer_results or [],
            session_id=session_id,
            request_id=request_id,
            metadata=metadata or {},
        )

        return self._save_case(case)

    def _determine_capture_reason(
        self,
        result: Dict[str, Any],
        confidence: float,
    ) -> Optional[CaptureReason]:
        """Determine why we should capture this case."""
        # Low confidence
        if confidence < self.config.confidence_threshold:
            return CaptureReason.LOW_CONFIDENCE

        # Layer disagreement
        if self.config.detect_layer_disagreement:
            layers = result.get("layers", [])
            if len(layers) >= 2:
                l0_label = layers[0].get("label")
                l1_label = layers[1].get("label") if len(layers) > 1 else None
                if l0_label and l1_label and l0_label != l1_label:
                    # Check confidence difference
                    l0_conf = layers[0].get("confidence", 0)
                    l1_conf = layers[1].get("confidence", 0) if len(layers) > 1 else 0
                    if abs(l0_conf - l1_conf) >= self.config.disagreement_confidence_diff:
                        return CaptureReason.LAYER_DISAGREEMENT

        # Borderline cases (confidence between 0.7 and 0.8)
        if 0.7 <= confidence < 0.8:
            return CaptureReason.BORDERLINE_CASE

        return None

    def _save_case(self, case: QuarantineCase) -> QuarantineCase:
        """Save case to database and log file."""
        # Save to database
        if self.db:
            self.db.insert(case)

        # Write to log file
        if self.config.enable_log_file:
            self._write_to_log(case)

        # Call registered callbacks
        for callback in self._on_capture:
            try:
                callback(case)
            except Exception:
                pass  # Don't let callbacks break the capture

        return case

    def _write_to_log(self, case: QuarantineCase):
        """Write case to JSONL log file."""
        try:
            with open(self.config.log_file, "a") as f:
                f.write(case.to_json().replace("\n", " ") + "\n")
        except Exception:
            pass  # Don't fail on log write errors

    def on_capture(self, callback: Callable[[QuarantineCase], None]):
        """Register a callback for when cases are captured."""
        self._on_capture.append(callback)

    def get_stats(self):
        """Get capture statistics from database."""
        if self.db:
            return self.db.get_stats()
        return None

    def get_pending(self, limit: int = 100) -> List[QuarantineCase]:
        """Get cases pending review."""
        if self.db:
            return self.db.get_pending_review(limit)
        return []


# Convenience function for quick integration
def capture_if_low_confidence(
    input_text: str,
    result: Dict[str, Any],
    source: str = "inbound",
    config: QuarantineConfig = None,
) -> Optional[QuarantineCase]:
    """
    Quick capture function for integration into existing cascades.

    Example:
        from cascade_quarantine.src.capture import capture_if_low_confidence

        result = cascade.classify(text)
        capture_if_low_confidence(text, result)
    """
    hook = CaptureHook(config)

    if source == "inbound":
        return hook.capture_from_inbound(input_text, result)
    elif source == "refusals":
        return hook.capture_from_refusals(input_text, result, result.get("category", "unknown"))
    elif source == "dlp":
        return hook.capture_from_dlp(input_text, result)
    else:
        return hook.capture_generic(
            input_text,
            CascadeSource(source) if source in [s.value for s in CascadeSource] else CascadeSource.INBOUND,
            result.get("label", "unknown"),
            result.get("confidence", 0.0),
        )
