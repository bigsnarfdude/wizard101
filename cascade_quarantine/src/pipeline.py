"""
Safe LLM Pipeline - Phase 4

Orchestrates all safety cascades into a complete pipeline:
1. cascade_inbound - Safety classification
2. cascade_refusals - Refusal generation (if harmful)
3. cascade_quarantine - Intent extraction + injection detection
4. Privileged LLM - Actual response generation
5. cascade_dlp - Output filtering

Based on Simon Willison's Dual LLM Pattern.
"""

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

try:
    from .audit import AuditLogger
    from .quarantine import Quarantine, ExtractedIntent, SuspicionLevel
except ImportError:
    from audit import AuditLogger
    from quarantine import Quarantine, ExtractedIntent, SuspicionLevel


@dataclass
class PipelineResult:
    """Result from complete safety pipeline."""
    response: str
    session_id: str

    # Stage results
    safety_label: str                    # "safe" or "harmful"
    safety_confidence: float
    injection_detected: bool
    dlp_action: str                      # "ALLOW", "REDACT", "BLOCK"

    # Timing
    total_latency_ms: float
    stage_latencies: Dict[str, float]

    # Decision path
    stopped_at: str                      # Which stage produced final response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "session_id": self.session_id,
            "safety_label": self.safety_label,
            "safety_confidence": self.safety_confidence,
            "injection_detected": self.injection_detected,
            "dlp_action": self.dlp_action,
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies": self.stage_latencies,
            "stopped_at": self.stopped_at,
        }


class SafeLLMPipeline:
    """
    Complete Phase 4 integration pipeline.

    Orchestrates all safety cascades with full audit logging.

    Flow:
        User Input
            ↓
        cascade_inbound (safety check)
            ├─ harmful → cascade_refusals → Return refusal
            ├─ safe → cascade_quarantine
            │   ├─ injection → Block
            │   ├─ safe → Privileged LLM → cascade_dlp → Return
    """

    def __init__(
        self,
        # Component configuration
        quarantine_model: str = "qwen3:4b",
        use_classifier: bool = True,

        # Audit configuration
        audit_db_path: str = "logs/audit.db",

        # LLM callback (user provides their own LLM)
        llm_callback: Optional[Callable[[Dict], str]] = None,

        # Optional: pre-initialized components (for testing)
        inbound_cascade=None,
        refusal_pipeline=None,
        dlp_cascade=None,
    ):
        """
        Initialize the safe LLM pipeline.

        Args:
            quarantine_model: Ollama model for intent extraction
            use_classifier: Whether to use ML classifier
            audit_db_path: Path to audit SQLite database
            llm_callback: Function that takes sanitized input and returns LLM response
            inbound_cascade: Optional pre-initialized SafetyCascade
            refusal_pipeline: Optional pre-initialized RefusalPipeline
            dlp_cascade: Optional pre-initialized DLPCascade
        """
        # Initialize quarantine (always available)
        self.quarantine = Quarantine(
            model=quarantine_model,
            use_classifier=use_classifier,
        )

        # Initialize audit logger
        self.audit = AuditLogger(db_path=audit_db_path)

        # Store LLM callback
        self._llm_callback = llm_callback

        # Lazy-load other cascades (may not be installed)
        self._inbound = inbound_cascade
        self._refusals = refusal_pipeline
        self._dlp = dlp_cascade

    @property
    def inbound(self):
        """Lazy-load cascade_inbound."""
        if self._inbound is None:
            try:
                from cascade_inbound import SafetyCascade
                self._inbound = SafetyCascade()
            except ImportError:
                raise ImportError(
                    "cascade_inbound not installed. "
                    "Install it or provide inbound_cascade parameter."
                )
        return self._inbound

    @property
    def refusals(self):
        """Lazy-load cascade_refusals."""
        if self._refusals is None:
            try:
                from cascade_refusals import RefusalPipeline
                self._refusals = RefusalPipeline()
            except ImportError:
                raise ImportError(
                    "cascade_refusals not installed. "
                    "Install it or provide refusal_pipeline parameter."
                )
        return self._refusals

    @property
    def dlp(self):
        """Lazy-load cascade_dlp."""
        if self._dlp is None:
            try:
                from cascade_dlp import DLPCascade
                self._dlp = DLPCascade()
            except ImportError:
                # DLP is optional - can proceed without it
                return None
        return self._dlp

    def process(
        self,
        user_input: str,
        session_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process user input through complete safety pipeline.

        Args:
            user_input: Untrusted user input
            session_id: Optional session ID for tracking (auto-generated if not provided)

        Returns:
            PipelineResult with response and metadata
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())[:8]
        stage_latencies = {}

        # STEP 1: Safety classification (cascade_inbound)
        stage_start = time.time()
        safety_result = self.inbound.classify(user_input)
        stage_latencies["inbound"] = (time.time() - stage_start) * 1000

        self.audit.log_stage(
            stage=AuditLogger.STAGE_INBOUND_SAFETY,
            session_id=session_id,
            result={
                "label": safety_result.label,
                "confidence": safety_result.confidence,
                "stopped_at": safety_result.stopped_at,
            },
            input_text=user_input,
            latency_ms=stage_latencies["inbound"],
            decision=safety_result.label,
        )

        # STEP 2: If harmful, generate refusal
        if safety_result.label == "harmful":
            stage_start = time.time()
            refusal_result = self.refusals.process(user_input)
            stage_latencies["refusals"] = (time.time() - stage_start) * 1000

            self.audit.log_stage(
                stage=AuditLogger.STAGE_REFUSAL_GENERATED,
                session_id=session_id,
                result=refusal_result,
                latency_ms=stage_latencies["refusals"],
                decision="refuse",
            )

            total_latency = (time.time() - start_time) * 1000

            return PipelineResult(
                response=refusal_result.get("response", "I cannot help with that request."),
                session_id=session_id,
                safety_label="harmful",
                safety_confidence=safety_result.confidence,
                injection_detected=False,
                dlp_action="N/A",
                total_latency_ms=total_latency,
                stage_latencies=stage_latencies,
                stopped_at="refusals",
            )

        # STEP 3: Safe content → Quarantine intent extraction
        stage_start = time.time()
        quarantine_result = self.quarantine.extract_intent(user_input)
        stage_latencies["quarantine"] = (time.time() - stage_start) * 1000

        self.audit.log_stage(
            stage=AuditLogger.STAGE_QUARANTINE_ANALYSIS,
            session_id=session_id,
            result=quarantine_result.to_dict(),
            input_text=user_input,
            latency_ms=stage_latencies["quarantine"],
            decision="injection" if quarantine_result.injection_detected else "safe",
        )

        # STEP 4: Check injection detection
        if quarantine_result.injection_detected:
            self.audit.log_stage(
                stage=AuditLogger.STAGE_INJECTION_BLOCKED,
                session_id=session_id,
                result={
                    "patterns": quarantine_result.suspicious_patterns,
                    "classifier_prob": quarantine_result.classifier_probability,
                    "suspicion_level": quarantine_result.suspicion_level.value,
                },
                decision="block",
            )

            total_latency = (time.time() - start_time) * 1000

            return PipelineResult(
                response="Your request appears to contain manipulation attempts. Please rephrase your question.",
                session_id=session_id,
                safety_label="safe",
                safety_confidence=safety_result.confidence,
                injection_detected=True,
                dlp_action="N/A",
                total_latency_ms=total_latency,
                stage_latencies=stage_latencies,
                stopped_at="quarantine",
            )

        # STEP 5: Call privileged LLM with sanitized input
        llm_input = {
            "original_input": user_input,
            "extracted_intent": quarantine_result.primary_intent,
            "sanitized_request": quarantine_result.sanitized_request or user_input,
            "entities": quarantine_result.entities,
            "intent_category": quarantine_result.intent_category,
            "confidence": quarantine_result.confidence,
        }

        self.audit.log_stage(
            stage=AuditLogger.STAGE_PRIVILEGED_LLM_INPUT,
            session_id=session_id,
            result=llm_input,
        )

        stage_start = time.time()
        if self._llm_callback:
            llm_response = self._llm_callback(llm_input)
        else:
            # Default: echo sanitized request (for testing)
            llm_response = f"[LLM would respond to: {llm_input['sanitized_request']}]"
        stage_latencies["llm"] = (time.time() - stage_start) * 1000

        self.audit.log_stage(
            stage=AuditLogger.STAGE_PRIVILEGED_LLM_OUTPUT,
            session_id=session_id,
            result={"response_length": len(llm_response)},
            latency_ms=stage_latencies["llm"],
        )

        # STEP 6: DLP outbound filtering
        dlp_action = "ALLOW"
        final_response = llm_response

        if self.dlp:
            stage_start = time.time()
            dlp_result = self.dlp.process(llm_response)
            stage_latencies["dlp"] = (time.time() - stage_start) * 1000

            dlp_action = dlp_result.action

            self.audit.log_stage(
                stage=AuditLogger.STAGE_DLP_OUTBOUND,
                session_id=session_id,
                result={
                    "action": dlp_action,
                    "detections": len(dlp_result.detections) if hasattr(dlp_result, 'detections') else 0,
                },
                latency_ms=stage_latencies["dlp"],
                decision=dlp_action,
            )

            if dlp_action == "BLOCK":
                final_response = "Response contained sensitive information and was blocked."
            elif dlp_action == "REDACT":
                final_response = dlp_result.redacted_text

        # Log final response
        total_latency = (time.time() - start_time) * 1000

        self.audit.log_stage(
            stage=AuditLogger.STAGE_FINAL_RESPONSE,
            session_id=session_id,
            result={"response_length": len(final_response)},
            latency_ms=total_latency,
            decision=dlp_action,
        )

        return PipelineResult(
            response=final_response,
            session_id=session_id,
            safety_label="safe",
            safety_confidence=safety_result.confidence,
            injection_detected=False,
            dlp_action=dlp_action,
            total_latency_ms=total_latency,
            stage_latencies=stage_latencies,
            stopped_at="dlp" if self.dlp else "llm",
        )

    def process_quarantine_only(
        self,
        user_input: str,
        session_id: Optional[str] = None,
    ) -> ExtractedIntent:
        """
        Run only the quarantine stage (for testing/standalone use).

        Args:
            user_input: Input to analyze
            session_id: Optional session ID

        Returns:
            ExtractedIntent from quarantine analysis
        """
        session_id = session_id or str(uuid.uuid4())[:8]

        start_time = time.time()
        result = self.quarantine.extract_intent(user_input)
        latency_ms = (time.time() - start_time) * 1000

        self.audit.log_stage(
            stage=AuditLogger.STAGE_QUARANTINE_ANALYSIS,
            session_id=session_id,
            result=result.to_dict(),
            input_text=user_input,
            latency_ms=latency_ms,
            decision="injection" if result.injection_detected else "safe",
        )

        return result

    def get_session_trace(self, session_id: str) -> list:
        """Get complete audit trace for a session."""
        return self.audit.get_session_trace(session_id)

    def get_statistics(self) -> dict:
        """Get pipeline statistics from audit log."""
        return self.audit.get_statistics()


# Convenience function for standalone quarantine usage
def quarantine_input(
    user_input: str,
    model: str = "qwen3:4b",
    use_classifier: bool = True,
) -> ExtractedIntent:
    """
    Quick function to quarantine a single input.

    Args:
        user_input: Text to analyze
        model: Ollama model for intent extraction
        use_classifier: Whether to use ML classifier

    Returns:
        ExtractedIntent with analysis
    """
    quarantine = Quarantine(model=model, use_classifier=use_classifier)
    return quarantine.extract_intent(user_input)
