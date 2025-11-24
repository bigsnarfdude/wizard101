"""
Audit logger for cascade_dlp.

Structured logging for SOC2/GDPR compliance.
"""

import json
import hashlib
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict

from router import RouterResult
from context import RequestContext
from policies import Action


@dataclass
class AuditEntry:
    """Structured audit log entry."""
    # Timestamp
    timestamp: str

    # Request identification
    request_id: str
    session_id: str

    # User context
    user_id: str
    user_role: str
    ip_address: str

    # Action taken
    action: str
    blocked: bool
    redacted: bool

    # Detection summary
    detection_count: int
    detection_types: List[str]
    policies_matched: List[str]

    # Timing
    total_latency_ms: float

    # Content hashes (for audit without exposing data)
    input_hash: str
    output_hash: str

    # Optional details
    block_reason: Optional[str] = None
    application: str = ""


class AuditLogger:
    """
    Log DLP decisions for compliance.

    Outputs structured JSON for easy ingestion by SIEM systems.
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.entries: List[AuditEntry] = []

    def log(self, result: RouterResult, context: RequestContext) -> AuditEntry:
        """
        Create audit entry from router result.

        Args:
            result: RouterResult from DLP pipeline
            context: RequestContext for the request

        Returns:
            AuditEntry that was logged
        """
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_id=context.request_id,
            session_id=context.session_id,
            user_id=context.user_id,
            user_role=context.user_role.value,
            ip_address=context.ip_address,
            action=result.action.value,
            blocked=result.blocked,
            redacted=result.redacted,
            detection_count=result.detection_count,
            detection_types=[d.entity_type for d in result.detections],
            policies_matched=list(set(d.policy_name for d in result.decisions)),
            total_latency_ms=result.total_time_ms,
            input_hash=self._hash_content(result.original_text),
            output_hash=self._hash_content(result.processed_text),
            block_reason=result.block_reason,
            application=context.application,
        )

        self.entries.append(entry)

        # Write to file if configured
        if self.log_file:
            self._write_entry(entry)

        return entry

    def _hash_content(self, content: str) -> str:
        """Create SHA256 hash of content for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _write_entry(self, entry: AuditEntry):
        """Write entry to log file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def get_entries(self) -> List[AuditEntry]:
        """Get all logged entries."""
        return self.entries

    def get_summary(self) -> dict:
        """Get summary statistics of logged entries."""
        if not self.entries:
            return {"total": 0}

        total = len(self.entries)
        blocked = sum(1 for e in self.entries if e.blocked)
        redacted = sum(1 for e in self.entries if e.redacted)
        allowed = total - blocked - redacted

        # Count by action
        action_counts = {}
        for e in self.entries:
            action_counts[e.action] = action_counts.get(e.action, 0) + 1

        # Detection types
        all_types = []
        for e in self.entries:
            all_types.extend(e.detection_types)
        type_counts = {}
        for t in all_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        # Average latency
        avg_latency = sum(e.total_latency_ms for e in self.entries) / total

        return {
            "total": total,
            "blocked": blocked,
            "redacted": redacted,
            "allowed": allowed,
            "action_counts": action_counts,
            "detection_type_counts": type_counts,
            "avg_latency_ms": round(avg_latency, 2),
        }

    def to_json(self) -> str:
        """Export all entries as JSON."""
        return json.dumps([asdict(e) for e in self.entries], indent=2)


def main():
    """Test audit logger."""
    from router import OutboundRouter
    from context import create_external_context, create_internal_context

    router = OutboundRouter()
    logger = AuditLogger()

    test_cases = [
        ("External user", "My AWS key is AKIAIOSFODNN7EXAMPLE", create_external_context("user1", "req1")),
        ("External user", "Contact john@example.com for help", create_external_context("user2", "req2")),
        ("Internal user", "The weather is nice today", create_internal_context("staff1", "req3")),
        ("External user", "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", create_external_context("user3", "req4")),
    ]

    print("=" * 70)
    print("AUDIT LOGGER TEST")
    print("=" * 70)

    for desc, text, context in test_cases:
        context.ip_address = "192.168.1.1"
        context.application = "test-app"

        result = router.process(text, context)
        entry = logger.log(result, context)

        print(f"\n{desc}: {text[:40]}...")
        print(f"  Action: {entry.action}")
        print(f"  Detections: {entry.detection_types}")
        print(f"  Input hash: {entry.input_hash}")

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    summary = logger.get_summary()
    print(f"\nTotal requests: {summary['total']}")
    print(f"Blocked: {summary['blocked']}")
    print(f"Redacted: {summary['redacted']}")
    print(f"Allowed: {summary['allowed']}")
    print(f"Avg latency: {summary['avg_latency_ms']}ms")

    print("\nDetection types:")
    for dtype, count in summary.get('detection_type_counts', {}).items():
        print(f"  {dtype}: {count}")

    # Sample JSON output
    print("\n" + "=" * 70)
    print("SAMPLE AUDIT ENTRY (JSON)")
    print("=" * 70)
    if logger.entries:
        print(json.dumps(asdict(logger.entries[0]), indent=2))


if __name__ == "__main__":
    main()
