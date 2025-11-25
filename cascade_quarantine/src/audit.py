"""
Audit Logger - Phase 4

Tracks all transformations through the safety pipeline.
Maintains immutable record for compliance and continuous improvement.
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class AuditLogEntry:
    """Single audit log entry."""
    timestamp: str
    session_id: str
    stage: str
    input_hash: str
    result: Dict[str, Any]
    latency_ms: float
    decision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "stage": self.stage,
            "input_hash": self.input_hash,
            "result": self.result,
            "latency_ms": self.latency_ms,
            "decision": self.decision,
        }


class AuditLogger:
    """
    Logs all transformations through the safety pipeline.

    Required for compliance and continuous improvement.
    Maintains immutable record of:
    - What input was received
    - How it was classified at each layer
    - What transformations were applied
    - What was sent to privileged LLM
    - What was blocked/redacted
    """

    # Pipeline stages
    STAGE_INBOUND_SAFETY = "inbound_safety"
    STAGE_REFUSAL_GENERATED = "refusal_generated"
    STAGE_QUARANTINE_ANALYSIS = "quarantine_analysis"
    STAGE_INJECTION_BLOCKED = "injection_blocked"
    STAGE_PRIVILEGED_LLM_INPUT = "privileged_llm_input"
    STAGE_PRIVILEGED_LLM_OUTPUT = "privileged_llm_output"
    STAGE_DLP_OUTBOUND = "dlp_outbound_filter"
    STAGE_FINAL_RESPONSE = "final_response"

    def __init__(self, db_path: str = "logs/audit.db"):
        """
        Initialize audit logger.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create audit log table if not exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                input_hash TEXT,
                result TEXT NOT NULL,
                latency_ms REAL,
                decision TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_session
            ON audit_log(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_stage
            ON audit_log(stage)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
            ON audit_log(timestamp)
        """)

        conn.commit()
        conn.close()

    @staticmethod
    def hash_input(text: str) -> str:
        """Generate SHA256 hash of input for tracking."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def log_stage(
        self,
        stage: str,
        session_id: str,
        result: Any,
        input_text: Optional[str] = None,
        latency_ms: float = 0.0,
        decision: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log a stage in the pipeline.

        Args:
            stage: Pipeline stage name (use STAGE_* constants)
            session_id: For tracking user session
            result: Stage output (any serializable object)
            input_text: Optional input text to hash
            latency_ms: Processing time
            decision: Decision made at this stage

        Returns:
            AuditLogEntry that was logged
        """
        # Serialize result
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"value": str(result)}

        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            stage=stage,
            input_hash=self.hash_input(input_text) if input_text else "N/A",
            result=result_dict,
            latency_ms=latency_ms,
            decision=decision,
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO audit_log
            (timestamp, session_id, stage, input_hash, result, latency_ms, decision)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry.timestamp,
                entry.session_id,
                entry.stage,
                entry.input_hash,
                json.dumps(entry.result),
                entry.latency_ms,
                entry.decision,
            ),
        )

        conn.commit()
        conn.close()

        return entry

    def get_session_trace(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get complete trace of all transformations for a session.

        Args:
            session_id: Session to retrieve

        Returns:
            List of audit entries in chronological order
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, stage, input_hash, result, latency_ms, decision
            FROM audit_log
            WHERE session_id = ?
            ORDER BY id ASC
        """,
            (session_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "timestamp": row[0],
                "stage": row[1],
                "input_hash": row[2],
                "result": json.loads(row[3]),
                "latency_ms": row[4],
                "decision": row[5],
            }
            for row in rows
        ]

    def get_recent_logs(
        self, limit: int = 100, stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent audit log entries.

        Args:
            limit: Maximum entries to return
            stage: Filter by specific stage

        Returns:
            List of recent audit entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if stage:
            cursor.execute(
                """
                SELECT timestamp, session_id, stage, input_hash, result, latency_ms, decision
                FROM audit_log
                WHERE stage = ?
                ORDER BY id DESC
                LIMIT ?
            """,
                (stage, limit),
            )
        else:
            cursor.execute(
                """
                SELECT timestamp, session_id, stage, input_hash, result, latency_ms, decision
                FROM audit_log
                ORDER BY id DESC
                LIMIT ?
            """,
                (limit,),
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "timestamp": row[0],
                "session_id": row[1],
                "stage": row[2],
                "input_hash": row[3],
                "result": json.loads(row[4]),
                "latency_ms": row[5],
                "decision": row[6],
            }
            for row in rows
        ]

    def get_injection_blocks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent injection blocks for analysis."""
        return self.get_recent_logs(limit=limit, stage=self.STAGE_INJECTION_BLOCKED)

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from audit log."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total entries
        cursor.execute("SELECT COUNT(*) FROM audit_log")
        total = cursor.fetchone()[0]

        # Entries by stage
        cursor.execute(
            """
            SELECT stage, COUNT(*)
            FROM audit_log
            GROUP BY stage
        """
        )
        by_stage = dict(cursor.fetchall())

        # Entries by decision
        cursor.execute(
            """
            SELECT decision, COUNT(*)
            FROM audit_log
            WHERE decision IS NOT NULL
            GROUP BY decision
        """
        )
        by_decision = dict(cursor.fetchall())

        # Average latency by stage
        cursor.execute(
            """
            SELECT stage, AVG(latency_ms)
            FROM audit_log
            WHERE latency_ms > 0
            GROUP BY stage
        """
        )
        avg_latency = {row[0]: round(row[1], 2) for row in cursor.fetchall()}

        # Unique sessions
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM audit_log")
        unique_sessions = cursor.fetchone()[0]

        conn.close()

        return {
            "total_entries": total,
            "unique_sessions": unique_sessions,
            "by_stage": by_stage,
            "by_decision": by_decision,
            "avg_latency_ms": avg_latency,
        }
