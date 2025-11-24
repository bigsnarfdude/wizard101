"""
SQLite database for quarantine cases.

Provides persistent storage with efficient querying for review workflow.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

try:
    from .models import (
        QuarantineCase,
        LayerResult,
        CaptureReason,
        CascadeSource,
        ReviewStatus,
        CaptureStats,
    )
    from .config import QuarantineConfig, DEFAULT_CONFIG
except ImportError:
    from models import (
        QuarantineCase,
        LayerResult,
        CaptureReason,
        CascadeSource,
        ReviewStatus,
        CaptureStats,
    )
    from config import QuarantineConfig, DEFAULT_CONFIG


class QuarantineDatabase:
    """
    SQLite database for storing quarantine cases.

    Thread-safe with connection pooling per-thread.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS quarantine_cases (
        -- Primary key
        case_id TEXT PRIMARY KEY,

        -- Timestamps
        timestamp TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,

        -- Input data
        input_text TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        input_length INTEGER NOT NULL,

        -- Capture metadata
        capture_reason TEXT NOT NULL,
        cascade_source TEXT NOT NULL,
        confidence REAL NOT NULL,
        threshold REAL NOT NULL,

        -- Classification result
        final_label TEXT NOT NULL,
        stopped_at TEXT NOT NULL,

        -- Layer results (JSON)
        layer_results TEXT,
        total_latency_ms REAL,

        -- Review tracking
        review_status TEXT DEFAULT 'pending',
        reviewer TEXT,
        review_timestamp TEXT,
        corrected_label TEXT,
        review_notes TEXT,

        -- Additional context
        session_id TEXT,
        request_id TEXT,
        metadata TEXT
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_timestamp ON quarantine_cases(timestamp);
    CREATE INDEX IF NOT EXISTS idx_review_status ON quarantine_cases(review_status);
    CREATE INDEX IF NOT EXISTS idx_confidence ON quarantine_cases(confidence);
    CREATE INDEX IF NOT EXISTS idx_capture_reason ON quarantine_cases(capture_reason);
    CREATE INDEX IF NOT EXISTS idx_cascade_source ON quarantine_cases(cascade_source);
    CREATE INDEX IF NOT EXISTS idx_input_hash ON quarantine_cases(input_hash);
    CREATE INDEX IF NOT EXISTS idx_final_label ON quarantine_cases(final_label);
    """

    def __init__(self, config: QuarantineConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.db_path = self.config.database_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert(self, case: QuarantineCase) -> bool:
        """
        Insert a quarantine case into the database.

        Returns True if inserted, False if duplicate (by case_id).
        """
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO quarantine_cases (
                        case_id, timestamp, input_text, input_hash, input_length,
                        capture_reason, cascade_source, confidence, threshold,
                        final_label, stopped_at, layer_results, total_latency_ms,
                        review_status, reviewer, review_timestamp, corrected_label,
                        review_notes, session_id, request_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    case.case_id,
                    case.timestamp,
                    case.input_text,
                    case.input_hash,
                    case.input_length,
                    case.capture_reason.value,
                    case.cascade_source.value,
                    case.confidence,
                    case.threshold,
                    case.final_label,
                    case.stopped_at,
                    json.dumps([lr.to_dict() if isinstance(lr, LayerResult) else lr
                               for lr in case.layer_results]),
                    case.total_latency_ms,
                    case.review_status.value,
                    case.reviewer,
                    case.review_timestamp,
                    case.corrected_label,
                    case.review_notes,
                    case.session_id,
                    case.request_id,
                    json.dumps(case.metadata),
                ))
                return True
            except sqlite3.IntegrityError:
                return False  # Duplicate case_id

    def insert_batch(self, cases: List[QuarantineCase]) -> int:
        """
        Insert multiple cases in a single transaction.

        Returns number of cases inserted.
        """
        inserted = 0
        with self._get_connection() as conn:
            for case in cases:
                try:
                    conn.execute("""
                        INSERT INTO quarantine_cases (
                            case_id, timestamp, input_text, input_hash, input_length,
                            capture_reason, cascade_source, confidence, threshold,
                            final_label, stopped_at, layer_results, total_latency_ms,
                            review_status, reviewer, review_timestamp, corrected_label,
                            review_notes, session_id, request_id, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        case.case_id,
                        case.timestamp,
                        case.input_text,
                        case.input_hash,
                        case.input_length,
                        case.capture_reason.value,
                        case.cascade_source.value,
                        case.confidence,
                        case.threshold,
                        case.final_label,
                        case.stopped_at,
                        json.dumps([lr.to_dict() if isinstance(lr, LayerResult) else lr
                                   for lr in case.layer_results]),
                        case.total_latency_ms,
                        case.review_status.value,
                        case.reviewer,
                        case.review_timestamp,
                        case.corrected_label,
                        case.review_notes,
                        case.session_id,
                        case.request_id,
                        json.dumps(case.metadata),
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass  # Skip duplicates
        return inserted

    def get(self, case_id: str) -> Optional[QuarantineCase]:
        """Get a case by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM quarantine_cases WHERE case_id = ?",
                (case_id,)
            ).fetchone()
            if row:
                return self._row_to_case(row)
            return None

    def get_by_hash(self, input_hash: str) -> List[QuarantineCase]:
        """Get all cases with the same input hash (deduplication check)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM quarantine_cases WHERE input_hash = ? ORDER BY timestamp DESC",
                (input_hash,)
            ).fetchall()
            return [self._row_to_case(row) for row in rows]

    def get_pending_review(self, limit: int = 100) -> List[QuarantineCase]:
        """Get cases pending review, ordered by confidence (lowest first)."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM quarantine_cases
                WHERE review_status = 'pending'
                ORDER BY confidence ASC, timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [self._row_to_case(row) for row in rows]

    def get_by_reason(self, reason: CaptureReason, limit: int = 100) -> List[QuarantineCase]:
        """Get cases by capture reason."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM quarantine_cases
                WHERE capture_reason = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (reason.value, limit)).fetchall()
            return [self._row_to_case(row) for row in rows]

    def get_by_source(self, source: CascadeSource, limit: int = 100) -> List[QuarantineCase]:
        """Get cases by cascade source."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM quarantine_cases
                WHERE cascade_source = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (source.value, limit)).fetchall()
            return [self._row_to_case(row) for row in rows]

    def get_low_confidence(self, threshold: float = 0.5, limit: int = 100) -> List[QuarantineCase]:
        """Get cases with very low confidence."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM quarantine_cases
                WHERE confidence < ?
                ORDER BY confidence ASC
                LIMIT ?
            """, (threshold, limit)).fetchall()
            return [self._row_to_case(row) for row in rows]

    def update_review(
        self,
        case_id: str,
        status: ReviewStatus,
        reviewer: str,
        corrected_label: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update the review status of a case.

        Returns True if updated, False if case not found.
        """
        with self._get_connection() as conn:
            result = conn.execute("""
                UPDATE quarantine_cases
                SET review_status = ?,
                    reviewer = ?,
                    review_timestamp = ?,
                    corrected_label = ?,
                    review_notes = ?
                WHERE case_id = ?
            """, (
                status.value,
                reviewer,
                datetime.utcnow().isoformat() + "Z",
                corrected_label,
                notes,
                case_id,
            ))
            return result.rowcount > 0

    def get_stats(self) -> CaptureStats:
        """Get statistics about captured cases."""
        stats = CaptureStats()

        with self._get_connection() as conn:
            # Total counts
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN review_status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN review_status != 'pending' THEN 1 ELSE 0 END) as reviewed,
                    SUM(CASE WHEN review_status = 'corrected' THEN 1 ELSE 0 END) as corrected,
                    AVG(confidence) as avg_conf,
                    MIN(confidence) as min_conf,
                    MAX(confidence) as max_conf,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM quarantine_cases
            """).fetchone()

            stats.total_captured = row["total"] or 0
            stats.pending_review = row["pending"] or 0
            stats.reviewed = row["reviewed"] or 0
            stats.corrected = row["corrected"] or 0
            stats.avg_confidence = row["avg_conf"] or 0.0
            stats.min_confidence = row["min_conf"] or 1.0
            stats.max_confidence = row["max_conf"] or 0.0
            stats.earliest = row["earliest"]
            stats.latest = row["latest"]

            # By reason
            rows = conn.execute("""
                SELECT capture_reason, COUNT(*) as count
                FROM quarantine_cases
                GROUP BY capture_reason
            """).fetchall()
            stats.by_reason = {row["capture_reason"]: row["count"] for row in rows}

            # By source
            rows = conn.execute("""
                SELECT cascade_source, COUNT(*) as count
                FROM quarantine_cases
                GROUP BY cascade_source
            """).fetchall()
            stats.by_source = {row["cascade_source"]: row["count"] for row in rows}

            # By label
            rows = conn.execute("""
                SELECT final_label, COUNT(*) as count
                FROM quarantine_cases
                GROUP BY final_label
            """).fetchall()
            stats.by_label = {row["final_label"]: row["count"] for row in rows}

        return stats

    def cleanup_old(self, days: int = None) -> int:
        """
        Delete cases older than retention period.

        Returns number of deleted cases.
        """
        days = days or self.config.retention_days
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        with self._get_connection() as conn:
            # Only delete reviewed cases
            result = conn.execute("""
                DELETE FROM quarantine_cases
                WHERE timestamp < ? AND review_status != 'pending'
            """, (cutoff,))
            return result.rowcount

    def _row_to_case(self, row: sqlite3.Row) -> QuarantineCase:
        """Convert database row to QuarantineCase."""
        layer_results_data = json.loads(row["layer_results"]) if row["layer_results"] else []
        layer_results = [
            LayerResult(**lr) if isinstance(lr, dict) else lr
            for lr in layer_results_data
        ]

        return QuarantineCase(
            case_id=row["case_id"],
            timestamp=row["timestamp"],
            input_text=row["input_text"],
            input_hash=row["input_hash"],
            input_length=row["input_length"],
            capture_reason=CaptureReason(row["capture_reason"]),
            cascade_source=CascadeSource(row["cascade_source"]),
            confidence=row["confidence"],
            threshold=row["threshold"],
            final_label=row["final_label"],
            stopped_at=row["stopped_at"],
            layer_results=layer_results,
            total_latency_ms=row["total_latency_ms"] or 0.0,
            review_status=ReviewStatus(row["review_status"]),
            reviewer=row["reviewer"],
            review_timestamp=row["review_timestamp"],
            corrected_label=row["corrected_label"],
            review_notes=row["review_notes"],
            session_id=row["session_id"],
            request_id=row["request_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def count_pending(self) -> int:
        """Get count of pending reviews."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM quarantine_cases WHERE review_status = 'pending'"
            ).fetchone()
            return row["count"]

    def search(
        self,
        query: str,
        limit: int = 100,
    ) -> List[QuarantineCase]:
        """Search cases by input text (simple LIKE search)."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM quarantine_cases
                WHERE input_text LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f"%{query}%", limit)).fetchall()
            return [self._row_to_case(row) for row in rows]
