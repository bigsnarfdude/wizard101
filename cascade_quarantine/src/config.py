"""
Configuration for cascade_quarantine.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class QuarantineConfig:
    """Configuration for the quarantine capture system."""

    # Capture thresholds
    confidence_threshold: float = 0.75          # Capture if confidence < this
    audit_sample_rate: float = 0.01             # % of high-confidence to sample

    # Database settings
    database_path: str = "logs/quarantine.db"   # SQLite database path
    enable_database: bool = True                # Store to database

    # JSONL log settings
    log_file: str = "logs/quarantine_captures.jsonl"  # JSONL backup
    enable_log_file: bool = True                # Write to JSONL file

    # Capture settings per cascade
    capture_from_inbound: bool = True           # Capture from cascade_inbound
    capture_from_refusals: bool = True          # Capture from cascade_refusals
    capture_from_dlp: bool = True               # Capture from cascade_dlp

    # Layer disagreement detection
    detect_layer_disagreement: bool = True      # Flag when L0 != L1
    disagreement_confidence_diff: float = 0.3   # Min diff to flag disagreement

    # Review settings
    auto_escalate_harmful: bool = True          # Auto-escalate harmful with low conf
    max_pending_review: int = 1000              # Alert when pending > this

    # Performance
    batch_insert_size: int = 100                # Batch DB inserts
    async_logging: bool = False                 # Use async logging (future)

    # Retention
    retention_days: int = 90                    # How long to keep cases

    def __post_init__(self):
        """Ensure directories exist."""
        # Create parent directories for database
        if self.database_path:
            db_dir = os.path.dirname(self.database_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        # Create parent directories for log file
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> "QuarantineConfig":
        """Create config from environment variables."""
        return cls(
            confidence_threshold=float(os.getenv("QUARANTINE_CONFIDENCE_THRESHOLD", "0.75")),
            audit_sample_rate=float(os.getenv("QUARANTINE_AUDIT_SAMPLE_RATE", "0.01")),
            database_path=os.getenv("QUARANTINE_DATABASE_PATH", "logs/quarantine.db"),
            log_file=os.getenv("QUARANTINE_LOG_FILE", "logs/quarantine_captures.jsonl"),
            retention_days=int(os.getenv("QUARANTINE_RETENTION_DAYS", "90")),
        )


# Default configuration instance
DEFAULT_CONFIG = QuarantineConfig()
