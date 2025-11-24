"""
Tests for cascade_quarantine Phase 1: Basic Capture System.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (
    QuarantineCase,
    LayerResult,
    CaptureReason,
    CascadeSource,
    ReviewStatus,
    CaptureStats,
)
from config import QuarantineConfig
from database import QuarantineDatabase
from capture import CaptureHook, capture_if_low_confidence


class TestQuarantineCase(unittest.TestCase):
    """Test QuarantineCase dataclass."""

    def test_create_case(self):
        """Test basic case creation."""
        case = QuarantineCase(
            case_id="qua_test123",
            input_text="How do I hack a computer?",
            capture_reason=CaptureReason.LOW_CONFIDENCE,
            cascade_source=CascadeSource.INBOUND,
            confidence=0.65,
            final_label="harmful",
            stopped_at="L1",
        )

        self.assertEqual(case.case_id, "qua_test123")
        self.assertEqual(case.confidence, 0.65)
        self.assertTrue(case.is_low_confidence)
        self.assertTrue(case.needs_review)
        self.assertEqual(case.input_length, len("How do I hack a computer?"))
        self.assertIsNotNone(case.input_hash)

    def test_auto_hash_generation(self):
        """Test input hash is auto-generated."""
        case = QuarantineCase(
            case_id="qua_test456",
            input_text="Test input",
            capture_reason=CaptureReason.LOW_CONFIDENCE,
            cascade_source=CascadeSource.INBOUND,
            confidence=0.5,
            final_label="safe",
            stopped_at="L0",
        )

        self.assertIsNotNone(case.input_hash)
        self.assertEqual(len(case.input_hash), 16)

    def test_case_id_generation(self):
        """Test unique case ID generation."""
        ids = [QuarantineCase.generate_case_id() for _ in range(100)]
        self.assertEqual(len(set(ids)), 100)  # All unique
        for case_id in ids:
            self.assertTrue(case_id.startswith("qua_"))

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        layer_results = [
            LayerResult(layer="L0", label="safe", confidence=0.8, latency_ms=2.5),
            LayerResult(layer="L1", label="harmful", confidence=0.7, latency_ms=500.0),
        ]

        original = QuarantineCase(
            case_id="qua_roundtrip",
            input_text="Test roundtrip",
            capture_reason=CaptureReason.LAYER_DISAGREEMENT,
            cascade_source=CascadeSource.INBOUND,
            confidence=0.7,
            final_label="harmful",
            stopped_at="L1",
            layer_results=layer_results,
            total_latency_ms=502.5,
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = QuarantineCase.from_dict(data)

        self.assertEqual(original.case_id, restored.case_id)
        self.assertEqual(original.input_text, restored.input_text)
        self.assertEqual(original.capture_reason, restored.capture_reason)
        self.assertEqual(original.confidence, restored.confidence)
        self.assertEqual(len(original.layer_results), len(restored.layer_results))


class TestQuarantineDatabase(unittest.TestCase):
    """Test SQLite database operations."""

    def setUp(self):
        """Create temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.config = QuarantineConfig(
            database_path=self.db_path,
            enable_log_file=False,
        )
        self.db = QuarantineDatabase(self.config)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_insert_and_get(self):
        """Test inserting and retrieving a case."""
        case = QuarantineCase(
            case_id="qua_dbtest1",
            input_text="Database test input",
            capture_reason=CaptureReason.LOW_CONFIDENCE,
            cascade_source=CascadeSource.INBOUND,
            confidence=0.6,
            final_label="uncertain",
            stopped_at="L1",
        )

        self.assertTrue(self.db.insert(case))
        retrieved = self.db.get("qua_dbtest1")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.case_id, case.case_id)
        self.assertEqual(retrieved.input_text, case.input_text)
        self.assertEqual(retrieved.confidence, case.confidence)

    def test_duplicate_insert(self):
        """Test duplicate case_id is rejected."""
        case = QuarantineCase(
            case_id="qua_duplicate",
            input_text="First insert",
            capture_reason=CaptureReason.LOW_CONFIDENCE,
            cascade_source=CascadeSource.INBOUND,
            confidence=0.5,
            final_label="safe",
            stopped_at="L0",
        )

        self.assertTrue(self.db.insert(case))
        case.input_text = "Second insert"
        self.assertFalse(self.db.insert(case))

    def test_batch_insert(self):
        """Test batch insertion."""
        cases = [
            QuarantineCase(
                case_id=f"qua_batch{i}",
                input_text=f"Batch test {i}",
                capture_reason=CaptureReason.LOW_CONFIDENCE,
                cascade_source=CascadeSource.INBOUND,
                confidence=0.5 + i * 0.05,
                final_label="safe" if i % 2 == 0 else "harmful",
                stopped_at="L0",
            )
            for i in range(10)
        ]

        inserted = self.db.insert_batch(cases)
        self.assertEqual(inserted, 10)

    def test_get_pending_review(self):
        """Test retrieving pending review cases."""
        # Insert cases with different confidences
        for i, conf in enumerate([0.3, 0.5, 0.7, 0.9]):
            case = QuarantineCase(
                case_id=f"qua_pending{i}",
                input_text=f"Pending test {i}",
                capture_reason=CaptureReason.LOW_CONFIDENCE,
                cascade_source=CascadeSource.INBOUND,
                confidence=conf,
                final_label="uncertain",
                stopped_at="L0",
            )
            self.db.insert(case)

        pending = self.db.get_pending_review(limit=10)

        # Should be ordered by confidence ascending
        self.assertEqual(len(pending), 4)
        self.assertEqual(pending[0].confidence, 0.3)
        self.assertEqual(pending[-1].confidence, 0.9)

    def test_update_review(self):
        """Test updating review status."""
        case = QuarantineCase(
            case_id="qua_review1",
            input_text="Review test",
            capture_reason=CaptureReason.LOW_CONFIDENCE,
            cascade_source=CascadeSource.INBOUND,
            confidence=0.5,
            final_label="uncertain",
            stopped_at="L0",
        )
        self.db.insert(case)

        success = self.db.update_review(
            "qua_review1",
            ReviewStatus.CORRECTED,
            "reviewer@example.com",
            corrected_label="safe",
            notes="False positive - benign request",
        )

        self.assertTrue(success)

        updated = self.db.get("qua_review1")
        self.assertEqual(updated.review_status, ReviewStatus.CORRECTED)
        self.assertEqual(updated.reviewer, "reviewer@example.com")
        self.assertEqual(updated.corrected_label, "safe")

    def test_get_stats(self):
        """Test statistics calculation."""
        # Insert diverse cases
        cases = [
            QuarantineCase(
                case_id="qua_stats1",
                input_text="Stats test 1",
                capture_reason=CaptureReason.LOW_CONFIDENCE,
                cascade_source=CascadeSource.INBOUND,
                confidence=0.5,
                final_label="safe",
                stopped_at="L0",
            ),
            QuarantineCase(
                case_id="qua_stats2",
                input_text="Stats test 2",
                capture_reason=CaptureReason.LAYER_DISAGREEMENT,
                cascade_source=CascadeSource.DLP,
                confidence=0.7,
                final_label="harmful",
                stopped_at="L1",
            ),
        ]

        for case in cases:
            self.db.insert(case)

        stats = self.db.get_stats()

        self.assertEqual(stats.total_captured, 2)
        self.assertEqual(stats.pending_review, 2)
        self.assertEqual(stats.by_reason.get("low_confidence", 0), 1)
        self.assertEqual(stats.by_reason.get("layer_disagreement", 0), 1)
        self.assertEqual(stats.by_source.get("cascade_inbound", 0), 1)
        self.assertEqual(stats.by_source.get("cascade_dlp", 0), 1)

    def test_get_by_hash(self):
        """Test deduplication by input hash."""
        # Same input text should have same hash
        case1 = QuarantineCase(
            case_id="qua_hash1",
            input_text="Duplicate input",
            capture_reason=CaptureReason.LOW_CONFIDENCE,
            cascade_source=CascadeSource.INBOUND,
            confidence=0.5,
            final_label="safe",
            stopped_at="L0",
        )
        case2 = QuarantineCase(
            case_id="qua_hash2",
            input_text="Duplicate input",
            capture_reason=CaptureReason.LOW_CONFIDENCE,
            cascade_source=CascadeSource.INBOUND,
            confidence=0.6,
            final_label="safe",
            stopped_at="L0",
        )

        self.db.insert(case1)
        self.db.insert(case2)

        duplicates = self.db.get_by_hash(case1.input_hash)
        self.assertEqual(len(duplicates), 2)


class TestCaptureHook(unittest.TestCase):
    """Test capture hooks."""

    def setUp(self):
        """Create temporary config."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.log_path = os.path.join(self.temp_dir, "test.jsonl")
        self.config = QuarantineConfig(
            database_path=self.db_path,
            log_file=self.log_path,
            confidence_threshold=0.75,
            audit_sample_rate=0.0,  # Disable random sampling for tests
        )
        self.hook = CaptureHook(self.config)

    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.db_path, self.log_path]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.temp_dir)

    def test_capture_low_confidence(self):
        """Test capturing low confidence results."""
        result = {
            "label": "uncertain",
            "confidence": 0.65,
            "stopped_at": "L1",
            "total_latency_ms": 150.0,
            "layers": [
                {"level": "L0", "label": "safe", "confidence": 0.6, "latency_ms": 2.0},
                {"level": "L1", "label": "uncertain", "confidence": 0.65, "latency_ms": 148.0},
            ],
        }

        case = self.hook.capture_from_inbound("Test low confidence", result)

        self.assertIsNotNone(case)
        self.assertEqual(case.capture_reason, CaptureReason.LOW_CONFIDENCE)
        self.assertEqual(case.confidence, 0.65)
        self.assertEqual(len(case.layer_results), 2)

    def test_no_capture_high_confidence(self):
        """Test high confidence results are not captured."""
        result = {
            "label": "safe",
            "confidence": 0.95,
            "stopped_at": "L0",
            "total_latency_ms": 2.0,
            "layers": [
                {"level": "L0", "label": "safe", "confidence": 0.95, "latency_ms": 2.0},
            ],
        }

        case = self.hook.capture_from_inbound("Test high confidence", result)

        self.assertIsNone(case)

    def test_capture_layer_disagreement(self):
        """Test capturing when layers disagree."""
        self.config.detect_layer_disagreement = True
        self.config.disagreement_confidence_diff = 0.3
        self.hook = CaptureHook(self.config)

        result = {
            "label": "harmful",
            "confidence": 0.85,
            "stopped_at": "L1",
            "total_latency_ms": 150.0,
            "layers": [
                # L0 says safe with 0.5 confidence
                {"level": "L0", "label": "safe", "confidence": 0.5, "latency_ms": 2.0},
                # L1 says harmful with 0.85 confidence - diff is 0.35 > 0.3 threshold
                {"level": "L1", "label": "harmful", "confidence": 0.85, "latency_ms": 148.0},
            ],
        }

        case = self.hook.capture_from_inbound("Test disagreement", result)

        # Should be captured due to disagreement even with OK confidence
        # (0.85 >= 0.75 threshold, but L0 safe != L1 harmful with 0.35 conf diff)
        self.assertIsNotNone(case)
        self.assertEqual(case.capture_reason, CaptureReason.LAYER_DISAGREEMENT)

    def test_capture_borderline(self):
        """Test capturing borderline cases."""
        result = {
            "label": "safe",
            "confidence": 0.75,  # Exactly at threshold
            "stopped_at": "L0",
            "total_latency_ms": 2.0,
            "layers": [
                {"level": "L0", "label": "safe", "confidence": 0.75, "latency_ms": 2.0},
            ],
        }

        case = self.hook.capture_from_inbound("Test borderline", result)

        # 0.75 is borderline (between 0.7 and 0.8)
        self.assertIsNotNone(case)
        self.assertEqual(case.capture_reason, CaptureReason.BORDERLINE_CASE)

    def test_callback_registration(self):
        """Test capture callbacks."""
        captured_cases = []

        def on_capture(case):
            captured_cases.append(case)

        self.hook.on_capture(on_capture)

        result = {
            "label": "uncertain",
            "confidence": 0.5,
            "stopped_at": "L1",
            "layers": [],
        }

        self.hook.capture_from_inbound("Test callback", result)

        self.assertEqual(len(captured_cases), 1)

    def test_log_file_written(self):
        """Test JSONL log file is written."""
        result = {
            "label": "uncertain",
            "confidence": 0.5,
            "stopped_at": "L0",
            "layers": [],
        }

        self.hook.capture_from_inbound("Test log write", result)

        self.assertTrue(os.path.exists(self.log_path))
        with open(self.log_path, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)


class TestCaptureFromDLP(unittest.TestCase):
    """Test DLP-specific capture."""

    def setUp(self):
        """Create temporary config."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = QuarantineConfig(
            database_path=os.path.join(self.temp_dir, "test.db"),
            enable_log_file=False,
            confidence_threshold=0.75,
            audit_sample_rate=0.0,
        )
        self.hook = CaptureHook(self.config)

    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_capture_dlp_low_confidence_detection(self):
        """Test capturing DLP detections with low confidence."""
        result = {
            "action": "redact",
            "blocked": False,
            "redacted": True,
            "detection_count": 1,
            "total_time_ms": 5.0,
            "detections": [
                {
                    "detector_name": "pii_detector",
                    "entity_type": "email",
                    "text": "test@example.com",
                    "start": 10,
                    "end": 26,
                    "confidence": 0.6,
                },
            ],
        }

        case = self.hook.capture_from_dlp("Contact test@example.com", result)

        self.assertIsNotNone(case)
        self.assertEqual(case.cascade_source, CascadeSource.DLP)
        self.assertEqual(case.confidence, 0.6)


if __name__ == "__main__":
    unittest.main()
