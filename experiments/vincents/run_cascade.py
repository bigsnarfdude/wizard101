#!/usr/bin/env python3
"""
Vincent's Cascade Evaluation Runner

Self-contained script to run the full cascade evaluation.
All outputs go to ./outputs/

Usage:
    cd experiments/vincents
    source ../guardreasoner/venv/bin/activate
    python run_cascade.py

    # Or individual layers:
    python run_cascade.py --layer l0
    python run_cascade.py --layer l1
    python run_cascade.py --layer l2
    python run_cascade.py --combine
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set output directory to local outputs folder
os.environ['CASCADE_OUTPUT_DIR'] = str(Path(__file__).parent / 'outputs')

from eval_layered_batch import main

if __name__ == "__main__":
    main()
