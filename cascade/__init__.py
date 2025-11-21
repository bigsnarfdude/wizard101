"""
Safety Cascade - Multi-tier Content Safety Classification

Cascata Fiduciae Fundata (Cascade of Founded Trust)

A hierarchical safety classification system with 4 tiers:
- L0: Fast DeBERTa bouncer (22M params, ~6ms)
- L1: Reasoning analyst with Llama 3.2 3B (3B params, ~50ms)
- L2: Multi-expert voting gauntlet (6 experts, ~200ms)
- L3: Final judge with large model (120B params, async)
"""

from .cascade import SafetyCascade, CascadeConfig, CascadeResult
from .l0_bouncer import L0Bouncer
from .l1_analyst import L1Analyst
from .l2_gauntlet import L2Gauntlet
from .l3_judge import L3Judge

__version__ = "0.1.0"
__all__ = [
    "SafetyCascade",
    "CascadeConfig",
    "CascadeResult",
    "L0Bouncer",
    "L1Analyst",
    "L2Gauntlet",
    "L3Judge",
]
