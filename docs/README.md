# Documentation

## Architecture

The wizard101 system implements a 5-layer AI safety cascade:

```
User Input
    ↓
[Layer 1: Inbound]     → Classify: Is this harmful?
    ↓ (if harmful)
[Layer 2: Refusals]    → Generate appropriate refusal
    ↓ (if safe)
[Layer 3: Quarantine]  → Sanitize untrusted input
    ↓
[Layer 4: LLM]         → Your model (external)
    ↓
[Layer 5: DLP]         → Prevent data leakage
    ↓
User Response
```

## Component Documentation

| Layer | Component | Documentation |
|-------|-----------|---------------|
| 1 | Inbound Safety | [cascade_inbound/README.md](../cascade_inbound/README.md) |
| 2 | Refusal Generation | [cascade_refusals/README.md](../cascade_refusals/README.md) |
| 3 | Input Quarantine | [cascade_quarantine/README.md](../cascade_quarantine/README.md) |
| 5 | Data Loss Prevention | [cascade_dlp/README.md](../cascade_dlp/README.md) |

## Quick Links

- [Main README](../README.md) - Project overview and getting started
- [Scripts README](../scripts/README.md) - Training and evaluation scripts
- [Archive README](../archive/README.md) - Archived experimental code

## Research & Results

Historical research documents are preserved in:
- `archive/2024-11-experiments/docs/` - Research notes, comparison summaries
- `archive/2024-11-dlp-old/docs/` - DLP architecture decisions

Key documents:
- `SESSION_LEARNINGS.md` - Key insights from development
- `L2_SOLUTION.md` - Architecture decisions for L2 Gauntlet
- `DLP_ARCHITECTURE.md` - DLP design rationale
