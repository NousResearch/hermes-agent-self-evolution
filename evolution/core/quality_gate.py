"""Artifact-agnostic quality-gate helpers extracted from ``evolve_skill``.

These helpers — preset table, proposer-mode resolution, and gate-decision
persistence — were never skill-specific. Hosting them in ``evolution.core``
lets the upcoming tool description pipeline reuse the deploy-gate scaffolding
without depending on the skill pipeline.
"""

import json
from pathlib import Path
from typing import Any

from evolution.skills.budget_aware_proposer import ProposerMode


# `default` is calibrated against the obsidian deploy (+24.2% growth,
# ~+0.07 expected improvement). `off` disables the slope/ceiling checks
# but still enforces bootstrap.mean ≥ 0 — see deprecation warning when
# users select it. `non-inferiority` is the recommended preset for
# compression-focused runs: it ships variants statistically not-worse-
# than-baseline by more than ``inferiority_tolerance``.
#
# Type widens to ``Any`` because ``gate_mode`` is a string and
# ``inferiority_tolerance`` is a float — no longer a uniform float dict.
QUALITY_GATE_PRESETS: dict[str, dict[str, Any]] = {
    "strict": {
        "growth_free_threshold": 0.10,
        "growth_quality_slope": 0.50,
        "max_absolute_chars": 3000,
    },
    "default": {
        "growth_free_threshold": 0.20,
        "growth_quality_slope": 0.30,
        "max_absolute_chars": 5000,
    },
    "lenient": {
        "growth_free_threshold": 0.30,
        "growth_quality_slope": 0.20,
        "max_absolute_chars": 8000,
    },
    "off": {
        "growth_free_threshold": 100.0,
        "growth_quality_slope": 0.0,
        "max_absolute_chars": 100_000,
    },
    "non-inferiority": {
        "growth_free_threshold": 100.0,
        "growth_quality_slope": 0.0,
        "max_absolute_chars": 100_000,
        "gate_mode": "non_inferiority",
        # See reports/calibration_findings.md for the tolerance sweep.
        "inferiority_tolerance": 0.05,
    },
}


def write_gate_decision(output_dir: Path, decision: dict[str, Any]) -> Path:
    """Persist the deploy-gate's structured decision for future calibration.

    Each run writes one of these regardless of outcome (deploy or reject).
    Recalibrating the curve is then `jq -s '...' output/*/*/gate_decision.json`
    rather than parsing free-form failure notes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gate_decision.json"
    path.write_text(json.dumps(decision, indent=2))
    return path


def resolve_proposer_mode(fitness_profile: str) -> ProposerMode:
    """Map the user's fitness profile to a proposer mode.

    Each profile selects its own template: `growth` swings the proposer toward
    additions, `compression` toward cuts, and `balanced` toward direction-agnostic
    revisions. Unknown values fall back to compression-mode defensively.
    """
    if fitness_profile == "growth":
        return "growth"
    if fitness_profile == "balanced":
        return "balanced"
    return "compression"  # compression profile, plus defensive fallback for unknown
