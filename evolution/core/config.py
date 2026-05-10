"""Configuration and skill-source discovery.

Skill discovery moved from a single hardcoded Hermes Agent layout to a
pluggable SkillSource list (see evolution/core/skill_sources.py).
The default list is built by sniffing the environment: HERMES_AGENT_REPO,
~/.claude/plugins/cache, plus any explicit --skill-source-dir from the CLI.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from evolution.core.skill_sources import SkillSource, discover_skill_sources


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    skill_sources: list[SkillSource] = field(
        default_factory=lambda: discover_skill_sources()
    )

    iterations: int = 10
    population_size: int = 5

    optimizer_model: str = "openai/gpt-4.1"
    # Decagon's blog: gpt-4o-mini "failed completely" as reflection LM —
    # reflection is where model strength meaningfully changes outcomes.
    # CLI overrides this default to gpt-5-mini.
    reflection_model: Optional[str] = None
    eval_model: str = "openai/gpt-4.1-mini"
    judge_model: str = "openai/gpt-4.1"

    max_skill_size: int = 15_000
    max_tool_desc_size: int = 500
    max_param_desc_size: int = 200
    # required(growth) = max(0, slope * (growth - free)).
    growth_free_threshold: float = 0.20
    growth_quality_slope: float = 0.30
    bap_max_growth: float = 0.20
    # Backstop for short baselines that legitimately need expansion —
    # a 200-char baseline growing to 1500 is +650% but only 1500 absolute.
    max_absolute_chars: int = 5000
    # Decision rule when required_improvement == 0 (growth ≤ free threshold).
    # "no_regression": pass when bootstrap.mean ≥ 0 (default; safer).
    # "non_inferiority": pass when bootstrap.lower_bound > -inferiority_tolerance
    # (Decagon-style; ships variants statistically not-worse-than-baseline by
    # more than the tolerance — necessary at small N where bootstrap CI swamps
    # tiny effects).
    gate_mode: str = "no_regression"
    inferiority_tolerance: float = 0.0
    fitness_profile: str = "balanced"
    # Basic (reverse percentile) bootstrap is the literature-recommended
    # method when N is small. BCa is the upgrade path once N≥20 routinely.
    bootstrap_confidence: float = 0.90
    bootstrap_n_resamples: int = 2000

    # Sized so the normalized 0.36/0.29/0.36 split lands ~54/43/53 — the
    # ~53-example holdout keeps the paired-bootstrap CI tight enough to
    # detect ±2% effects. Smaller N produces CIs too wide to gate on.
    eval_dataset_size: int = 150
    train_ratio: float = 0.5
    val_ratio: float = 0.40
    holdout_ratio: float = 0.50
    # Below this the bootstrap lower bound has too little resolution to
    # gate on. Raise eval_dataset_size or holdout_ratio rather than override.
    min_holdout_size: int = 10

    run_pytest: bool = True
    run_tblite: bool = False
    tblite_regression_threshold: float = 0.02

    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True

    seed: int = 42


