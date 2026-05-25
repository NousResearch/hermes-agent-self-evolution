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

from evolution.core.hermes_provider import ResolvedLM, Role, resolve_default_lm
from evolution.core.skill_sources import SkillSource, discover_skill_sources


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    skill_sources: list[SkillSource] = field(
        default_factory=lambda: discover_skill_sources()
    )

    iterations: int = 10
    population_size: int = 5

    # GEPA's reflective minibatch size — the number of training examples
    # sampled per reflective step for the sum() acceptance gate at
    # gepa/core/engine.py:491-493. Default 3 matches GEPA's own default
    # (no behavior change). Users hitting the weak_signal saturation
    # band can bump this to ~8 to widen the sampling window so
    # discriminating examples appear more often per minibatch — see
    # reports/pareto_frontier_feasibility.md spike #2 for the
    # motivating case and saturation_check.py's weak_signal suggestions
    # for the actionable hint surfaced to users.
    reflection_minibatch_size: int = 3

    # GEPA acceptance criterion. "improvement_or_equal" (default) accepts
    # plateau-equal candidates so noisy LM-judge ties don't reject "true
    # zero-difference" mutations ~50% of the time; "strict_improvement"
    # preserves the gepa<0.1.2 implicit behavior. Forwarded as the literal
    # kwarg expected by gepa.optimize via dspy.GEPA's gepa_kwargs passthrough
    # (valid gepa values: "strict_improvement", "improvement_or_equal").
    gepa_acceptance: str = "improvement_or_equal"

    # Per-role model overrides. When set, treated as explicit LiteLLM model
    # strings that bypass Hermes resolution. When None, get_lm() falls back
    # to resolve_default_lm() against ~/.hermes/config.yaml + auth.json +
    # provider env vars. Field type stays str-or-None for backward
    # compatibility with callers that pass model strings directly.
    optimizer_model: Optional[str] = None
    reflection_model: Optional[str] = None
    eval_model: Optional[str] = None
    judge_model: Optional[str] = None

    def get_lm(self, role: Role) -> ResolvedLM:
        """Return the ResolvedLM for the given role.

        Reads the ``<role>_model`` override field; if set, treats it as an
        explicit LiteLLM model string. If unset, resolves from Hermes config
        via ``resolve_default_lm``. The reflection role falls back to the
        optimizer's resolved model when its own override is unset.
        """
        explicit = getattr(self, f"{role}_model", None)
        if not explicit and role == "reflection":
            explicit = self.optimizer_model
        return resolve_default_lm(role=role, explicit_model=explicit)

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

    # Off by default: the tool-selection dataset's confusable_neighbor bucket
    # is opt-in. When False the bucket's allocation rolls into target_correct.
    enable_confusable_bucket: bool = False

    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True

    seed: int = 42


