"""Configuration and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path. Discovered lazily and non-fatally so the config
    # can be constructed even when no repo is present (e.g. unit tests, or
    # callers that pass an explicit path). Use resolve_hermes_agent_path() when
    # an explicit override should win, or get_hermes_agent_path() to require one.
    hermes_agent_path: Optional[Path] = field(default_factory=lambda: _discover_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"  # Model for GEPA reflections
    eval_model: str = "openai/gpt-4.1-mini"  # Model for LLM-as-judge scoring
    judge_model: str = "openai/gpt-4.1"  # Model for dataset generation

    # Constraints — size is class-aware (see resolve_skill_soft_cap) and graduated.
    # Rationale: a single flat cap over-penalizes persistent reference/runbook
    # skills (which legitimately encode many endpoints, parameters and pitfalls,
    # and are meant to survive context compression) purely for size; and the old
    # penalty ramp (which started at 90% of the cap) even docked skills that were
    # already *under* the cap. New policy:
    #   - soft target: no length penalty at/below it (class-aware)
    #   - hard ceiling: rejection above it (bounds genuine bloat)
    #   - graduated penalty linearly between soft and ceiling (no pre-cap cliff)
    max_skill_size: int = 15_000  # soft target for default (generative/prompt) skills
    max_skill_size_reference: int = 22_000  # soft target for persistent reference/runbook skills
    max_skill_hard_ceiling: int = 30_000  # absolute rejection ceiling for any skill
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline (per-mutation, default skills)
    max_prompt_growth_reference: float = 0.6  # reference skills may restructure more aggressively

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # Benchmark gating
    run_pytest: bool = True
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% regression allowed

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True


# --- Class-aware size policy ----------------------------------------------

# Signals (in frontmatter/body) that a skill is a persistent reference/runbook
# rather than a generative/prompt skill. Such skills legitimately encode many
# endpoints, parameters, and pitfalls and are meant to "survive compression",
# so they get a larger soft target before any length penalty applies.
_REFERENCE_TAGS = ("reference", "runbook", "workflow", "persistent", "operations", "pipeline")
_REFERENCE_PHRASES = ("survives context compression", "persistent", "master reference", "every api endpoint")


def is_reference_skill(skill_text: str) -> bool:
    """Heuristically classify a skill as a persistent reference/runbook.

    Looks at the frontmatter tags and the description/opening for the documented
    markers of a reference skill. Conservative: defaults to False (the tighter
    default cap) when no clear signal is present.
    """
    if not skill_text:
        return False
    head = skill_text[:1500].lower()
    if any(tag in head for tag in _REFERENCE_TAGS):
        return True
    if any(phrase in head for phrase in _REFERENCE_PHRASES):
        return True
    return False


def resolve_skill_soft_cap(skill_text: str, config: "EvolutionConfig") -> int:
    """Return the no-penalty soft size target for this skill's class."""
    return config.max_skill_size_reference if is_reference_skill(skill_text) else config.max_skill_size


def length_penalty_for(size: int, soft_cap: int, hard_ceiling: int, max_penalty: float = 0.3) -> float:
    """Graduated length penalty: 0 at/below soft_cap, ramping linearly to
    max_penalty at hard_ceiling, then clamped. No penalty before the cap."""
    if size <= soft_cap or hard_ceiling <= soft_cap:
        return 0.0
    over = (size - soft_cap) / (hard_ceiling - soft_cap)
    return min(max_penalty, max(0.0, over) * max_penalty)


def _discover_hermes_agent_path() -> Optional[Path]:
    """Best-effort hermes-agent repo discovery that never raises.

    Returns the discovered path, or None when no repo can be found. Used as
    the EvolutionConfig default so construction never crashes; callers that
    truly require the repo should use get_hermes_agent_path().
    """
    try:
        return get_hermes_agent_path()
    except FileNotFoundError:
        return None


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )


def resolve_hermes_agent_path(hermes_repo: Optional[str] = None) -> Path:
    """Return the hermes-agent repo path, honoring an explicit override.

    An explicit path (for example from ``--hermes-repo``) is expanded and used
    as-is, taking precedence over auto-discovery. This lets callers point at a
    repo in a non-default location without the tool crashing just because
    ``~/.hermes/hermes-agent`` happens to be absent. When no override is given,
    falls back to :func:`get_hermes_agent_path`.
    """
    if hermes_repo:
        return Path(hermes_repo).expanduser()
    return get_hermes_agent_path()
