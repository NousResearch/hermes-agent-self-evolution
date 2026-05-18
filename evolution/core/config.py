"""Configuration and hermes-agent repo discovery."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path
    hermes_agent_path: Path = field(default_factory=lambda: get_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"  # Model for GEPA reflections
    eval_model: str = "openai/gpt-4.1-mini"  # Model for LLM-as-judge scoring
    judge_model: str = "openai/gpt-4.1"  # Model for dataset generation

    # Constraints
    max_skill_size: int = 15_000  # 15KB default
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline

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

    # Phase 0 hardening
    phase0_enforce: bool = True
    max_phase_budget_usd: float = 25.0
    max_phase_runtime_minutes: int = 120
    minimum_detectable_effect: float = 0.02
    required_stable_runs: int = 3

    # Phase 1 gate
    phase1_min_relative_gain: float = 0.10
    phase1_min_absolute_gain: float = 0.03
    phase1_max_benchmark_regression: float = 0.02
    phase1_require_reproducibility_manifest: bool = True
    phase1_require_zero_safety_incidents: bool = True
    minimum_model_families: int = 3


def _candidate_hermes_paths() -> list[Path]:
    """Return candidate hermes-agent locations in priority order."""
    env_path = os.getenv("HERMES_AGENT_REPO")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            Path.home() / ".hermes" / "hermes-agent",
            repo_root / "hermes-agent",
            repo_root.parent / "hermes-agent",
            Path.cwd() / "hermes-agent",
        ],
    )
    return candidates


def discover_hermes_agent_path() -> Path | None:
    """Discover the hermes-agent path without raising."""
    for candidate in _candidate_hermes_paths():
        if candidate.exists():
            return candidate
    return None


def get_hermes_agent_path(*, strict: bool = False) -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. repo-local ./hermes-agent
    4. repo sibling ../hermes-agent
    5. ./hermes-agent from current working directory

    Args:
        strict: If True, raise when not found. If False, return the
            default ~/.hermes/hermes-agent path even when missing.
    """
    discovered = discover_hermes_agent_path()
    if discovered:
        return discovered

    if strict:
        raise FileNotFoundError(
            "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
            "or ensure it exists at ~/.hermes/hermes-agent",
        )

    return Path.home() / ".hermes" / "hermes-agent"
