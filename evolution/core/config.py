"""Configuration and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


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
    max_skill_size: int = 20_000  # 20KB default (was 15KB — NLAH frontmatter adds size)
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.5  # 50% max growth over baseline (was 0.2)
    # Hard absolute growth cap — never exceeded, even with a waiver. Prevents
    # unbounded bloat: an artifact may only grow past max_prompt_growth when a
    # material valset improvement (growth_waiver_min_improvement) justifies it.
    max_prompt_growth_hard: float = 1.0  # 100% max growth, absolute ceiling
    # Minimum absolute score delta (on the 0-1 metric scale) an evolved
    # artifact must beat the baseline by to earn a growth waiver.
    growth_waiver_min_improvement: float = 0.03

    # Random seed for reproducibility
    random_seed: int = 42

    # Eval dataset
    eval_dataset_size: int = 60  # Base examples for small skills; auto-scaled up
    base_eval_dataset_size: int = 60  # Minimum examples before scaling by skill size
    dataset_size_per_10k_chars: int = 10  # Extra examples per 10K chars beyond first 5K
    max_eval_dataset_size: int = 150  # Cap on total scaled dataset size
    eval_temperature: float = 0.0  # 0 = deterministic generation for stable scoring
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
