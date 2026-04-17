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

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"  # Model for mutations/crossover
    eval_model: str = "openai/gpt-4.1-mini"  # Model for evaluation/judging
    judge_model: str = "openai/gpt-4.1"  # Model for dataset generation
    api_base: Optional[str] = None  # Custom OpenAI-compatible endpoint
    api_key: Optional[str] = None  # API key (prefer env var over this)

    # EA parameters
    num_islands: int = 3
    population_size: int = 8
    num_generations: int = 10
    migration_interval: int = 2
    elite_count: int = 2
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    stagnation_limit: int = 4
    fast_screen_threshold: float = 0.3

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
    run_tblite: bool = False
    tblite_regression_threshold: float = 0.02

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
