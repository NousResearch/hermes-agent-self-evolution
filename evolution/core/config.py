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
    optimizer_model: str = field(default_factory=lambda: DEFAULT_OPTIMIZER_MODEL)
    eval_model: str = field(default_factory=lambda: DEFAULT_EVAL_MODEL)
    judge_model: str = field(default_factory=lambda: DEFAULT_JUDGE_MODEL)

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

    # Semantic preservation
    enable_semantic_check: bool = True
    semantic_similarity_threshold: float = 0.7  # Min cosine similarity
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True


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


# Known good default models per capability tier
DEFAULT_EVAL_MODEL = os.getenv("HERMES_EVAL_MODEL", "openai/gpt-4o-mini")
DEFAULT_OPTIMIZER_MODEL = os.getenv("HERMES_OPTIMIZER_MODEL", "openai/gpt-4o-mini")
DEFAULT_JUDGE_MODEL = os.getenv("HERMES_JUDGE_MODEL", "openai/gpt-4o-mini")

# Alias map for common model names
MODEL_ALIASES = {
    "hermes-agent": DEFAULT_OPTIMIZER_MODEL,
    "hermes": DEFAULT_OPTIMIZER_MODEL,
    "default": DEFAULT_OPTIMIZER_MODEL,
}


def resolve_model(model_name: str) -> str:
    """Resolve a model name, handling aliases and env-var overrides.

    If model_name matches an alias key (case-insensitive), the alias
    value is returned. Otherwise model_name is returned as-is.
    """
    key = model_name.lower().strip()
    return MODEL_ALIASES.get(key, model_name)


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
