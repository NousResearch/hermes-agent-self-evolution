"""Configuration and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path
    hermes_agent_path: Path | None = field(default_factory=lambda: _maybe_get_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # Execution backend
    execution_backend: str = field(default_factory=lambda: os.getenv("HERMES_EVOLUTION_EXECUTION_BACKEND", "codex-batch"))
    allow_live_model: bool = field(default_factory=lambda: _env_bool("HERMES_EVOLUTION_ALLOW_LIVE_MODEL", False))
    max_codex_calls: int = field(default_factory=lambda: int(os.getenv("HERMES_EVOLUTION_MAX_CODEX_CALLS", "3")))
    max_examples: int = field(default_factory=lambda: int(os.getenv("HERMES_EVOLUTION_MAX_EXAMPLES", "8")))
    phase_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("HERMES_EVOLUTION_PHASE_TIMEOUT_SECONDS", "180")))
    max_run_seconds: int = field(default_factory=lambda: int(os.getenv("HERMES_EVOLUTION_MAX_RUN_SECONDS", "600")))
    max_candidates_per_iteration: int = field(default_factory=lambda: int(os.getenv("HERMES_EVOLUTION_MAX_CANDIDATES_PER_ITERATION", "1")))
    budget_strict: bool = field(default_factory=lambda: _env_bool("HERMES_EVOLUTION_BUDGET_STRICT", True))
    codex_bin: str = field(default_factory=lambda: os.getenv("HERMES_EVOLUTION_CODEX_BIN", "codex"))

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"
    eval_model: str = "openai/gpt-4.1-mini"
    judge_model: str = "openai/gpt-4.1"
    lm_api_base: str = field(default_factory=lambda: _first_set_env(
        "HERMES_EVOLUTION_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    ))
    lm_api_key: str = field(default_factory=lambda: _first_set_env(
        "HERMES_EVOLUTION_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    ))
    lm_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("HERMES_EVOLUTION_LM_TIMEOUT", "180")))

    # Constraints
    max_skill_size: int = 15_000
    max_tool_desc_size: int = 500
    max_param_desc_size: int = 200
    max_prompt_growth: float = 0.2

    # Eval dataset
    eval_dataset_size: int = 20
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

    def resolve_hermes_agent_path(self) -> Path:
        """Return a usable hermes-agent path or raise a clear error."""
        if self.hermes_agent_path is not None:
            return self.hermes_agent_path
        return get_hermes_agent_path()


def _first_set_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_get_hermes_agent_path() -> Path | None:
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
