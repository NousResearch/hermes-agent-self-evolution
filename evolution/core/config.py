"""Configuration and hermes-agent repo discovery."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from evolution.core.objectives import ObjectiveWeights


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path. Discovered lazily and non-fatally so the config
    # can be constructed even when no repo is present (e.g. unit tests, or
    # callers that pass an explicit path). Use resolve_hermes_agent_path() when
    # an explicit override should win, or get_hermes_agent_path() to require one.
    hermes_agent_path: Optional[Path] = field(default_factory=lambda: _discover_hermes_agent_path())

    # Hermes *data* directory (~/.hermes or the container's bind mount). This is
    # separate from the repo: the repo holds code and shipped skills, the data
    # dir holds state.db, profiles, cron and the user's own skills. Resolved by
    # evolution.core.hermes_paths, never from Path.home() at the call site.
    hermes_data_dir: Optional[str] = None

    # Restrict mining to specific profiles. None means every profile found.
    profiles: Optional[list[str]] = None

    # Optimization parameters
    iterations: int = 10

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"  # Model for GEPA reflections
    eval_model: str = "openai/gpt-4.1-mini"  # Model for LLM-as-judge scoring
    judge_model: str = "openai/gpt-4.1"  # Model for dataset generation

    # Custom base URL for local models (e.g., vLLM, Ollama)
    api_base: Optional[str] = None  # e.g., "http://localhost:8000/v1"
    api_key: Optional[str] = None  # e.g., "sk_test_key"

    # Constraints. max_skill_size is only the fallback — the real budget is
    # derived from the installed skill corpus, because a fixed 15KB cap
    # rejects 27 of the 201 shipped skills at their own baseline.
    max_skill_size: int = 15_000
    size_percentile: int = 90  # corpus percentile used as the budget
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline

    # Objective weights for the multi-objective scalarization.
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    # Holdout is deliberately not a field: the split takes it as the
    # remainder, so a holdout_ratio knob would silently do nothing whatever it
    # was set to.
    train_ratio: float = 0.5
    val_ratio: float = 0.25

    # Agent-in-the-loop evaluation. When enabled, candidates are scored by
    # running the real Hermes AIAgent rather than a single completion.
    agent_eval: bool = False
    agent_eval_reps: int = 1
    agent_toolsets: tuple[str, ...] = ("file", "terminal", "search")

    # Gating
    run_pytest: bool = False  # run the hermes-agent test suite before deploy
    test_timeout_s: int = 300

    # Deployment
    create_pr: bool = False
    pr_base_branch: str = "main"
    pr_draft: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    def __post_init__(self) -> None:
        # Holdout is what is left over, so ratios that sum to 1.0 or more leave
        # nothing to evaluate on and the run reports a delta with no basis.
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) "
                "must be under 1.0 — the holdout split is the remainder."
            )

    def resolved_output_dir(self) -> Path:
        """Output root, honoring EVOLUTION_OUTPUT_DIR when set."""
        env = os.getenv("EVOLUTION_OUTPUT_DIR")
        if env:
            return Path(env).expanduser()
        return Path(self.output_dir)


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
    2. HERMES_AGENT_SOURCE_REPO env var (set by the containerized runner)
    3. ~/.hermes/hermes-agent (standard install location)
    4. ../hermes-agent (sibling directory)
    """
    for env_var in ("HERMES_AGENT_REPO", "HERMES_AGENT_SOURCE_REPO"):
        env_path = os.getenv(env_var)
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


def skill_search_roots(
    config: EvolutionConfig,
    install=None,
    profile: Optional[str] = None,
) -> list[Path]:
    """Every directory that might hold the skill being evolved.

    Order matters: profile-specific skills win over user-tree skills, which
    win over the ones shipped in the repo, because that is the precedence the
    running agent applies.
    """
    roots: list[Path] = []

    if install is not None:
        try:
            if profile:
                roots.append(install.profile(profile).skills_dir)
            else:
                for prof in install.profiles():
                    roots.append(prof.skills_dir)
        except Exception:  # noqa: BLE001 — a bad profile name must not break discovery
            pass
        roots.append(install.skills_dir)

    if config.hermes_agent_path:
        roots.append(Path(config.hermes_agent_path) / "skills")
        roots.append(Path(config.hermes_agent_path) / "optional-skills")

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root and root.is_dir() and root not in seen:
            seen.add(root)
            unique.append(root)
    return unique
