"""Configuration and hermes-agent repo discovery."""

import os
import re
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
    hermes_agent_path: Optional[Path] = field(default_factory=lambda: _resolve_hermes_agent_path())

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

    # Semantic preservation: evolved text must retain a minimum share of the
    # baseline's content vocabulary, so a skill cannot drift into a different
    # domain (PLAN.md guardrail #4). Deterministic and free -- it is a drift
    # gate, not a quality score; see constraints._check_semantic_preservation.
    semantic_preservation_enabled: bool = True
    min_semantic_similarity: float = 0.5  # fraction of baseline content terms that must survive

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


def _resolve_hermes_agent_path() -> Optional[Path]:
    """Best-effort hermes-agent repo discovery that never raises.

    Returns the discovered path, or None when no repo can be found. Used as
    the EvolutionConfig default so construction never crashes; callers that
    truly require the repo should use get_hermes_agent_path().
    """
    try:
        return get_hermes_agent_path()
    except FileNotFoundError:
        return None


def local_router_base_url() -> Optional[str]:
    """Read the codex-router base URL from ~/.codex/config.toml, if present.

    The router is the only local backend that speaks the Responses API, and it
    serves *only* `/responses` — `/chat/completions` returns
    `proxy_route_not_found`. DSPy/litellm default to chat completions, so the
    caller must pair this base with `model_type="responses"`.

    Returns None when no router route is configured, so callers fall back to
    whatever provider they were already using.
    """
    cfg = Path.home() / ".codex" / "config.toml"
    if not cfg.exists():
        return None
    try:
        m = re.search(
            r'codex-router/([A-Za-z0-9_\-]+)/v1',
            cfg.read_text(),
        )
    except OSError:
        return None
    if not m:
        return None
    return f"http://127.0.0.1:4222/_codex-router/{m.group(1)}/v1"


def normalize_model(model: str) -> str:
    """Make a model string routable through the local codex-router.

    litellm inspects the `provider/model` prefix and, for a known provider like
    `openrouter/`, discards `api_base` and calls that vendor directly — which
    fails with an auth error instead of reaching the local router. Prefixing
    with `openai/` keeps litellm on the OpenAI-compatible path, where `api_base`
    is honoured.

    Idempotent: an existing `openai/` prefix is left alone.
    """
    if model.startswith("openai/"):
        return model
    return f"openai/{model}"


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
