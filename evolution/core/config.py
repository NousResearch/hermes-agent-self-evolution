"""Configuration and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import dspy


# MiniMax model IDs supported via the OpenAI-compatible endpoint.
# Only Chat models — MiniMax has no Embedding model.
MINIMAX_MODELS = (
    "MiniMax-M3",              # Latest: 512K context, 128K max output, image input. Default.
    "MiniMax-M2.7",            # Previous generation.
    "MiniMax-M2.7-highspeed",  # Previous generation, lower latency.
)

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


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

    # MiniMax provider configuration
    minimax_api_key: str = field(default_factory=lambda: os.getenv("MINIMAX_API_KEY", ""))
    minimax_base_url: str = MINIMAX_BASE_URL

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

    def make_lm(self, model: str) -> "dspy.LM":
        """Create a DSPy LM instance, with MiniMax routing handled automatically.

        For MiniMax models (MiniMax-M3, MiniMax-M2.7, MiniMax-M2.7-highspeed), this
        sets the correct base URL and API key. Pass either the bare model ID or a
        prefixed form such as ``minimax/MiniMax-M3`` or ``openai/MiniMax-M3``.

        All other models are forwarded to ``dspy.LM`` unchanged, so existing
        OpenAI / OpenRouter / LiteLLM strings continue to work.
        """
        import dspy

        # Strip any provider prefix (e.g. "openai/", "minimax/") to get the bare ID
        bare = model.split("/")[-1]

        if bare in MINIMAX_MODELS:
            if not self.minimax_api_key:
                raise ValueError(
                    f"MINIMAX_API_KEY is not set. Export it before using '{bare}'."
                )
            return dspy.LM(
                f"openai/{bare}",
                api_key=self.minimax_api_key,
                base_url=self.minimax_base_url,
                temperature=1.0,  # MiniMax requires temperature in (0.0, 1.0]
            )

        return dspy.LM(model)


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
