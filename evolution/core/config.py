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
    "MiniMax-M2.7",            # Peak Performance. Ultimate Value.
    "MiniMax-M2.7-highspeed",  # Same performance, faster.
)

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


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

        For MiniMax models (MiniMax-M2.7 or MiniMax-M2.7-highspeed), this sets
        the correct base URL and API key. Pass either the bare model ID or a
        prefixed form such as ``minimax/MiniMax-M2.7`` or ``openai/MiniMax-M2.7``.

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
