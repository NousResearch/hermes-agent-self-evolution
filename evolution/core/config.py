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
    # Routed to OpenRouter (session's model access) via OPENAI_API_KEY/OPENAI_API_BASE.
    optimizer_model: str = "tencent/hy3:free"  # Model for GEPA reflections
    eval_model: str = "tencent/hy3:free"  # Model for LLM-as-judge scoring
    judge_model: str = "tencent/hy3:free"  # Model for dataset generation

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


def _install_litellm_shim() -> None:
    """Route vendor models that lack structured-output support through a proxy.

    DSPy's adapter forces a JSON-schema ``response_format``. Some OpenRouter
    models (e.g. ``tencent/hy3:free``) reject that. We intercept litellm's
    completion call, drop ``response_format``, call the proxy via the openai
    client, and return a litellm-compatible ``ModelResponse`` built from plain
    text. Idempotent — safe to call repeatedly.
    """
    import litellm
    from openai import OpenAI
    from litellm import ModelResponse, Choices, MessageData, Usage

    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_base or not api_key:
        return

    client = OpenAI(api_key=api_key, base_url=api_base)
    _orig = litellm.completion

    def _patched(model=None, messages=None, **kwargs):
        if model and "hy3" in str(model):
            kwargs.pop("response_format", None)
            kwargs.pop("extra_body", None)
            cleaned = {
                k: v for k, v in kwargs.items()
                if k in ("temperature", "max_tokens", "top_p", "stop",
                         "tools", "tool_choice", "n", "seed")
            }
            oa = client.chat.completions.create(
                model="tencent/hy3:free",
                messages=messages or [{"role": "user", "content": ""}],
                **cleaned,
            )
            c = oa.choices[0]
            return ModelResponse(
                model=oa.model,
                choices=[Choices(
                    message=MessageData(content=c.message.content or "", role="assistant"),
                    finish_reason=(c.finish_reason or "stop"),
                    index=0,
                )],
                usage=Usage(
                    prompt_tokens=oa.usage.prompt_tokens,
                    completion_tokens=oa.usage.completion_tokens,
                    total_tokens=oa.usage.total_tokens,
                ) if oa.usage else None,
            )
        return _orig(model=model, messages=messages, **kwargs)

    litellm.completion = _patched
    # DSPy imports `completion` by name; patch that binding too.
    try:
        import dspy.clients.lm as _dspy_lm
        _dspy_lm.completion = _patched
    except Exception:
        pass


def make_lm(model: str) -> "dspy.LM":  # noqa: F821
    """Build a DSPy LM that honors OPENAI_API_BASE / OPENAI_API_KEY.

    When an OpenAI-compatible proxy (OpenRouter, Nous gateway, etc.) is
    configured via ``OPENAI_API_BASE``, pass api_base + api_key explicitly so
    dspy/litellm route calls to that endpoint. A shim is installed so models
    that reject JSON-schema structured output (e.g. ``tencent/hy3:free``) still
    work: the adapter's response_format is dropped and plain-text JSON is
    returned in litellm's response shape.
    """
    import dspy

    _install_litellm_shim()

    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    kwargs = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return dspy.LM(model, **kwargs)


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
