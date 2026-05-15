"""Pre-run cost suggester for the eval LM.

The framework defaults all four LM roles (optimizer, reflection, eval, judge)
to whatever Hermes resolved as ``model.default``. On a Hermes config pinned
to a frontier model like ``claude-opus-4-5``, the eval + judge roles fire
~100 calls per evolution, often costing 5-20x more than necessary because
a same-provider haiku/mini model would suffice for evaluation.

This module surfaces a Rich panel after preflight, before any expensive
work, suggesting a cheaper same-provider model. Optimizer + reflection
roles are intentionally untouched — those benefit from reasoning quality,
and a cheaper-model swap there can silently degrade evolution outcomes.

Pricing comes from ``litellm.model_cost`` (a curated dict shipped with
LiteLLM). Bedrock, Codex, and local-server endpoints aren't in the
catalog and produce no suggestion — the advisor gracefully returns
``None`` rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import litellm
from rich.panel import Panel


# Minimum input-cost ratio for a suggestion to surface. 1.5× is the
# threshold below which the panel becomes noise — e.g., suggesting a
# date-suffixed older Bedrock snapshot ($3.00/M) over the canonical newer
# version ($3.30/M) for 1.1× savings is not worth interrupting the user
# for. 1.7× sonnet-vs-opus and 5× haiku-vs-opus stay above this floor.
_MIN_INPUT_COST_RATIO = 1.5


@dataclass(frozen=True)
class CheaperAlternative:
    """A same-provider model that's strictly cheaper for input tokens with
    at least the same context window as the current choice.
    """

    current_model: str
    current_input_cost_per_1m: float
    current_output_cost_per_1m: float
    current_max_input_tokens: int
    suggested_model: str
    suggested_input_cost_per_1m: float
    suggested_output_cost_per_1m: float
    suggested_max_input_tokens: int
    input_cost_ratio: float
    output_cost_ratio: float
    provider: str


def find_cheaper_alternative(model: str) -> Optional[CheaperAlternative]:
    """Return the cheapest same-provider model with >= context window, or None.

    The lookup tolerates two model-string shapes:
      * direct keys (``openrouter/openai/gpt-5``) that LiteLLM stores verbatim
      * provider-prefixed keys (``anthropic/claude-opus-4-5``) where LiteLLM
        only carries the bare name (``claude-opus-4-5``); strip and retry

    Returns None when the model is unknown to LiteLLM (Bedrock, Codex,
    local-server endpoints) or when no same-provider alternative is both
    cheaper for input tokens and at least as wide on context. Callers
    interpret None as "no advice to give" — silent skip, no error.
    """
    catalog: Dict[str, Dict[str, Any]] = litellm.model_cost
    current_key, current_entry = _lookup(model, catalog)
    if current_entry is None:
        return None

    current_input = current_entry.get("input_cost_per_token")
    current_output = current_entry.get("output_cost_per_token")
    current_ctx = current_entry.get("max_input_tokens") or current_entry.get(
        "max_tokens"
    )
    provider = current_entry.get("litellm_provider")
    if not (current_input and current_output and current_ctx and provider):
        return None

    current_major, _ = _version_tuple(current_key)
    current_namespace = _namespace(current_key)

    # Enumerate same-provider candidates that are strictly cheaper on input
    # cost, have at least the current context window, share the current
    # model's major version, AND share its namespace path. The version
    # filter blocks gen-3 downgrades when the user is on gen-4. The
    # namespace filter blocks cross-routing surprises:
    #   * Bedrock cross-region (us.anthropic.X) vs regional (anthropic.X) —
    #     the user picked us.* deliberately for failover/throughput; the
    #     regional model is a different routing profile, not a substitute.
    #   * OpenRouter cross-vendor (openrouter/anthropic/claude-X vs
    #     openrouter/z-ai/glm-X) — same litellm_provider, completely
    #     different upstream model.
    candidates = []
    for cand_key, cand_entry in catalog.items():
        if cand_entry.get("litellm_provider") != provider:
            continue
        cand_input = cand_entry.get("input_cost_per_token")
        cand_output = cand_entry.get("output_cost_per_token")
        cand_ctx = cand_entry.get("max_input_tokens") or cand_entry.get("max_tokens")
        if not (cand_input and cand_output and cand_ctx):
            continue
        if cand_input >= current_input:
            continue
        if cand_ctx < current_ctx:
            continue
        cand_major, cand_minor = _version_tuple(cand_key)
        if current_major is not None and cand_major != current_major:
            continue
        if _namespace(cand_key) != current_namespace:
            continue
        candidates.append((cand_input, -cand_minor, cand_key, cand_entry))

    if not candidates:
        return None

    # Sort by (input cost asc, minor version desc, name asc). Cost wins
    # primarily; ties on cost prefer the newer minor (so a tied
    # claude-sonnet-4-6 beats claude-4-sonnet-20250514, both at $3/M);
    # ties on minor prefer the shorter/canonical name.
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    _, _, suggested_key, suggested_entry = candidates[0]

    # Suppress weak suggestions. Below ~1.5× the panel becomes noise the
    # user has to read past every run; above it, the savings are worth
    # surfacing.
    if current_input / suggested_entry["input_cost_per_token"] < _MIN_INPUT_COST_RATIO:
        return None

    # Reconstruct a paste-ready model string. If the user passed a
    # provider-prefixed name (e.g., ``anthropic/claude-opus-4-5``) but
    # the catalog stores it bare, the suggestion needs the prefix back so
    # ``--eval-model`` flows through the resolver correctly.
    suggested_model = _with_provider_prefix(
        suggested_key, original_model=model, original_key=current_key
    )

    return CheaperAlternative(
        current_model=model,
        current_input_cost_per_1m=current_input * 1_000_000,
        current_output_cost_per_1m=current_output * 1_000_000,
        current_max_input_tokens=current_ctx,
        suggested_model=suggested_model,
        suggested_input_cost_per_1m=suggested_entry["input_cost_per_token"]
        * 1_000_000,
        suggested_output_cost_per_1m=suggested_entry["output_cost_per_token"]
        * 1_000_000,
        suggested_max_input_tokens=suggested_entry.get("max_input_tokens")
        or suggested_entry.get("max_tokens"),
        input_cost_ratio=current_input / suggested_entry["input_cost_per_token"],
        output_cost_ratio=current_output
        / suggested_entry["output_cost_per_token"],
        provider=provider,
    )


def render_suggestion_panel(role: str, alt: CheaperAlternative) -> Panel:
    """Build the Rich panel surfaced after preflight."""
    in_ratio = alt.input_cost_ratio
    out_ratio = alt.output_cost_ratio
    body = (
        f"Your [bold]{role}[/bold] LM is [bold]{alt.current_model}[/bold] "
        f"(${alt.current_input_cost_per_1m:.2f}/M input, "
        f"${alt.current_output_cost_per_1m:.2f}/M output).\n\n"
        f"On [bold]{alt.provider}[/bold], "
        f"[bold cyan]{alt.suggested_model}[/bold cyan] "
        f"(${alt.suggested_input_cost_per_1m:.2f}/M input, "
        f"${alt.suggested_output_cost_per_1m:.2f}/M output) is "
        f"[bold]{in_ratio:.1f}× cheaper[/bold] for input "
        f"({out_ratio:.1f}× for output) with the same "
        f"{alt.suggested_max_input_tokens:,}-token context.\n\n"
        f"To apply: [green]--eval-model {alt.suggested_model}[/green]\n\n"
        f"[dim]Eval + judge roles fire ~100× per evolution; cheaper models "
        f"save the most here. Optimizer + reflection roles benefit from "
        f"reasoning quality — those are unchanged.[/dim]"
    )
    return Panel(
        body,
        title="[bold cyan]💡 Cost suggestion[/bold cyan]",
        border_style="cyan",
    )


def _namespace(catalog_key: str) -> str:
    """Extract a routing-namespace path that suggestions must match.

    For slash-segmented keys (OpenRouter, Bedrock-with-explicit-region):
    everything before the last ``/`` segment.

      ``openrouter/anthropic/claude-opus-4`` -> ``"openrouter/anthropic"``
      ``openrouter/z-ai/glm-4.7-flash``      -> ``"openrouter/z-ai"``

    For Bedrock-style dot-prefixed keys (cross-region inference profiles
    or regional-only): the leading dot-segments stripped of the trailing
    model body, where leading routing tokens look like short alphabetic
    words (us, eu, apac, anthropic, ...). Stops at the first segment that
    contains a digit (e.g. ``claude-3``), which marks the model body.

      ``us.anthropic.claude-haiku-4-5-20251001-v1:0`` -> ``"us.anthropic"``
      ``anthropic.claude-haiku-4-5-20251001-v1:0``    -> ``"anthropic"``
      ``us-gov.anthropic.claude-X``                   -> ``"us-gov.anthropic"``

    For bare keys (Anthropic-direct, OpenAI-direct): empty string —
    candidates with empty namespace are interchangeable within the same
    litellm provider.

      ``claude-opus-4-5`` -> ``""``
      ``gpt-5``           -> ``""``
      ``claude-3.5-sonnet`` -> ``""``  (claude-3 has a digit, not a route)
    """
    if "/" in catalog_key:
        return catalog_key.rsplit("/", 1)[0]

    parts = catalog_key.split(".")
    if len(parts) < 2:
        return ""
    namespace_segments = []
    for seg in parts[:-1]:
        # Routing tokens: short, alphabetic (with optional hyphens like
        # us-gov). Anything containing a digit is part of the model body
        # (e.g. claude-3, gpt-4) and breaks the namespace chain.
        if not seg or not seg.replace("-", "").isalpha():
            break
        namespace_segments.append(seg)
    return ".".join(namespace_segments)


def _version_tuple(model_key: str) -> Tuple[Optional[int], int]:
    """Extract a (major, minor) version tuple from a model name.

    Used to gate cross-generation suggestions: we don't want to recommend
    a 20-month-old gen-3 model just because it's cheaper than the user's
    gen-4 choice. Returns (None, 0) for keys with no parseable version,
    which means "unknown" — treated as same-major-as-anything for legacy
    or custom model names so the filter degrades open rather than closed.

    Patterns handled:
      ``claude-opus-4-5``         -> (4, 5)
      ``claude-opus-4-5-20251101`` -> (4, 5)  (date suffix ignored)
      ``claude-3-opus-20240229``  -> (3, 0)  (single-digit major, no minor)
      ``claude-4-sonnet-20250514`` -> (4, 0)  (digit followed by non-digit)
      ``claude-sonnet-4-6``       -> (4, 6)
      ``gpt-5``                    -> (5, 0)
      ``gpt-5-mini``               -> (5, 0)
      ``custom-local-model``      -> (None, 0)
    """
    # Look for a major-minor pair (e.g., "4-5"). Skip date-shaped patterns
    # where the second number is suspiciously large or has > 3 digits.
    for major_s, minor_s in re.findall(r"\b(\d+)-(\d+)\b", model_key):
        if len(minor_s) > 3:
            continue
        minor = int(minor_s)
        if minor > 60:  # months/days don't get above 31; allow some margin
            continue
        return int(major_s), minor

    # Fall back to single-digit major (e.g., "claude-3-opus", "gpt-5").
    m = re.search(r"\b(\d+)\b", model_key)
    if m:
        return int(m.group(1)), 0
    return None, 0


def _lookup(
    model: str, catalog: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Resolve ``model`` to (catalog_key, entry) or (None, None)."""
    if model in catalog:
        return model, catalog[model]
    if "/" in model:
        bare = model.split("/", 1)[1]
        if bare in catalog:
            return bare, catalog[bare]
    return None, None


def _with_provider_prefix(
    suggested_key: str, *, original_model: str, original_key: Optional[str]
) -> str:
    """Restore the provider prefix on the suggestion if the original had one.

    Catalog keys are inconsistent: some carry the provider prefix
    (``openrouter/openai/gpt-5``), others don't (``claude-opus-4-5`` for
    Anthropic-direct). We need the suggestion to be paste-ready into
    ``--eval-model``, which means matching the resolver's expected shape —
    which mirrors whatever the caller had.
    """
    # If catalog keys for current and suggestion are both bare and the
    # caller provided a prefix, re-apply that prefix to the suggestion.
    if "/" in original_model and original_key == original_model.split("/", 1)[1]:
        prefix = original_model.split("/", 1)[0]
        return f"{prefix}/{suggested_key}"
    return suggested_key
