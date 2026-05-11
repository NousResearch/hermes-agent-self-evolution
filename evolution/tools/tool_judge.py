"""Tool-flavored LLM judge + fitness metric.

ToolJudgeSignature mirrors the 3-dim output shape of JudgeSignature but
its inputs are (task, expected_tool, chosen_tool, reasoning). The metric
parses the agent's chosen_tool name (with normalization) before reaching
the judge — unparseable outputs and nonexistent tool choices short-circuit
to score 0.0 with diagnostic feedback.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import dspy

from evolution.core.fitness import _augment_feedback_with_pred_trace
from evolution.tools.tool_source import ToolManifest

logger = logging.getLogger(__name__)


class ToolJudgeSignature(dspy.Signature):
    """Judge the quality of a tool-selection decision on three dimensions."""
    task: str = dspy.InputField(desc="The user task")
    expected_tool: str = dspy.InputField(desc="The correct tool name")
    chosen_tool: str = dspy.InputField(desc="The tool the agent picked")
    reasoning: str = dspy.InputField(desc="The agent's stated reasoning")

    correctness: str = dspy.OutputField(
        desc="0.0-1.0: was the chosen tool the correct or a defensibly-equivalent choice?"
    )
    procedure_following: str = dspy.OutputField(
        desc="0.0-1.0: did the agent reason from the manifest descriptions appropriately?"
    )
    conciseness: str = dspy.OutputField(
        desc="0.0-1.0: was the reasoning crisp and unbloated?"
    )
    feedback: str = dspy.OutputField(
        desc="One-sentence diagnosis if any dimension is below 1.0; empty otherwise"
    )


def _normalize_tool_name_for_match(text: str) -> str:
    """Generous normalization for tool-name matching: lowercase, strip
    quotes/backticks/whitespace, replace hyphens with underscores.
    """
    s = text.strip()
    s = s.strip("\"'`")
    s = s.lower()
    s = s.replace("-", "_")
    return s


def _parse_chosen_tool(raw_output: str, manifest: ToolManifest) -> str:
    """Parse the agent's chosen_tool output into a manifest tool name.

    Returns the matched manifest tool name (with the manifest's original
    casing) when the input normalizes to a known tool. Returns the
    normalized form when the input looks like a single identifier-shaped
    token (so the caller can distinguish "agent named a nonexistent tool"
    from "agent emitted free-form prose"). Returns "" only when the input
    is blank or clearly not a tool name (whitespace inside the token).
    """
    if not raw_output or not raw_output.strip():
        return ""
    normalized = _normalize_tool_name_for_match(raw_output)
    manifest_names = {_normalize_tool_name_for_match(t.name): t.name for t in manifest.tools}
    if normalized in manifest_names:
        return manifest_names[normalized]
    # Free-form prose (contains whitespace after normalization) is unparseable.
    if not normalized or any(ch.isspace() for ch in normalized):
        return ""
    return normalized


def make_tool_fitness_metric(
    judge,
    baseline_description: str,
    manifest: ToolManifest,
    target_tool_name: str,
    max_growth: float,
    text_extractor: Optional[Callable[[Any], str]] = None,
) -> Callable:
    """Construct a GEPA-shaped 5-arg fitness metric.

    The returned callable runs the agent's prediction, parses chosen_tool,
    and feeds (task, expected_tool, chosen_tool, reasoning) to the judge.
    Unparseable outputs and nonexistent-tool choices short-circuit before
    reaching the judge.

    ``text_extractor`` is the predictor-text extractor passed to
    ``_augment_feedback_with_pred_trace`` so the [BUDGET] reflection line
    measures the description region between sentinels rather than the
    full rendered manifest. Without it the budget framing is wrong by an
    order of magnitude on multi-tool manifests.
    """
    available_names = sorted(t.name for t in manifest.tools)
    baseline_len = len(baseline_description or "")
    target_len = int(baseline_len * (1 + max_growth)) if baseline_len else 0

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        raw_chosen = getattr(pred, "chosen_tool", "") or ""
        reasoning = getattr(pred, "reasoning", "") or ""

        chosen = _parse_chosen_tool(raw_chosen, manifest)
        if not chosen:
            logger.warning(
                "make_tool_fitness_metric: unparseable chosen_tool output %r for task %r",
                raw_chosen, gold.task_input,
            )
            return dspy.Prediction(
                score=0.0,
                feedback="Agent did not produce a parseable tool selection.",
            )

        if chosen not in {t.name for t in manifest.tools}:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"Agent chose nonexistent tool {chosen!r}; "
                    f"available tools are: {available_names}"
                ),
            )

        score = judge.score(
            task_input=gold.task_input,
            expected_behavior=gold.expected_behavior,
            agent_output=f"chosen_tool: {chosen}\nreasoning: {reasoning}",
            artifact_size=None,
            max_size=None,
        )
        feedback = _augment_feedback_with_pred_trace(
            score.feedback,
            pred_trace,
            baseline_len=baseline_len,
            target_len=target_len,
            text_extractor=text_extractor,
        )
        return dspy.Prediction(score=score.composite, feedback=feedback)

    return metric
