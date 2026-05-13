"""Tool-flavored LLM judge + fitness metric.

ToolJudgeSignature mirrors the 3-dim output shape of JudgeSignature but
its inputs are (task, expected_tool, chosen_tool, agent_reasoning). The metric
parses the agent's chosen_tool name (with normalization) before reaching
the judge — unparseable outputs and nonexistent tool choices short-circuit
to score 0.0 with diagnostic feedback.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import (
    FitnessScore,
    _PROFILE_WEIGHTS,
    _augment_feedback_with_closed_loop,
    _augment_feedback_with_pred_trace,
    _clamp_to_unit,
)
from evolution.tools.tool_source import ToolManifest

logger = logging.getLogger(__name__)


class ToolJudgeSignature(dspy.Signature):
    """Judge the quality of a tool-selection decision on three dimensions.

    Note: the agent-reasoning input field is named ``agent_reasoning`` rather
    than ``reasoning`` because ``dspy.ChainOfThought`` (which wraps this
    signature in ``ToolJudge``) prepends its own ``reasoning`` *output*
    field. A user input named ``reasoning`` would be shadowed and silently
    dropped by ``dspy.Predict``'s kwarg validation.
    """
    task: str = dspy.InputField(desc="The user task")
    expected_tool: str = dspy.InputField(desc="The correct tool name")
    chosen_tool: str = dspy.InputField(desc="The tool the agent picked")
    agent_reasoning: str = dspy.InputField(desc="The agent's stated reasoning for its tool choice")

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


class ToolJudge:
    """LLM-as-judge scorer for tool-selection outputs.

    Mirrors the skill-shaped judge's contract but takes the four
    tool-selection input fields (``task``, ``expected_tool``,
    ``chosen_tool``, ``agent_reasoning``) instead of the three
    skill-shaped fields. Returns a ``FitnessScore`` with
    ``length_penalty=0.0`` — length pressure on the tool path lives in
    the proposer's budget-aware slope, not the judge.
    """

    def __init__(self, config: EvolutionConfig):
        if config.fitness_profile not in _PROFILE_WEIGHTS:
            raise ValueError(
                f"Unknown fitness_profile {config.fitness_profile!r}; "
                f"expected one of {sorted(_PROFILE_WEIGHTS)}"
            )
        self.config = config
        self.profile = config.fitness_profile
        self.judge = dspy.ChainOfThought(ToolJudgeSignature)

    def score(
        self,
        task: str,
        expected_tool: str,
        chosen_tool: str,
        agent_reasoning: str,
    ) -> FitnessScore:
        """Score a tool-selection decision using LLM-as-judge."""

        lm = dspy.LM(
            self.config.eval_model,
            temperature=0.0,
            max_tokens=4000,
            request_timeout=60,
            num_retries=5,
        )

        with dspy.context(lm=lm):
            result = self.judge(
                task=task,
                expected_tool=expected_tool,
                chosen_tool=chosen_tool,
                agent_reasoning=agent_reasoning,
            )

        return FitnessScore(
            correctness=_clamp_to_unit(result.correctness),
            procedure_following=_clamp_to_unit(result.procedure_following),
            conciseness=_clamp_to_unit(result.conciseness),
            length_penalty=0.0,
            feedback=str(result.feedback),
            profile=self.profile,
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
    closed_loop_cache: Optional[Any] = None,
) -> Callable:
    """Construct a GEPA-shaped 5-arg fitness metric.

    The returned callable runs the agent's prediction, parses chosen_tool,
    and feeds (task, expected_tool, chosen_tool, agent_reasoning) to the judge.
    Unparseable outputs and nonexistent-tool choices short-circuit before
    reaching the judge.

    ``text_extractor`` is the predictor-text extractor passed to
    ``_augment_feedback_with_pred_trace`` so the [BUDGET] reflection line
    measures the description region between sentinels rather than the
    full rendered manifest. Without it the budget framing is wrong by an
    order of magnitude on multi-tool manifests.

    ``closed_loop_cache`` is an optional ``ClosedLoopFeedbackCache``
    (typed as ``Any`` to keep imports light). When set, judge scores are
    recorded on every call and a ``[CLOSED_LOOP]`` block is appended to
    the feedback string in the reflective-feedback path.
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
            task=gold.task_input,
            expected_tool=gold.expected_behavior,
            chosen_tool=chosen,
            agent_reasoning=reasoning,
        )
        if closed_loop_cache is not None:
            closed_loop_cache.record_judge_score(score.composite)
        feedback = _augment_feedback_with_pred_trace(
            score.feedback,
            pred_trace,
            baseline_len=baseline_len,
            target_len=target_len,
            text_extractor=text_extractor,
        )
        feedback = _augment_feedback_with_closed_loop(
            feedback,
            closed_loop_cache,
            pred_trace,
            text_extractor=text_extractor,
        )
        return dspy.Prediction(score=score.composite, feedback=feedback)

    return metric
