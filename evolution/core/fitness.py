"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import dspy

from evolution.core.config import EvolutionConfig

logger = logging.getLogger(__name__)


_PROFILE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "balanced": (0.5, 0.3, 0.2),
    "compression": (0.4, 0.2, 0.4),
    "growth": (0.6, 0.4, 0.0),
}


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score. All sub-scores are in [0, 1]."""
    correctness: float = 0.0
    procedure_following: float = 0.0
    conciseness: float = 0.0
    length_penalty: float = 0.0
    feedback: str = ""
    profile: str = "balanced"

    @property
    def composite(self) -> float:
        """Weighted composite. Length penalty is subtracted, then floored at 0."""
        w_correctness, w_procedure, w_conciseness = _PROFILE_WEIGHTS[self.profile]
        raw = (
            w_correctness * self.correctness
            + w_procedure * self.procedure_following
            + w_conciseness * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        # Typed `str` (not `float`) because LLMs sometimes emit explanatory
        # text or confidence ranges; _clamp_to_unit parses + clamps in score().
        correctness: str = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: str = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: str = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        if config.fitness_profile not in _PROFILE_WEIGHTS:
            raise ValueError(
                f"Unknown fitness_profile {config.fitness_profile!r}; "
                f"expected one of {sorted(_PROFILE_WEIGHTS)}"
            )
        self.config = config
        self.profile = config.fitness_profile
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        # request_timeout=60 ≈ 6x P99 of the slowest observed gpt-4.1-mini
        # call. TimeoutError after retries propagates to GEPA's fallback path.
        lm = dspy.LM(self.config.eval_model, temperature=0.0, max_tokens=4000, request_timeout=60, num_retries=5)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
            )

        correctness = _clamp_to_unit(result.correctness)
        procedure_following = _clamp_to_unit(result.procedure_following)
        conciseness = _clamp_to_unit(result.conciseness)

        # Penalty ramps from 0 at 90% of max_size to 0.3 at 100%+.
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
            profile=self.profile,
        )


def make_skill_fitness_metric(
    judge: "LLMJudge",
    baseline_skill_text: str = "",
    max_growth: float = 0.2,
    closed_loop_cache: Optional[Any] = None,
):
    """Build a GEPA-compatible metric closed over an LLMJudge instance.

    Returns a callable matching GEPA's 5-arg protocol:
    ``(gold, pred, trace=None, pred_name=None, pred_trace=None) ->
    dspy.Prediction(score: float, feedback: str)``.

    Returning ``Prediction(score, feedback)`` lets GEPA's reflective loop
    consume the judge's natural-language critique directly instead of falling
    back to its canned "trajectory got a score of {x}" template
    (``dspy/teleprompt/gepa/gepa.py:537``).

    Score must be byte-identical between predictor-level and module-level
    call sites — GEPA warns and overrides on divergence — so the
    pred_trace-driven feedback enrichment never touches the score.

    ``closed_loop_cache`` is an optional ``ClosedLoopFeedbackCache`` (typed
    as ``Any`` here to keep this module free of the validation-stack
    import). When set, the metric records each judge score on the cache
    (for the saturation gate) and appends the cache's rendered
    ``[CLOSED_LOOP]`` block to the feedback string in the reflective-feedback
    path (``pred_trace`` is set). Never modifies the score.
    """
    baseline_len = len(baseline_skill_text or "")
    target_len = int(baseline_len * (1 + max_growth)) if baseline_len else 0

    def skill_fitness_metric(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
    ):
        if hasattr(prediction, "_closed_loop_task_id"):
            return _score_behavioral_example(prediction, closed_loop_cache)

        agent_output = getattr(prediction, "output", "") or ""
        task = getattr(example, "task_input", "") or ""

        if not agent_output.strip():
            # Surfaces real upstream failures (timeout, content filter,
            # malformed prompt) that GEPA otherwise can't distinguish from
            # a wrong answer.
            logger.warning(
                "skill_fitness_metric: empty agent output for task=%r",
                task[:80],
            )
            return dspy.Prediction(
                score=0.0,
                feedback="Agent produced empty output (no characters returned).",
            )

        score = judge.score(
            task_input=task,
            expected_behavior=getattr(example, "expected_behavior", "") or "",
            agent_output=agent_output,
        )
        if closed_loop_cache is not None:
            closed_loop_cache.record_judge_score(score.composite)
        feedback = _augment_feedback_with_pred_trace(
            score.feedback,
            pred_trace,
            baseline_len=baseline_len,
            target_len=target_len,
        )
        feedback = _augment_feedback_with_closed_loop(
            feedback,
            closed_loop_cache,
            pred_trace,
            text_extractor=None,
        )
        return dspy.Prediction(score=score.composite, feedback=feedback)

    return skill_fitness_metric


def _augment_feedback_with_pred_trace(
    base_feedback: str,
    pred_trace,
    baseline_len: int,
    target_len: int,
    text_extractor: Optional[Callable[[Any], str]] = None,
) -> str:
    """Append [BUDGET] + [REASONING] context when pred_trace is set.

    GEPA passes ``pred_trace=None`` from its Evaluate-over-valset path and
    a one-tuple ``[(predictor, inputs, output)]`` from its reflective
    feedback path. The feedback is enriched only in the latter case.

    ``text_extractor`` is an optional ``predictor -> str`` callable used to
    pick the budget-relevant text. Defaults to reading
    ``predictor.signature.instructions``. Pass a custom extractor when the
    budget-relevant text is a sub-field of the predictor (e.g. a tool
    description) rather than the full instruction string.
    """
    if not pred_trace:
        return base_feedback

    try:
        predictor, _inputs, output = pred_trace[0]
    except (IndexError, TypeError, ValueError):
        return base_feedback

    extras: list[str] = []

    if baseline_len > 0:
        try:
            if text_extractor is not None:
                current_text = text_extractor(predictor) or ""
            else:
                current_text = predictor.signature.instructions or ""
        except AttributeError:
            current_text = ""
        if current_text:
            current_len = len(current_text)
            growth_pct = (current_len - baseline_len) / baseline_len * 100
            extras.append(
                f"[BUDGET] Your current instruction is {current_len} chars vs "
                f"baseline {baseline_len} chars ({growth_pct:+.1f}%); "
                f"target ≤{target_len} chars. Prefer a tighter instruction "
                f"that preserves the essential procedure."
            )

    reasoning = getattr(output, "reasoning", "") or getattr(output, "rationale", "")
    if reasoning:
        snippet = str(reasoning).strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        extras.append(f"[REASONING] Agent's chain-of-thought: {snippet}")

    if not extras:
        return base_feedback
    return base_feedback + "\n\n" + "\n\n".join(extras)


def _score_behavioral_example(
    prediction: dspy.Prediction,
    closed_loop_cache: Optional[Any],
) -> dspy.Prediction:
    """Score a behavioral example via the closed-loop cache.

    Reads ``_closed_loop_task_id`` and ``_candidate_text`` from the
    prediction (stuffed there by ``ToolModule.forward`` on the behavioral
    branch — see ``evolution/tools/tool_module.py``). Returns a
    ``Prediction(score, feedback)`` with binary score:

      - 1.0 if the cached ``TaskResult.passed`` is True and not abstained
      - 0.0 on fail, abstain, cache miss, or missing cache

    Score is deterministic over candidate text (cache is keyed by it), so
    repeated calls with the same prediction return byte-identical scores —
    GEPA's predictor-vs-module byte-identity contract holds automatically.
    """
    task_id = prediction._closed_loop_task_id
    candidate_text = getattr(prediction, "_candidate_text", "") or ""

    if closed_loop_cache is None or not candidate_text:
        return dspy.Prediction(
            score=0.0,
            feedback=f"[BEHAVIORAL] task {task_id}: cache unavailable",
        )

    verdict = closed_loop_cache.get_task_verdict(candidate_text, task_id)

    if verdict is None:
        return dspy.Prediction(
            score=0.0,
            feedback=f"[BEHAVIORAL] task {task_id}: no verdict (cache miss or validator error)",
        )

    if verdict.abstained:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"[BEHAVIORAL] task {task_id}: abstained "
                f"(runner error: {verdict.error or 'unknown'})"
            ),
        )

    outcome = "pass" if verdict.passed else "fail"
    score = 1.0 if verdict.passed else 0.0
    return dspy.Prediction(
        score=score,
        feedback=(
            f"[BEHAVIORAL] task {task_id}: {outcome}\n"
            f"  tools_invoked: {list(verdict.tool_calls_seq)!r}"
        ),
    )


def _augment_feedback_with_closed_loop(
    base_feedback: str,
    closed_loop_cache: Optional[Any],
    pred_trace,
    text_extractor: Optional[Callable[[Any], str]] = None,
) -> str:
    """Append a [CLOSED_LOOP] block when the cache returns a verdict.

    Gated identically to ``_augment_feedback_with_pred_trace``: only fires
    when ``pred_trace`` is set (GEPA's reflective-feedback path), so the
    Pareto-evaluation path stays cheap.

    Closed-loop verdict retrieval is best-effort: cache miss with the gate
    closed returns ``None``, validator errors are swallowed inside the
    cache and return ``None``. In either case base_feedback passes through
    unchanged.
    """
    if closed_loop_cache is None or not pred_trace:
        return base_feedback

    try:
        predictor, _inputs, _output = pred_trace[0]
    except (IndexError, TypeError, ValueError):
        return base_feedback

    try:
        if text_extractor is not None:
            candidate_text = text_extractor(predictor) or ""
        else:
            candidate_text = predictor.signature.instructions or ""
    except AttributeError:
        return base_feedback

    if not candidate_text:
        return base_feedback

    report = closed_loop_cache.get_or_run(candidate_text)
    if report is None:
        return base_feedback

    # Local import to avoid pulling the validation stack into every metric
    # build site; the cache parameter is what gates this import.
    from evolution.core.closed_loop_feedback import render_feedback_block

    block = render_feedback_block(report)
    return base_feedback + "\n\n" + block


def _clamp_to_unit(value: str) -> float:
    """Parse an LLM-emitted score string and clamp it to [0, 1].

    Returns the neutral 0.5 on malformed input rather than raising — a single
    noisy judge call shouldn't crash a whole optimization run.
    """
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5
