"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import dspy
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig

@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
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
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        # The cap matters for reasoning models: they spend most of the budget
        # on hidden `reasoning` tokens, so a low cap truncates the reply and the
        # typed output fields below parse as empty.
        lm = dspy.LM(self.config.eval_model, max_tokens=self.config.max_tokens)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def skill_fitness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """DSPy-compatible metric function for skill optimization.

    This is what gets passed to dspy.GEPA(metric=...).
    Returns a float 0-1 score.
    """
    # The prediction should have an 'output' field with the agent's response
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        return 0.0

    # Quick heuristic scoring (for speed during optimization)
    # Full LLM-as-judge scoring is expensive — use it selectively
    score = 0.5  # Base score for non-empty output

    # Check if key phrases from expected behavior appear
    expected_lower = expected.lower()
    output_lower = agent_output.lower()

    # Simple keyword overlap as a fast proxy
    expected_words = set(expected_lower.split())
    output_words = set(output_lower.split())
    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score = 0.3 + (0.7 * overlap)

    return min(1.0, max(0.0, score))


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure


# ── GEPA compatibility (DSPy 3.x) ───────────────────────────────────────────
#
# DSPy 3.x invokes a GEPA metric with FIVE positional arguments:
#   (gold_example, prediction, trace, pred_name, pred_trace)
# and probes the callable with `inspect.signature(metric).bind(None x 5)` at
# construction time. The repo's own metrics predate that contract and take
# three, so passing them directly raises
#   TypeError: GEPA metric must accept five arguments
# The adapters below restore the expected arity without changing scoring.


def make_gepa_metric(metric_fn):
    """Bind a repo metric to a DSPy 3.x-compatible 5-arg GEPA metric."""
    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
        return metric_fn(gold, pred, trace)

    gepa_metric.__name__ = getattr(metric_fn, "__name__", "gepa_metric")
    gepa_metric.__doc__ = getattr(metric_fn, "__doc__", None)
    return gepa_metric


def make_llm_judge_metric(config: EvolutionConfig, judge_lm=None):
    """Build a GEPA metric backed by the multi-dimensional LLMJudge.

    This is the *real* fitness signal: correctness / procedure_following /
    conciseness judged by an LLM against the example's rubric, with textual
    feedback GEPA consumes for reflective mutation. The legacy
    `skill_fitness_metric` is a keyword-overlap heuristic that rewards
    surface word matches, not behaviour, and GEPA optimises against whatever
    the metric rewards — so choosing this metric is the difference between
    optimising a skill and optimising word frequency.

    A judge failure must not zero out a rollout (that starves GEPA's
    reflective step), so failures fall back to the heuristic.
    """
    judge = LLMJudge(config)
    fallback = skill_fitness_metric

    def llm_judge_metric(gold, pred, trace=None) -> float:
        task = getattr(gold, "task_input", "") or ""
        expected = getattr(gold, "expected_behavior", "") or ""
        output = getattr(pred, "output", "") or ""
        if not output.strip():
            return 0.0
        skill_text = getattr(trace, "skill_text", "") if trace is not None else ""
        if not skill_text and isinstance(pred, dspy.Prediction):
            skill_text = ""
        try:
            if judge_lm is not None:
                with dspy.context(lm=judge_lm):
                    score = judge.score(
                        task_input=task,
                        expected_behavior=expected,
                        agent_output=output,
                        skill_text=skill_text,
                    )
            else:
                score = judge.score(
                    task_input=task,
                    expected_behavior=expected,
                    agent_output=output,
                    skill_text=skill_text,
                )
            return score.composite
        except Exception:
            # Never let a provider error masquerade as a bad candidate.
            return fallback(gold, pred, trace)

    return llm_judge_metric
