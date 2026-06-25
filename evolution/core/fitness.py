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

        lm = dspy.LM(self.config.eval_model)

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


def skill_fitness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None, pred_name: str = "", pred_trace=None):
    """DSPy-compatible metric function for skill optimization.

    GEPA requires 5 args: (gold, pred, trace, pred_name, pred_trace).
    MIPROv2 and other optimizers call with 3 (example, prediction, trace).

    Returns (score, feedback_dict) for GEPA's reflective analysis.
    The feedback dict has key 'feedback' with natural-language critique.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        return 0.0

    # Use LLM-as-judge for real quality scoring.
    # GEPA's reflection_lm generates its own feedback from trajectories —
    # the metric only needs to return an accurate float score.
    try:
        judge = dspy.ChainOfThought(_JudgeSignature)
        lm = _get_judge_lm()  # Reuse single LM instance (avoids fd leak)
        with dspy.context(lm=lm):
            result = judge(
                task_input=task,
                expected_behavior=expected,
                agent_output=agent_output[:4000],  # Truncate to keep token cost manageable
            )

        correctness = _parse_score(result.correctness)
        procedure = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Weighted composite (same weights as FitnessScore)
        score = 0.5 * correctness + 0.3 * procedure + 0.2 * conciseness

        return min(1.0, max(0.0, score))
    except Exception:
        # Fallback: simple heuristic if judge fails
        expected_words = set(expected.lower().split())
        output_words = set(agent_output.lower().split())
        overlap = len(expected_words & output_words) / max(1, len(expected_words))
        return min(1.0, max(0.0, 0.3 + 0.7 * overlap))


class _JudgeSignature(dspy.Signature):
    """Evaluate a Slack draft against the expected behavior rubric.

    Score three dimensions (0.0 to 1.0 each):
    1. correctness: Does the draft address what was asked?
    2. procedure_following: Does it follow the Slack style rules (direct, first-person, no hyphens, proper @mentions, appropriate length)?
    3. conciseness: Is it appropriately concise without padding?

    Provide specific, actionable feedback on what to improve.
    """
    task_input: str = dspy.InputField(desc="The task the user gave")
    expected_behavior: str = dspy.InputField(desc="Rubric describing what a good draft looks like")
    agent_output: str = dspy.InputField(desc="The agent's draft")
    correctness: float = dspy.OutputField(desc="Score 0.0-1.0")
    procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0")
    conciseness: float = dspy.OutputField(desc="Score 0.0-1.0")
    feedback: str = dspy.OutputField(desc="Specific, actionable feedback")


# Module-level config for the metric model — set by evolve_skill before optimization
_METRIC_MODEL = None
_JUDGE_LM = None  # Cached LM instance to avoid fd leak

def _get_metric_model() -> str:
    return _METRIC_MODEL or "openrouter/anthropic/claude-opus-4.8"

def _get_judge_lm():
    """Return a cached LM instance to avoid creating new connections per call."""
    global _JUDGE_LM
    if _JUDGE_LM is None:
        _JUDGE_LM = dspy.LM(_get_metric_model(), temperature=0.0, max_tokens=4096)
    return _JUDGE_LM

def set_metric_model(model: str):
    """Set the model used for LLM-as-judge scoring during optimization."""
    global _METRIC_MODEL, _JUDGE_LM
    _METRIC_MODEL = model
    _JUDGE_LM = None  # Reset cache so next call creates a new LM with updated model


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
