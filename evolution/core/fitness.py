"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import dspy
from dataclasses import dataclass
from typing import Optional, Union

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
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved.")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self._lm: Optional[dspy.LM] = None

    def _get_lm(self) -> dspy.LM:
        if self._lm is None:
            self._lm = dspy.LM(self.config.eval_model)
        return self._lm

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
        try:
            lm = self._get_lm()
            with dspy.context(lm=lm):
                judge = dspy.ChainOfThought(self.JudgeSignature)
                result = judge(
                    task_input=task_input,
                    expected_behavior=expected_behavior,
                    agent_output=agent_output,
                    skill_text=skill_text,
                )

            # Parse scores (clamp to 0-1)
            correctness = _parse_score(result.correctness)
            procedure_following = _parse_score(result.procedure_following)
            conciseness = _parse_score(result.conciseness)
            feedback = str(result.feedback) if result.feedback else ""

        except Exception as e:
            # Fall back to keyword overlap when LLM call fails (offline/rate-limited)
            correctness, procedure_following, conciseness, feedback = _keyword_fallback(
                agent_output, expected_behavior, task_input, e
            )

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
            feedback=feedback,
        )


def _keyword_fallback(agent_output: str, expected_behavior: str, task: str, error: Exception) -> tuple:
    """Keyword-overlap fallback when LLM judge is unavailable."""
    if not agent_output.strip():
        return 0.0, 0.0, 0.0, "Empty output"

    expected_words = set(expected_behavior.lower().split())
    output_words = set(agent_output.lower().split())

    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
    else:
        overlap = 0.5

    # Broader score band: 0.2-0.9 instead of 0.3-1.0
    base_score = 0.2 + (0.7 * overlap)
    score = min(1.0, max(0.0, base_score))

    feedback = (
        f"LLM judge unavailable ({error}), used keyword fallback. "
        f"Keyword overlap: {overlap:.1%}. "
        f"Expected keywords covered: {len(expected_words & output_words)}/{len(expected_words)}."
    )
    return score, score, score, feedback


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> Union[float, dspy.Prediction]:
    """DSPy-compatible metric function for skill optimization.

    When trace is provided (GEPA path): returns dspy.Prediction(score, feedback)
    for reflective mutation. When trace is None (MIPROv2 / BootstrapFewShot path):
    returns a plain float for numerical optimization.

    Uses LLM-as-judge as the primary scorer. Falls back to keyword overlap
    on LLM failures (offline, rate-limited, etc.).
    """
    # The prediction should have an 'output' field with the agent's response
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""
    skill_text = getattr(example, "skill_text", "") or ""

    if not agent_output.strip():
        if trace is not None:
            return dspy.Prediction(score=0.0, feedback="Empty agent output")
        return 0.0

    try:
        config = EvolutionConfig()
        judge = LLMJudge(config)
        result = judge.score(
            task_input=task,
            expected_behavior=expected,
            agent_output=agent_output,
            skill_text=skill_text,
        )
        score = result.composite
        feedback = result.feedback
    except Exception as e:
        # Keyword fallback: broader signal than the old 0.3-1.0 band
        score, _, _, feedback = _keyword_fallback(agent_output, expected, task, e)

    if trace is not None:
        # GEPA path: return Prediction for reflective mutation
        return dspy.Prediction(score=score, feedback=feedback)

    # MIPROv2 / BootstrapFewShot path: return plain float
    return float(score)


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
