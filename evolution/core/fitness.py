"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import dspy
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache

from evolution.core.config import EvolutionConfig


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    completeness: float = 0.0  # Did it include all necessary info? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.4 * self.correctness
            + 0.3 * self.procedure_following
            + 0.3 * self.completeness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA/MIPROv2 can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. completeness: Did it include all necessary details, references, examples, and edge cases?

        Completeness is critical — a response that is correct but omits important
        API references, code examples, or error handling instructions is INCOMPLETE.
        Do NOT reward brevity over thoroughness. A longer, detailed response that
        covers all necessary information is better than a terse one that skips important details.

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        completeness: float = dspy.OutputField(desc="Score 0.0-1.0: Did it include all necessary details, references, examples, and edge cases? Penalize omissions harshly.")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved or was missing")

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
        completeness = _parse_score(result.completeness)

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
            completeness=completeness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


# ── Global judge instance (initialized lazily) ─────────────────────────
_judge: Optional[LLMJudge] = None
_judge_config: Optional[EvolutionConfig] = None
_skill_text_for_metric: str = ""


def init_fitness_metric(config: EvolutionConfig, skill_text: str = "") -> None:
    """Initialize the global judge for use in the metric function.

    Must be called before skill_fitness_metric is used by the optimizer.
    """
    global _judge, _judge_config, _skill_text_for_metric
    _judge_config = config
    _judge = LLMJudge(config)
    _skill_text_for_metric = skill_text


def skill_fitness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """DSPy-compatible metric function for skill optimization.

    Uses LLM-as-judge for multi-dimensional scoring:
    - correctness (40%): Did the response address the task?
    - procedure_following (30%): Did it follow the skill's procedure?
    - completeness (30%): Did it include all necessary details/examples/references?

    Falls back to keyword overlap only if the judge is not initialized.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        return 0.0

    # ── LLM-as-judge scoring (preferred) ──────────────────────────────
    if _judge is not None:
        try:
            score_obj = _judge.score(
                task_input=task,
                expected_behavior=expected,
                agent_output=agent_output,
                skill_text=_skill_text_for_metric,
            )
            return score_obj.composite
        except Exception:
            pass  # Fall through to heuristic

    # ── Fallback: keyword overlap heuristic ───────────────────────────
    # This is intentionally less weighted toward brevity — we penalize
    # outputs that are much shorter than expected
    expected_lower = expected.lower()
    output_lower = agent_output.lower()

    expected_words = set(expected_lower.split())
    output_words = set(output_lower.split())

    if not expected_words:
        return 0.5

    overlap = len(expected_words & output_words) / len(expected_words)

    # Coverage penalty: if the output is much shorter than expected,
    # it probably skipped important content
    coverage = min(1.0, len(output_words) / max(1, len(expected_words)))
    if coverage < 0.5:
        # Output is less than half the length of expected — likely missing content
        overlap *= coverage  # Scale down by coverage ratio

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
