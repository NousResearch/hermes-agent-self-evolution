"""Fitness functions for evaluating evolved artifacts.

Two defects lived here and in the metric that wrapped this module, and both
are corrected in the design below.

*The judge was shown the wrong artifact.* Every candidate was scored with the
baseline skill body passed as ``skill_text``, so GEPA's reflective feedback —
the entire mechanism by which it improves — described an artifact that was not
under evaluation. The candidate now carries its own text on the prediction
(:meth:`SkillModule.forward` attaches it) and the judge reads that.

*The search and the report measured different things.* GEPA optimized the
judge composite while the holdout comparison used a bag-of-words overlap
heuristic, so the reported "improvement" was not the quantity being improved.
There is now one metric, :func:`make_fitness_metric`, used by both.

Size does not live here at all any more. It belongs in
:mod:`evolution.core.objectives`, where the penalty is computed from the
candidate rather than from a constant.
"""

from __future__ import annotations

import dspy
from dataclasses import dataclass
from typing import Callable, Optional

from evolution.core.config import EvolutionConfig
from evolution.core.dspy_lm import make_dspy_lm
from evolution.core.objectives import ObjectiveVector, ObjectiveWeights


@dataclass
class FitnessScore:
    """What the judge thought of one response.

    Purely a quality verdict. Length is deliberately absent — folding a size
    penalty in here is what hid the artifact-size problem in the first place.
    """

    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted quality score in [0, 1]."""
        return max(
            0.0,
            min(
                1.0,
                0.5 * self.correctness
                + 0.3 * self.procedure_following
                + 0.2 * self.conciseness,
            ),
        )


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides textual feedback
    that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        Address the feedback to whoever will edit the skill instructions: say
        what to change in the instructions, not just what the response got
        wrong.
        """

        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The exact skill/instructions this response was produced under")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable changes to make to the skill instructions")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
    ) -> FitnessScore:
        """Score one agent output.

        Args:
            skill_text: The instructions the response was actually produced
                under — the candidate, never the baseline.
        """
        lm = make_dspy_lm(
            self.config.eval_model,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
        )

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        return FitnessScore(
            correctness=_parse_score(result.correctness),
            procedure_following=_parse_score(result.procedure_following),
            conciseness=_parse_score(result.conciseness),
            feedback=str(result.feedback),
        )


def heuristic_quality(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """Cheap keyword-overlap proxy for the judge.

    Kept only as a declared fallback for when the judge call fails mid-run —
    losing a whole optimization to one flaky API response is worse than one
    example scored crudely. It must never be used to *report* results
    alongside judge-driven optimization; that mismatch is what made the old
    holdout numbers meaningless.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""

    if not agent_output.strip():
        return 0.0
    if not expected.strip():
        return 0.5

    expected_words = set(expected.lower().split())
    output_words = set(agent_output.lower().split())
    if not expected_words:
        return 0.5

    overlap = len(expected_words & output_words) / len(expected_words)
    return min(1.0, max(0.0, 0.3 + 0.7 * overlap))


def candidate_text(prediction: dspy.Prediction, fallback: str = "") -> str:
    """The skill text a prediction was produced under.

    ``SkillModule.forward`` attaches the live instructions to every
    prediction, which is what makes candidate-accurate judging possible. The
    fallback covers predictions from other modules.
    """
    text = getattr(prediction, "skill_text", "") or ""
    return text if text.strip() else fallback


def make_fitness_metric(
    config: EvolutionConfig,
    baseline_text: str,
    size_budget: int,
    weights: Optional[ObjectiveWeights] = None,
    judge: Optional[LLMJudge] = None,
    on_vector: Optional[Callable[[ObjectiveVector], None]] = None,
) -> Callable:
    """Build the one metric used for both optimization and reporting.

    The returned callable satisfies GEPA's five-argument feedback protocol and
    also works as a plain DSPy metric, so the search and the holdout evaluation
    cannot drift apart the way they previously did.

    Args:
        baseline_text: The original artifact, for the growth term.
        size_budget: Corpus-derived character budget for this artifact.
        on_vector: Optional sink for the full objective vector of every
            evaluation, so callers can build a Pareto front afterwards.
    """
    _judge = judge or LLMJudge(config)
    _weights = weights or ObjectiveWeights()
    baseline_chars = len(baseline_text or "")

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        text = candidate_text(pred, fallback=baseline_text)

        try:
            score = _judge.score(
                task_input=getattr(gold, "task_input", "") or "",
                expected_behavior=getattr(gold, "expected_behavior", "") or "",
                agent_output=getattr(pred, "output", "") or "",
                skill_text=text,
            )
            quality = score.composite
            feedback = score.feedback
        except Exception as judge_error:  # noqa: BLE001 — never fail the whole run
            quality = heuristic_quality(gold, pred)
            feedback = (
                f"Judge unavailable ({judge_error}); scored by keyword overlap only. "
                "Treat this example's score as low-confidence."
            )

        vector = ObjectiveVector(
            quality=quality,
            size_chars=len(text),
            size_budget=size_budget,
            baseline_chars=baseline_chars,
            max_growth=config.max_prompt_growth,
            tokens=int(getattr(pred, "total_tokens", 0) or 0),
            tool_calls=int(getattr(pred, "tool_calls", 0) or 0),
            feedback=feedback,
        )
        if on_vector is not None:
            on_vector(vector)

        final = vector.scalarize(_weights)

        # Tell the optimizer *why* it lost points on size. Without this the
        # reflection LM sees a lower score after a rewrite and has no way to
        # attribute it to length, so it keeps growing the artifact.
        size_note = _size_feedback(vector)
        if size_note:
            feedback = f"{feedback}\n\n{size_note}"

        return dspy.Prediction(score=final, feedback=feedback)

    return metric


def _size_feedback(vector: ObjectiveVector) -> str:
    """A plain-language note about size pressure, or empty when there is none."""
    penalties = vector.size_penalty() + vector.growth_penalty()
    if penalties <= 0.001:
        return ""

    parts = [
        f"SIZE: this version is {vector.size_chars:,} characters against a "
        f"{vector.size_budget:,} budget ({vector.size_ratio:.0%} of it), "
        f"costing {penalties:.2f} of score."
    ]
    if vector.growth_penalty() > 0.001:
        parts.append(
            f"It has grown {vector.growth:+.0%} over the original, past the "
            f"{vector.max_growth:.0%} allowance."
        )
    parts.append(
        "Cut redundant explanation and consolidate overlapping sections rather "
        "than adding more guidance."
    )
    return " ".join(parts)


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure


# Backwards-compatible alias. The old name promised a DSPy metric but returned
# a bare keyword-overlap float; callers should use make_fitness_metric.
def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    """Deprecated: keyword-overlap proxy. Use :func:`make_fitness_metric`."""
    return heuristic_quality(example, prediction)
