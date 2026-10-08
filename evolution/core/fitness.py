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


def _lm_for(model: str):
    """Build a dspy LM, preferring the Responses API for a local router.

    Imported lazily from evolve_skill to avoid a circular import: that module
    owns the factory because it is the CLI entrypoint.
    """
    try:
        from evolution.skills.evolve_skill import make_lm
        return make_lm(model)
    except Exception:  # noqa: BLE001 - fall back to DSPy's default handling
        return dspy.LM(model)


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

        lm = _lm_for(self.config.eval_model)

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


_JUDGE_SINGLETON: Optional["LLMJudge"] = None
_CURRENT_SKILL_TEXT: str = ""


def get_llm_judge(config: Optional[EvolutionConfig] = None) -> "LLMJudge":
    """Return a shared LLMJudge, constructing one from `config` on first use."""
    global _JUDGE_SINGLETON
    if _JUDGE_SINGLETON is None:
        _JUDGE_SINGLETON = LLMJudge(config or EvolutionConfig())
    return _JUDGE_SINGLETON


def set_llm_judge(judge: Optional["LLMJudge"]) -> None:
    """Override or clear the shared judge. Used by tests.

    Accepts any object exposing ``score(...) -> FitnessScore``; tests inject a
    stub rather than constructing a real judge.
    """
    global _JUDGE_SINGLETON
    _JUDGE_SINGLETON = judge


def set_current_skill_text(text: str) -> None:
    """Record the skill variant currently being scored.

    GEPA calls the metric as `metric(example, prediction, trace)`, so the
    candidate skill cannot be passed through the signature. The judge must
    see the *same* instructions the candidate was given, otherwise it grades
    output against the wrong rubric. Callers set this before each candidate.
    """
    global _CURRENT_SKILL_TEXT
    _CURRENT_SKILL_TEXT = text or ""


def skill_fitness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """DSPy-compatible metric function for skill optimization.

    This is what gets passed to dspy.GEPA(metric=...).

    Returns the rubric-based composite from :class:`LLMJudge` (0.5 correctness
    + 0.3 procedure_following + 0.2 conciseness, minus length penalty). That
    score is the real signal: GEPA reads its textual feedback to decide what
    to mutate next.

    Cost note: unlike a lexical-overlap proxy, this spends one LLM call per
    scored example, so a run costs proportionally more per iteration.

    If the judge is unreachable the metric degrades to neutral rather than
    failing the run, and says so once — an optimisation driven by a silent
    fallback score would be worse than a visible stop.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""
    skill_text = getattr(example, "skill_text", "") or _CURRENT_SKILL_TEXT

    if not agent_output.strip():
        return 0.0

    # Read the size ceiling defensively: the judge is swappable, and a stub or
    # alternate judge without a config must not break scoring.
    try:
        max_size = get_llm_judge().config.max_skill_size
    except AttributeError:
        max_size = DEFAULT_MAX_SKILL_SIZE

    try:
        judge = get_llm_judge()
        score = judge.score(
            task_input=task,
            expected_behavior=expected,
            agent_output=agent_output,
            skill_text=skill_text,
            artifact_size=len(skill_text) if skill_text else None,
            max_size=max_size,
        )
    except Exception as e:  # noqa: BLE001 - never let scoring kill the run
        warn_judge_unavailable(e)
        return 0.5

    return score.composite


DEFAULT_MAX_SKILL_SIZE = 15_000  # mirrors EvolutionConfig.max_skill_size


_WARNED_JUDGE_FAILURE: set[str] = set()


def warn_judge_unavailable(err: Exception) -> None:
    """Print the judge failure once per distinct cause, not once per example."""
    key = f"{type(err).__name__}: {str(err)[:120]}"
    if key in _WARNED_JUDGE_FAILURE:
        return
    _WARNED_JUDGE_FAILURE.add(key)
    console_stderr(
        f"[yellow]LLM judge unavailable ({type(err).__name__}: {str(err)[:200]}). "
        f"Falling back to a neutral 0.5 — scores are NOT meaningful.[/yellow]"
    )


def console_stderr(message: str) -> None:
    """Write a warning to stderr so it never pollutes stdout parsing."""
    import sys
    print(message, file=sys.stderr)


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
