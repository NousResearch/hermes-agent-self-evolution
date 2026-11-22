"""Fitness functions for evaluating evolved skills.

The optimiser-facing metric is :class:`SkillMetric`: a pluggable scorer
wrapped in GEPA's feedback-metric calling convention
``(gold, pred, trace, pred_name, pred_trace)``. It returns
``dspy.Prediction(score=..., feedback=...)``. GEPA reads the score for
selection and the feedback text for reflective mutation. ``SkillMetric.score``
gives the plain float for holdout evaluation.

Scorers (``(example, prediction) -> (score in [0, 1], feedback)``):

* :func:`judge_scorer` (default): an LLM judge scores the response against
  the example's ``expected_behavior`` rubric on correctness, procedure
  following and conciseness, and returns actionable feedback. The judge sees
  the task, rubric and response only, NOT the skill text being optimised, so
  the text under optimisation cannot steer its own score.
* :func:`keyword_overlap_scorer` (opt-in, ``--metric keyword``): share of
  rubric terms that appear in the response. Free and offline, but only a
  proxy: it rewards echoing rubric words and verbosity, not correctness.

Limits (read before trusting a score):

* With ``--eval-source synthetic`` the tasks and rubrics are themselves
  LLM-generated from the skill, and an LLM judges the answers. This is
  circular: a higher score means "more aligned with what an LLM thinks the
  skill implies", not "better on real work". Prefer ``golden`` or
  ``sessiondb`` datasets.
* An LLM judge is noisy and biased (length, style, self-preference). Use a
  judge model different from the task model where possible (``--judge-model``).
* Each metric call costs one task-model call plus one judge call.
* A judge score that cannot be parsed counts as 0 and says so in the
  feedback; it is never silently treated as a neutral 0.5.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

import dspy

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

    Scores agent outputs on multiple dimensions and provides textual feedback
    that GEPA uses for reflective mutation. Pass ``lm`` to choose (or stub)
    the judge model; otherwise ``dspy.LM(config.eval_model)`` is used.
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
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: Optional[EvolutionConfig] = None, *, lm=None):
        self.config = config
        self._lm = lm
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    @property
    def lm(self):
        if self._lm is None:
            if self.config is None:
                raise ValueError("LLMJudge needs an lm or a config with eval_model")
            self._lm = dspy.LM(self.config.eval_model)
        return self._lm

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""
        with dspy.context(lm=self.lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
            )

        notes = []
        scores = {}
        for name in ("correctness", "procedure_following", "conciseness"):
            value = _parse_score(getattr(result, name, None))
            if value is None:
                notes.append(f"judge returned an unparseable {name} score (counted as 0)")
                value = 0.0
            scores[name] = value

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        feedback = str(getattr(result, "feedback", "") or "").strip()
        return FitnessScore(
            length_penalty=length_penalty,
            feedback="\n".join([feedback, *notes]).strip(),
            **scores,
        )


# ── Pluggable, feedback-producing metric ────────────────────────────────────

Scorer = Callable[[dspy.Example, dspy.Prediction], "tuple[float, str]"]


def judge_scorer(judge: LLMJudge) -> Scorer:
    """Rubric-based LLM-judge scorer. The judge never sees the skill text."""
    def score(example, prediction):
        result = judge.score(
            task_input=getattr(example, "task_input", "") or "",
            expected_behavior=getattr(example, "expected_behavior", "") or "",
            agent_output=getattr(prediction, "output", "") or "",
        )
        detail = (f"correctness={result.correctness:.2f}, "
                  f"procedure_following={result.procedure_following:.2f}, "
                  f"conciseness={result.conciseness:.2f}")
        return result.composite, f"{detail}. {result.feedback}".strip()
    score.name = "llm-judge"
    return score


_WORD = re.compile(r"[a-z0-9][a-z0-9_\-']+")
_STOPWORDS = frozenset(
    "the and for with that this from should into your about which their have will "
    "when then than them they what where while also must does each such only been".split())


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOPWORDS}


def keyword_overlap_scorer(example, prediction) -> tuple[float, str]:
    """PROXY: share of rubric terms present in the response. Not a quality measure."""
    expected = _terms(getattr(example, "expected_behavior", "") or "")
    output = _terms(getattr(prediction, "output", "") or "")
    if not expected:
        return 0.0, "The rubric has no usable terms; this example cannot be scored."
    hit = expected & output
    missing = sorted(expected - output)
    feedback = (f"Covered {len(hit)}/{len(expected)} rubric terms (keyword proxy)."
                + (f" Missing: {', '.join(missing[:12])}." if missing else ""))
    return len(hit) / len(expected), feedback


keyword_overlap_scorer.name = "keyword-overlap"


class SkillMetric:
    """GEPA-compatible metric: score plus textual feedback from a pluggable scorer."""

    def __init__(self, scorer: Scorer, name: Optional[str] = None):
        if not callable(scorer):
            raise TypeError("scorer must be callable (example, prediction) -> (score, feedback)")
        self.scorer = scorer
        self.name = name or getattr(scorer, "name", getattr(scorer, "__name__", "custom"))

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        output = getattr(pred, "output", "") or ""
        if not str(output).strip():
            return dspy.Prediction(score=0.0, feedback="The response was empty.")
        score, feedback = self.scorer(gold, pred)
        score = _clamp(score)
        return dspy.Prediction(score=score,
                               feedback=str(feedback or f"Score {score:.3f}."))

    def score(self, gold, pred) -> float:
        return float(self(gold, pred).score)


def make_skill_metric(kind: str, *, judge_lm=None, config: Optional[EvolutionConfig] = None
                      ) -> SkillMetric:
    """``"judge"`` (default pipeline metric) or ``"keyword"`` (offline proxy)."""
    if kind == "judge":
        return SkillMetric(judge_scorer(LLMJudge(config, lm=judge_lm)), name="llm-judge")
    if kind == "keyword":
        return SkillMetric(keyword_overlap_scorer, name="keyword-overlap")
    raise ValueError(f"unknown metric {kind!r}; expected 'judge' or 'keyword'")


def skill_fitness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Deprecated float-only keyword proxy, kept for callers of the old API.

    Use :func:`make_skill_metric` for optimisation (GEPA needs feedback).
    """
    return SkillMetric(keyword_overlap_scorer).score(example, prediction)


def _clamp(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _parse_score(value) -> Optional[float]:
    """Parse a judge score into [0, 1]; None when it cannot be parsed."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _clamp(value) if math.isfinite(value) else None
    try:
        parsed = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    return _clamp(parsed) if math.isfinite(parsed) else None
