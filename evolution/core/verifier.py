"""Objective fitness verifiers for skill evolution.

A verifier grades agent output against checkable ground truth instead of
keyword overlap or LLM judgment. That makes the fitness signal:

  - non-circular: no LLM grades another LLM's paraphrase
  - non-gameable: echoing rubric vocabulary scores nothing
  - reproducible: the same output always gets the same score
  - free: grading is pure Python, no API calls

Each verifier owns both halves of the evaluation contract: it builds the
eval dataset (tasks whose answers are verifiable facts) and it scores
outputs against those facts. The two must travel together, because the
grader can only score tasks it knows the ground truth for.

Verifiers register themselves with @register_verifier and are looked up
by skill name via get_verifier(). verifier_metric() adapts a verifier to
the DSPy metric protocol, including GEPA's feedback-aware form.
"""

import abc
from collections.abc import Callable

import dspy

from evolution.core.dataset_builder import EvalDataset
from evolution.core.fitness import FitnessScore


class Verifier(abc.ABC):
    """Grades agent outputs for one skill against objective ground truth."""

    #: Skill name this verifier applies to (the SKILL.md directory name).
    skill_name: str = ""

    @abc.abstractmethod
    def build_dataset(self, num_cases: int = 24, seed: int = 13) -> EvalDataset:
        """Build a deterministic eval dataset of verifiable tasks.

        The returned examples carry human-readable rubrics in
        expected_behavior (so they stay compatible with judge and keyword
        scoring), but the authoritative answers live inside the verifier.
        """

    @abc.abstractmethod
    def score(self, task_input: str, output: str) -> FitnessScore:
        """Score an agent output for a task from this verifier's dataset.

        Returns a FitnessScore whose feedback explains exactly what was
        right or wrong. The feedback doubles as GEPA's reflection signal.
        """


_REGISTRY: dict[str, type[Verifier]] = {}


def register_verifier(cls: type[Verifier]) -> type[Verifier]:
    """Class decorator that registers a Verifier by its skill_name."""
    if not cls.skill_name:
        raise ValueError(f"{cls.__name__} must set a non-empty skill_name")
    _REGISTRY[cls.skill_name] = cls
    return cls


def get_verifier(skill_name: str) -> Verifier | None:
    """Return a verifier instance for a skill, or None if none exists."""
    from evolution.verifiers import load_builtins

    load_builtins()
    cls = _REGISTRY.get(skill_name)
    return cls() if cls else None


def registered_skills() -> list[str]:
    """Names of skills that have a registered verifier."""
    from evolution.verifiers import load_builtins

    load_builtins()
    return sorted(_REGISTRY)


def verifier_metric(verifier: Verifier) -> Callable:
    """Adapt a Verifier to the DSPy metric protocol.

    Handles all three calling conventions used across the pipeline:

      - metric(gold, pred)                        holdout comparison
      - metric(gold, pred, trace)                 MIPROv2 / classic DSPy
      - metric(gold, pred, trace, pred_name, pred_trace)   GEPA

    Returns a plain float normally. When GEPA asks for feedback on a
    specific predictor (pred_name is set), returns a dspy.Prediction with
    score and feedback so GEPA's reflection sees why the output scored
    the way it did.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        task_input = getattr(gold, "task_input", "") or ""
        output = getattr(pred, "output", "") or ""
        fitness = verifier.score(task_input, output)
        # Objective verification is an admission boundary: presentation and
        # procedure feedback may guide reflection, but they cannot compensate
        # for a factually wrong answer.
        score = fitness.composite if fitness.correctness > 0 else 0.0
        if pred_name is not None:
            return dspy.Prediction(score=score, feedback=fitness.feedback)
        return score

    metric.__name__ = f"{verifier.skill_name}_verifier_metric"
    return metric
