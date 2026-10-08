"""Tests for the LLM-judge fitness metric.

The pipeline's real signal is `LLMJudge.score()`'s composite. These tests
stub the judge so no network call happens, and assert that the metric:

- routes through the judge (not the old lexical-overlap proxy)
- returns the judge's composite score
- scores zero on empty output
- grades against the *current* candidate skill text
- degrades to a neutral score, once, when the judge is unreachable
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dspy  # noqa: E402

from evolution.core.fitness import (  # noqa: E402
    FitnessScore,
    LLMJudge,
    get_llm_judge,
    set_current_skill_text,
    set_llm_judge,
    skill_fitness_metric,
    warn_judge_unavailable,
    DEFAULT_MAX_SKILL_SIZE,
    _WARNED_JUDGE_FAILURE,
)
from evolution.skills.skill_module import find_skill  # noqa: E402


class StubJudge:
    """Records calls and returns a fixed composite."""

    def __init__(self, composite=0.75, error=None):
        self.composite = composite
        self.error = error
        self.calls = []

    def score(self, task_input, expected_behavior, agent_output,
              skill_text, artifact_size=None, max_size=None):
        if self.error:
            raise self.error
        self.calls.append({
            "task_input": task_input,
            "expected_behavior": expected_behavior,
            "agent_output": agent_output,
            "skill_text": skill_text,
            "artifact_size": artifact_size,
            "max_size": max_size,
        })
        # composite is a derived property, so drive it through the dimensions.
        return FitnessScore(correctness=self.composite, feedback="stub")


@pytest.fixture(autouse=True)
def clean_judge_state():
    _WARNED_JUDGE_FAILURE.clear()
    set_llm_judge(None)
    set_current_skill_text("")
    yield
    set_llm_judge(None)
    set_current_skill_text("")
    _WARNED_JUDGE_FAILURE.clear()


def _example(task="do the thing", expected="cite file:line"):
    return dspy.Example(task_input=task, expected_behavior=expected).with_inputs("task_input")


# ── routing through the judge ─────────────────────────────────────────────


def test_metric_returns_judge_composite():
    # StubJudge sets correctness=0.83; composite weights it at 0.5.
    set_llm_judge(StubJudge(composite=0.83))
    score = skill_fitness_metric(_example(), dspy.Prediction(output="a real answer"))
    assert score == pytest.approx(0.415)


def test_metric_is_not_lexical_overlap():
    """Guards the regression that made word-count overlap the fitness signal."""
    expected = "cite the file path and line number"
    # Output shares almost no vocabulary with expected but is a fine answer.
    output = "Confirmed defect at line 412 of App.tsx; severity blocking."
    set_llm_judge(StubJudge(composite=0.415))
    judged = skill_fitness_metric(
        dspy.Example(task_input="t", expected_behavior=expected).with_inputs("task_input"),
        dspy.Prediction(output=output),
    )
    # No skill text set, so no length penalty: composite == 0.5 * correctness.
    # The point is that the judge decided, not word overlap.
    assert judged == pytest.approx(0.5 * 0.415)


def test_metric_calls_judge_with_all_fields():
    judge = StubJudge()
    set_llm_judge(judge)
    skill_fitness_metric(
        dspy.Example(task_input="T", expected_behavior="E").with_inputs("task_input"),
        dspy.Prediction(output="O"),
    )
    assert len(judge.calls) == 1
    call = judge.calls[0]
    assert call["task_input"] == "T"
    assert call["expected_behavior"] == "E"
    assert call["agent_output"] == "O"


def test_metric_passes_length_penalty_bounds():
    judge = StubJudge()
    set_llm_judge(judge)
    set_current_skill_text("x" * 500)
    skill_fitness_metric(_example(), dspy.Prediction(output="O"))
    call = judge.calls[0]
    assert call["artifact_size"] == 500
    assert call["max_size"] == DEFAULT_MAX_SKILL_SIZE


def test_metric_scores_zero_on_empty_output_without_calling_judge():
    judge = StubJudge()
    set_llm_judge(judge)
    assert skill_fitness_metric(_example(), dspy.Prediction(output="   ")) == 0.0
    assert judge.calls == []


# ── grading against the current candidate ────────────────────────────────


def test_metric_grades_against_current_skill_text():
    judge = StubJudge()
    set_llm_judge(judge)
    set_current_skill_text("CANDIDATE INSTRUCTIONS")
    skill_fitness_metric(_example(), dspy.Prediction(output="O"))
    assert judge.calls[0]["skill_text"] == "CANDIDATE INSTRUCTIONS"


def test_current_skill_text_updates_between_candidates():
    judge = StubJudge()
    set_llm_judge(judge)

    set_current_skill_text("BASELINE")
    skill_fitness_metric(_example(), dspy.Prediction(output="O"))

    set_current_skill_text("EVOLVED")
    skill_fitness_metric(_example(), dspy.Prediction(output="O"))

    assert [c["skill_text"] for c in judge.calls] == ["BASELINE", "EVOLVED"]


# ── failure handling ──────────────────────────────────────────────────────


def test_metric_degrades_to_neutral_when_judge_unreachable(capsys):
    set_llm_judge(StubJudge(error=RuntimeError("no API key")))
    score = skill_fitness_metric(_example(), dspy.Prediction(output="O"))
    assert score == 0.5
    err = capsys.readouterr().err
    assert "not meaningful" in err.lower() or "NOT meaningful" in err


def test_judge_failure_warns_once_per_cause(capsys):
    set_llm_judge(StubJudge(error=RuntimeError("no API key")))
    for _ in range(4):
        skill_fitness_metric(_example(), dspy.Prediction(output="O"))
    err = capsys.readouterr().err
    assert err.count("LLM judge unavailable") == 1


def test_warn_judge_unavailable_dedupes():
    warn_judge_unavailable(ValueError("boom"))
    warn_judge_unavailable(ValueError("boom"))
    assert len([k for k in _WARNED_JUDGE_FAILURE if "ValueError" in k]) == 1


# ── the judge actually composes as documented ────────────────────────────


def test_composite_weights():
    s = FitnessScore(correctness=1.0, procedure_following=1.0, conciseness=1.0)
    assert s.composite == pytest.approx(1.0)
    s2 = FitnessScore(correctness=1.0, procedure_following=0.0, conciseness=0.0)
    assert s2.composite == pytest.approx(0.5)
    s3 = FitnessScore(correctness=1.0, length_penalty=0.3)
    assert s3.composite == pytest.approx(0.2)


def test_composite_never_negative():
    assert FitnessScore(correctness=0.0, length_penalty=0.3).composite == 0.0


def test_get_llm_judge_is_singleton():
    a = get_llm_judge()
    b = get_llm_judge()
    assert a is b


# ── skill discovery fallback ──────────────────────────────────────────────


def test_find_skill_falls_back_to_profile_skills():
    repo = Path.home() / ".hermes" / "hermes-agent"
    found = find_skill("gate-evidence-ledger", repo)
    assert found is not None
    assert found.name == "SKILL.md"
    assert "gate-evidence-ledger" in str(found)


def test_find_skill_returns_none_for_unknown():
    repo = Path.home() / ".hermes" / "hermes-agent"
    assert find_skill("definitely-not-a-real-skill-xyz", repo) is None