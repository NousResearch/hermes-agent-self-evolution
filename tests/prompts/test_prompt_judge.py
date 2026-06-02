"""Tests for the SaveCallJudge — scores memory-save args against MEMORY_GUIDANCE rules."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evolution.prompts.prompt_judge import SaveCallJudge, judge_save_calls


def test_no_save_calls_yields_default():
    """No save calls at all → score 1.0 (vacuously correct). Layer 1 catches
    'should have saved but didn't'; Layer 2 only scores content of calls made."""
    assert judge_save_calls(judge=None, calls=[], expected_content=None) == 1.0


def test_invokes_judge_per_call_and_means():
    fake_judge = MagicMock(spec=SaveCallJudge)
    fake_judge.score.side_effect = [0.8, 0.6]
    calls = [
        {"action": "add", "content": "user prefers concise responses"},
        {"action": "replace", "content": "completed phase 3"},
    ]
    score = judge_save_calls(
        judge=fake_judge, calls=calls,
        expected_content="user preference about response style",
    )
    assert score == pytest.approx(0.7)
    assert fake_judge.score.call_count == 2


def test_caps_at_five_calls():
    """Pathological: agent saves on every turn. Judge at most 5; excess score 0."""
    fake_judge = MagicMock(spec=SaveCallJudge)
    fake_judge.score.return_value = 1.0
    calls = [{"action": "add", "content": f"item {i}"} for i in range(10)]
    score = judge_save_calls(judge=fake_judge, calls=calls, expected_content="any")
    # 5 scored 1.0, 5 unjudged scored 0 → mean 0.5
    assert score == pytest.approx(0.5)
    assert fake_judge.score.call_count == 5


def test_filters_non_save_actions():
    """Only content-bearing actions (add/replace) are judged; remove/read skipped."""
    fake_judge = MagicMock(spec=SaveCallJudge)
    fake_judge.score.return_value = 1.0
    calls = [
        {"action": "remove", "old_text": "x"},
        {"action": "read"},
        {"action": "add", "content": "real save"},
    ]
    score = judge_save_calls(judge=fake_judge, calls=calls, expected_content="any")
    assert score == pytest.approx(1.0)
    assert fake_judge.score.call_count == 1


def test_none_judge_or_expected_is_vacuous_pass():
    """A save call exists but no judge/rubric configured → don't penalize."""
    calls = [{"action": "add", "content": "x"}]
    assert judge_save_calls(judge=None, calls=calls, expected_content="r") == 1.0
    fake = MagicMock(spec=SaveCallJudge)
    assert judge_save_calls(judge=fake, calls=calls, expected_content=None) == 1.0
    fake.score.assert_not_called()


# ---- make_prompt_fitness_metric ----

from evolution.prompts.prompt_judge import (
    make_memoizing_splice_scorer,
    make_prompt_fitness_metric,
)


def _behavioral_pred(task_id="task-001", candidate="evolved body"):
    pred = type("Pred", (), {})()
    pred._closed_loop_task_id = task_id
    pred._candidate_text = candidate
    return pred


def test_metric_routes_behavioral_through_scorer():
    seen = []

    def fake_scorer(task_id, candidate_text):
        seen.append((task_id, candidate_text))
        return 0.85

    metric = make_prompt_fitness_metric(
        baseline_text="baseline", max_growth=0.2, closed_loop_scorer=fake_scorer,
    )
    result = metric(gold=object(), pred=_behavioral_pred())
    assert result.score == 0.85
    assert seen == [("task-001", "evolved body")]
    assert "BUDGET" in result.feedback  # length feedback present


def test_metric_without_task_id_scores_zero():
    metric = make_prompt_fitness_metric(
        baseline_text="b", max_growth=0.2, closed_loop_scorer=lambda *_: 1.0,
    )
    pred = type("Pred", (), {})()  # no _closed_loop_task_id
    result = metric(gold=object(), pred=pred)
    assert result.score == 0.0
    assert "behavioral" in result.feedback.lower()


def test_metric_without_scorer_scores_zero():
    metric = make_prompt_fitness_metric(
        baseline_text="b", max_growth=0.2, closed_loop_scorer=None,
    )
    result = metric(gold=object(), pred=_behavioral_pred())
    assert result.score == 0.0


# ---- make_memoizing_splice_scorer ----

def test_memoizing_scorer_splices_only_on_candidate_change():
    installs: list[str] = []
    scores = {"task-a": 0.7, "task-b": 0.9}

    scorer = make_memoizing_splice_scorer(
        install_fn=lambda text: installs.append(text),
        score_fn=lambda task_id: scores[task_id],
    )
    # Same candidate across two tasks → one install.
    assert scorer("task-a", "cand-1") == 0.7
    assert scorer("task-b", "cand-1") == 0.9
    assert installs == ["cand-1"]
    # New candidate → re-splice.
    assert scorer("task-a", "cand-2") == 0.7
    assert installs == ["cand-1", "cand-2"]
    # Back to a prior candidate is NOT cached across changes → re-splice.
    assert scorer("task-a", "cand-1") == 0.7
    assert installs == ["cand-1", "cand-2", "cand-1"]
