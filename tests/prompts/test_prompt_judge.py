"""Tests for the SaveCallJudge — scores memory-save args against MEMORY_GUIDANCE rules."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evolution.prompts.prompt_judge import SaveCallJudge, ScoreResult, judge_save_calls


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
    assert scorer("task-a", "cand-1").score == pytest.approx(0.7)
    assert scorer("task-b", "cand-1").score == pytest.approx(0.9)
    assert installs == ["cand-1"]
    # New candidate → re-splice.
    assert scorer("task-a", "cand-2").score == pytest.approx(0.7)
    assert installs == ["cand-1", "cand-2"]
    # Back to a prior candidate is NOT cached across changes → re-splice.
    assert scorer("task-a", "cand-1").score == pytest.approx(0.7)
    assert installs == ["cand-1", "cand-2", "cand-1"]


# ---- ScoreResult + SaveCallJudge.score_with_feedback ----

def _make_fake_prediction(quality: str, feedback: str):
    pred = MagicMock()
    pred.quality = quality
    pred.feedback = feedback
    return pred


def _make_judge_with_fake_lm(quality: str, feedback: str):
    """Build a SaveCallJudge whose dspy ChainOfThought call is monkeypatched.

    The patch targets ``SaveCallJudge.judge`` (the bound ChainOfThought instance)
    so that calling it returns a fake prediction with the given quality/feedback
    fields — exactly the approach used by the existing judge_save_calls tests
    (which mock the judge at the SaveCallJudge level via MagicMock(spec=...)).
    No real LM call is made.
    """
    from evolution.core.config import EvolutionConfig

    cfg = MagicMock(spec=EvolutionConfig)
    lm_cfg = MagicMock()
    lm_cfg.model = "openai/gpt-4o-mini"
    lm_cfg.lm_kwargs = {}
    cfg.get_lm.return_value = lm_cfg

    judge = SaveCallJudge(config=cfg)
    fake_pred = _make_fake_prediction(quality=quality, feedback=feedback)
    judge.judge = MagicMock(return_value=fake_pred)
    return judge


def test_score_result_is_dataclass():
    r = ScoreResult(score=0.8, feedback="x")
    assert r.score == 0.8
    assert r.feedback == "x"


def test_score_result_feedback_defaults_to_empty():
    r = ScoreResult(score=0.5)
    assert r.feedback == ""


def test_score_with_feedback_returns_score_and_feedback():
    judge = _make_judge_with_fake_lm(quality="0.9", feedback="good save")
    with patch("dspy.LM", return_value=MagicMock()), \
         patch("dspy.context") as mock_ctx:
        mock_ctx.return_value.__enter__ = lambda s: s
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        result = judge.score_with_feedback(
            task="store user preference",
            expected_content="durable preference fact",
            saved_content="user prefers dark mode",
        )
    assert isinstance(result, ScoreResult)
    assert result.score == pytest.approx(0.9)
    assert result.feedback == "good save"


def test_score_still_returns_bare_float():
    judge = _make_judge_with_fake_lm(quality="0.75", feedback="minor issue")
    with patch("dspy.LM", return_value=MagicMock()), \
         patch("dspy.context") as mock_ctx:
        mock_ctx.return_value.__enter__ = lambda s: s
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        result = judge.score(
            task="store user preference",
            expected_content="durable preference fact",
            saved_content="user prefers dark mode",
        )
    assert isinstance(result, float)
    assert result == pytest.approx(0.75)


def test_score_with_feedback_empty_feedback_when_judge_returns_empty():
    judge = _make_judge_with_fake_lm(quality="1.0", feedback="")
    with patch("dspy.LM", return_value=MagicMock()), \
         patch("dspy.context") as mock_ctx:
        mock_ctx.return_value.__enter__ = lambda s: s
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        result = judge.score_with_feedback(
            task="t", expected_content="e", saved_content="s",
        )
    assert result.score == pytest.approx(1.0)
    assert result.feedback == ""


# ---- make_memoizing_splice_scorer: ScoreResult propagation ----

def test_memoizing_scorer_propagates_score_result():
    """score_fn returning ScoreResult → scorer returns that ScoreResult (score + feedback)."""
    installs: list[str] = []

    def score_fn(task_id: str):
        return ScoreResult(0.4, "patched 1/4")

    scorer = make_memoizing_splice_scorer(
        install_fn=lambda text: installs.append(text),
        score_fn=score_fn,
    )
    result = scorer("task-a", "cand-x")
    assert isinstance(result, ScoreResult)
    assert result.score == pytest.approx(0.4)
    assert result.feedback == "patched 1/4"


def test_memoizing_scorer_propagates_install_memoization_with_score_result():
    """Install-memoization still works when score_fn returns ScoreResult."""
    installs: list[str] = []
    call_count = [0]

    def score_fn(task_id: str):
        call_count[0] += 1
        return ScoreResult(0.5, "ok")

    scorer = make_memoizing_splice_scorer(
        install_fn=lambda text: installs.append(text),
        score_fn=score_fn,
    )
    scorer("task-a", "cand-1")
    scorer("task-b", "cand-1")  # same candidate → no re-install
    assert installs == ["cand-1"]
    scorer("task-a", "cand-2")  # new candidate → re-install
    assert installs == ["cand-1", "cand-2"]


def test_memoizing_scorer_normalizes_bare_float():
    """score_fn returning a bare float → scorer returns ScoreResult(score=x, feedback="")."""
    scorer = make_memoizing_splice_scorer(
        install_fn=lambda text: None,
        score_fn=lambda task_id: 0.7,
    )
    result = scorer("task-a", "cand-x")
    assert isinstance(result, ScoreResult)
    assert result.score == pytest.approx(0.7)
    assert result.feedback == ""


# ---- make_prompt_fitness_metric: ScoreResult propagation ----

def test_metric_merges_behavioral_feedback_with_budget_note():
    """closed_loop_scorer returning ScoreResult → metric combines agent feedback + budget note."""
    def fake_scorer(task_id, candidate_text):
        return ScoreResult(0.4, "agent patched 0/4 reps")

    metric = make_prompt_fitness_metric(
        baseline_text="baseline body",
        max_growth=0.2,
        closed_loop_scorer=fake_scorer,
    )
    result = metric(gold=object(), pred=_behavioral_pred(task_id="t1", candidate="..."))
    assert result.score == pytest.approx(0.4)
    assert "agent patched 0/4 reps" in result.feedback
    assert "BUDGET" in result.feedback


def test_metric_bare_float_scorer_feedback_is_budget_only():
    """closed_loop_scorer returning bare float → feedback is just the budget note."""
    metric = make_prompt_fitness_metric(
        baseline_text="baseline",
        max_growth=0.2,
        closed_loop_scorer=lambda task_id, candidate_text: 0.8,
    )
    result = metric(gold=object(), pred=_behavioral_pred())
    assert result.score == pytest.approx(0.8)
    assert "BUDGET" in result.feedback
    # no extra behavioral feedback prefix
    assert result.feedback.startswith("[BUDGET]")


def test_metric_task_id_none_degenerate_path_unchanged():
    """Degenerate path (no task_id) still scores 0 with diagnostic feedback."""
    metric = make_prompt_fitness_metric(
        baseline_text="b", max_growth=0.2,
        closed_loop_scorer=lambda *_: ScoreResult(1.0, "should not appear"),
    )
    pred = type("Pred", (), {})()  # no _closed_loop_task_id
    result = metric(gold=object(), pred=pred)
    assert result.score == 0.0
    assert "behavioral" in result.feedback.lower()
