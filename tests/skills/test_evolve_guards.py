"""Tests for the evolve_skill error guards and holdout scoring helpers.

Exercises the exact failure modes from esc-20260731-201857-1474:
  - 'error 1': OSError errno 24 EMFILE on artifact writes and during the
    holdout evaluation — must surface as an explicit exit-2 with a hint,
    never a bare traceback exit-1 that cron mislabels as an API error.
  - median aggregation / valset improvement estimation (growth waiver input).
"""

import errno
import pytest

from evolution.skills.evolve_skill import (
    estimate_improvement,
    run_holdout_evaluation,
    write_text_guarded,
)


class FakePrediction:
    def __init__(self, output):
        self.output = output


class FakeModule:
    """Minimal stand-in for SkillModule: returns a fixed prediction."""

    def __init__(self, output="ok", exc=None):
        self._output = output
        self._exc = exc

    def __call__(self, task_input):
        if self._exc is not None:
            raise self._exc
        return FakePrediction(self._output)


class FakeExample:
    def __init__(self, task_input="task"):
        self.task_input = task_input


class TestWriteTextGuarded:
    def test_emfile_exits_2_with_hint(self, capsys):
        class FailingPath:
            def __init__(self):
                self.name = "output/web-research/evolved_FAILED.md"

            def write_text(self, text):
                raise OSError(errno.EMFILE, "Too many open files")

        with pytest.raises(SystemExit) as excinfo:
            write_text_guarded(FailingPath(), "content", "failed variant")
        assert excinfo.value.code == 2
        out = capsys.readouterr().out
        assert "Too many open files" in out
        assert "ulimit" in out

    def test_other_oserror_warns_and_continues(self, capsys):
        class FailingPath:
            def write_text(self, text):
                raise OSError(errno.EACCES, "Permission denied")

        write_text_guarded(FailingPath(), "content", "metrics")  # must not raise
        out = capsys.readouterr().out
        assert "Could not write" in out

    def test_success_writes_quietly(self, tmp_path, capsys):
        target = tmp_path / "out.md"
        write_text_guarded(target, "hello", "artifact")
        assert target.read_text() == "hello"
        assert "Could not write" not in capsys.readouterr().out


class TestRunHoldoutEvaluation:
    def test_median_aggregation(self):
        # 3 samples per program; medians of [0.4, 0.5, 0.9] and [0.1, 0.2, 0.3]
        # are 0.5 and 0.2. Interleaving order: baseline, evolved per sample.
        baseline = FakeModule()
        evolved = FakeModule()
        scores = iter([0.4, 0.1, 0.5, 0.2, 0.9, 0.3])
        metric = lambda ex, pred: next(scores)
        examples = [FakeExample()]

        base_scores, evo_scores = run_holdout_evaluation(
            baseline, evolved, examples, metric, samples=3
        )
        assert base_scores == [0.5]
        assert evo_scores == [0.2]

    def test_emfile_during_eval_propagates_for_classification(self):
        # The holdout loop must not swallow the OSError — it propagates so the
        # caller can classify it as exit 2 (this was the 'error 1' crash point)
        def boom(ex, pred):
            raise OSError(errno.EMFILE, "Too many open files")

        with pytest.raises(OSError) as excinfo:
            run_holdout_evaluation(
                FakeModule(), FakeModule(), [FakeExample()], boom, samples=1
            )
        assert excinfo.value.errno == errno.EMFILE


class TestEstimateImprovement:
    def test_positive_improvement_detected(self):
        # baseline scores 0.4, evolved scores 0.6 → +0.2 improvement
        examples = [FakeExample(f"task{i}") for i in range(3)]
        baseline = FakeModule(output="base")
        evolved = FakeModule(output="evo")
        metric = lambda ex, pred: 0.4 if pred.output == "base" else 0.6

        improvement = estimate_improvement(
            baseline, evolved, examples, metric, samples=1
        )
        assert improvement == pytest.approx(0.2)

    def test_max_examples_cap(self):
        # Only the first max_examples examples are scored
        examples = [FakeExample(f"task{i}") for i in range(5)]
        baseline = FakeModule(output="base")
        evolved = FakeModule(output="evo")
        calls = {"n": 0}

        def metric(ex, pred):
            calls["n"] += 1
            return 0.5

        estimate_improvement(baseline, evolved, examples, metric, samples=1, max_examples=2)
        assert calls["n"] == 4  # 2 examples x 2 programs

    def test_no_regression(self):
        examples = [FakeExample()] * 2
        baseline = FakeModule(output="base")
        evolved = FakeModule(output="evo")
        metric = lambda ex, pred: 0.7 if pred.output == "base" else 0.5

        improvement = estimate_improvement(baseline, evolved, examples, metric, samples=1)
        assert improvement == pytest.approx(-0.2)
