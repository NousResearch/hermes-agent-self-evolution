"""Tests for composite code fitness.

The gate runners are injected, so nothing here starts a benchmark or the
hermes-agent test suite. Reproduction scripts are real files run with
``sys.executable`` in tmp_path: that path has to work for real, and it costs
milliseconds. No network, no LM.
"""

import json
import sys

import pytest

from evolution.core.gates import GateResult, GateStatus
from evolution.code.fitness_code import (
    DEFAULT_RANKING_RESOLUTION,
    BaselineSnapshot,
    BugReproduction,
    CandidateRanking,
    CodeFitness,
    CodeFitnessEvaluator,
    FitnessError,
    FitnessWeights,
    PerTestGateResult,
    PerTestPytestRunner,
    ReproResult,
    ReproStatus,
    ReproTrials,
    compare_test_suites,
    parse_pytest_outcomes,
    pytest_outcomes_from_result,
    rank_candidates,
)

BEFORE = '''"""Toy tools."""


def read_lines(path, limit=10):
    """Return up to *limit* lines."""
    try:
        with open(path) as handle:
            return handle.read().splitlines()[:limit - 1]
    except OSError:
        return []
'''

# The fix: same shape, one slice corrected.
FIXED = BEFORE.replace("[:limit - 1]", "[:limit]")

# Fixes the bug but changes the signature, which the guardrails refuse.
UNSAFE = FIXED.replace("def read_lines(path, limit=10):", "def read_lines(path, limit=10, encoding='utf-8'):")


def passed(name="pytest", score=None, message="ok"):
    return GateResult(name, GateStatus.PASSED, message, score=score)


def failed(name="pytest", message="1 failed"):
    return GateResult(name, GateStatus.FAILED, message)


def unavailable(name="tblite", message="not found"):
    return GateResult(name, GateStatus.UNAVAILABLE, message)


class Recorder:
    """A stand-in gate runner that records how often it was called."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, repo, *args, **kwargs):
        self.calls += 1
        if len(self.results) == 1:
            return self.results[0]
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class NamedBenchmarks:
    """Benchmark runner that answers per benchmark name."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, repo, name, **kwargs):
        self.calls.append(name)
        return self.mapping[name]


def make_evaluator(tmp_path, **kwargs):
    kwargs.setdefault("pytest_runner", Recorder(passed()))
    kwargs.setdefault("benchmark_runner", Recorder(unavailable()))
    return CodeFitnessEvaluator(repo=tmp_path, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Bug reproduction
# ──────────────────────────────────────────────────────────────────────────


def write_script(tmp_path, name, body, mode=None):
    path = tmp_path / name
    path.write_text(body)
    if mode is not None:
        path.chmod(mode)
    return path


class TestBugReproductionCommand:
    def test_plain_script_runs_as_a_script(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "pass\n"))
        assert repro.command("py") == ["py", str(repro.script)]

    def test_test_prefixed_script_runs_under_pytest(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "test_repro.py", "def test_x():\n    pass\n"))
        assert repro.command("py")[:4] == ["py", "-m", "pytest", str(repro.script)]

    def test_test_suffixed_script_runs_under_pytest(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "issue742_test.py", "def test_x():\n    pass\n"))
        assert "pytest" in repro.command("py")

    def test_executable_non_python_script_runs_directly(self, tmp_path):
        script = write_script(tmp_path, "repro.sh", "#!/bin/sh\nexit 0\n", mode=0o755)
        assert BugReproduction(script=script).command() == [str(script)]

    def test_unrunnable_script_is_an_error(self, tmp_path):
        script = write_script(tmp_path, "repro.txt", "not a script\n", mode=0o644)
        with pytest.raises(FitnessError, match="do not know how to run"):
            BugReproduction(script=script).command()

    def test_explicit_interpreter_wins(self, tmp_path):
        repro = BugReproduction(
            script=write_script(tmp_path, "repro.py", "pass\n"), python="/opt/py"
        )
        assert repro.command("ignored")[0] == "/opt/py"


class TestBugReproductionRun:
    def test_missing_script_is_unavailable_not_fixed(self, tmp_path):
        repro = BugReproduction(script=tmp_path / "ghost.py")
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.UNAVAILABLE
        assert not result.fixed
        assert not result.measured

    def test_exit_zero_means_fixed(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "import sys\nsys.exit(0)\n"))
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.FIXED
        assert result.fixed and result.measured
        assert result.exit_code == 0

    def test_non_zero_exit_means_the_bug_is_present(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "import sys\nsys.exit(3)\n"))
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.PRESENT
        assert not result.fixed
        assert result.exit_code == 3

    def test_present_marker_overrides_a_clean_exit(self, tmp_path):
        repro = BugReproduction(
            script=write_script(tmp_path, "repro.py", "print('BUG_PRESENT: still truncating')\n")
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.PRESENT

    def test_fixed_marker_overrides_a_dirty_exit(self, tmp_path):
        repro = BugReproduction(
            script=write_script(
                tmp_path, "repro.py", "import sys\nprint('BUG_FIXED')\nsys.exit(1)\n"
            )
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.FIXED

    def test_timeout_is_an_error_not_a_fix(self, tmp_path):
        repro = BugReproduction(
            script=write_script(tmp_path, "repro.py", "import time\ntime.sleep(30)\n"),
            timeout=1,
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.ERROR
        assert not result.fixed
        assert "timed out" in result.message

    def test_a_pytest_style_reproduction_works(self, tmp_path):
        repro = BugReproduction(
            script=write_script(
                tmp_path, "test_issue_742.py", "def test_fixed():\n    assert 1 == 1\n"
            )
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.FIXED

    def test_a_failing_pytest_reproduction_reports_the_bug(self, tmp_path):
        repro = BugReproduction(
            script=write_script(
                tmp_path, "test_issue_742.py", "def test_fixed():\n    assert 1 == 2\n"
            )
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.PRESENT
        assert result.details

    def test_result_is_json_serialisable(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "pass\n"))
        blob = json.loads(json.dumps(repro.run(tmp_path, sys.executable).to_dict()))
        assert blob["status"] == "fixed"


class FakeRepro(BugReproduction):
    """A reproduction whose verdict is fixed in advance."""

    def __init__(self, status):
        super().__init__(script=__file__)
        self._status = status

    def run(self, repo, python=None):
        from evolution.code.fitness_code import ReproResult

        return ReproResult(self._status, f"stubbed {self._status.value}")


class ScriptedRepro(BugReproduction):
    """A reproduction that walks a fixed list of verdicts, then repeats the last."""

    def __init__(self, *statuses):
        super().__init__(script=__file__)
        self.statuses = list(statuses)
        self.calls = 0

    def run(self, repo, python=None):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return ReproResult(status, f"stubbed {status.value}")


def trials(*statuses, confidence=0.95):
    return ReproTrials(
        runs=[ReproResult(s, f"stubbed {s.value}") for s in statuses],
        confidence=confidence,
    )


# ──────────────────────────────────────────────────────────────────────────
# Composite scoring
# ──────────────────────────────────────────────────────────────────────────


class TestHardGate:
    def test_a_safety_failure_skips_the_expensive_gates(self, tmp_path):
        runner = Recorder(passed())
        evaluator = make_evaluator(tmp_path, pytest_runner=runner)
        fitness = evaluator.evaluate(BEFORE, UNSAFE, label="c01")

        assert not fitness.accepted
        assert fitness.total == 0.0
        assert fitness.rejection_reason.startswith("safety:")
        assert fitness.pytest_result.status is GateStatus.SKIPPED
        assert runner.calls == 0

    def test_an_unchanged_candidate_is_rejected_without_running_anything(self, tmp_path):
        runner = Recorder(passed())
        evaluator = make_evaluator(tmp_path, pytest_runner=runner)
        fitness = evaluator.evaluate(BEFORE, BEFORE, label="c01")

        assert not fitness.accepted
        assert fitness.total == 0.0
        assert fitness.rejection_reason == "no change from the baseline"
        assert runner.calls == 0

    def test_a_failing_test_suite_is_fatal_regardless_of_everything_else(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path,
            pytest_runner=Recorder(failed()),
            repro=FakeRepro(ReproStatus.FIXED),
        )
        fitness = evaluator.evaluate(BEFORE, FIXED, label="c01")

        assert not fitness.accepted
        assert fitness.total == 0.0
        assert "hard gate" in fitness.rejection_reason
        assert fitness.quality.score == pytest.approx(1.0)

    def test_a_failing_test_suite_stops_the_benchmarks_running(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": passed("tblite", score=0.9)})
        evaluator = make_evaluator(
            tmp_path,
            pytest_runner=Recorder(failed()),
            benchmark_runner=benchmarks,
            benchmarks=["tblite"],
        )
        evaluator.evaluate(BEFORE, FIXED)
        assert benchmarks.calls == []

    def test_unavailable_pytest_is_noted_but_not_fatal(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path, pytest_runner=Recorder(unavailable("pytest", "no tests/"))
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert fitness.accepted
        assert any("hard gate did not actually verify" in n for n in fitness.notes)

    def test_strict_mode_rejects_an_unavailable_pytest(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path,
            pytest_runner=Recorder(unavailable("pytest", "no tests/")),
            strict=True,
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert not fitness.accepted
        assert fitness.total == 0.0


class TestBugFitness:
    def test_fixing_the_bug_scores_full_marks(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.FIXED))
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert fitness.components["bug_fix"] == 1.0
        assert fitness.total == pytest.approx(1.0)

    def test_not_fixing_the_bug_is_a_rejection_by_default(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.PRESENT))
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert not fitness.accepted
        assert "bug not fixed" in fitness.rejection_reason
        assert fitness.total == 0.0

    def test_require_bug_fix_off_scores_it_instead_of_rejecting(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path, repro=FakeRepro(ReproStatus.PRESENT), require_bug_fix=False
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert fitness.components["bug_fix"] == 0.0
        # quality 1.0 at weight 0.2, bug 0.0 at weight 0.5
        assert fitness.total == pytest.approx(0.2 / 0.7)

    def test_a_timed_out_reproduction_is_not_a_fix(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.ERROR))
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert not fitness.accepted

    def test_an_unavailable_reproduction_is_dropped_from_the_score(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.UNAVAILABLE))
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert "bug_fix" not in fitness.components


class TestBenchmarkFitness:
    def test_a_regression_rejects_the_candidate(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": failed("tblite", "regressed -8%")})
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite"]
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert not fitness.accepted
        assert "tblite" in fitness.rejection_reason
        assert fitness.total == 0.0

    def test_a_benchmark_score_enters_the_weighted_total(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": passed("tblite", score=0.5)})
        evaluator = make_evaluator(
            tmp_path,
            benchmark_runner=benchmarks,
            benchmarks=["tblite"],
            repro=FakeRepro(ReproStatus.FIXED),
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.components["benchmark"] == 0.5
        assert fitness.total == pytest.approx((1.0 * 0.5 + 0.5 * 0.3 + 1.0 * 0.2))

    def test_two_benchmarks_average(self, tmp_path):
        benchmarks = NamedBenchmarks(
            {
                "tblite": passed("tblite", score=0.6),
                "yc_bench": passed("yc_bench", score=1.0),
            }
        )
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite", "yc_bench"]
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert fitness.components["benchmark"] == pytest.approx(0.8)

    def test_an_unavailable_benchmark_is_excluded_not_scored_zero(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": unavailable("tblite")})
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite"]
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert "benchmark" not in fitness.components
        assert fitness.total == pytest.approx(1.0)

    def test_strict_mode_rejects_an_unavailable_benchmark(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": unavailable("tblite")})
        evaluator = make_evaluator(
            tmp_path,
            benchmark_runner=benchmarks,
            benchmarks=["tblite"],
            strict=True,
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert not fitness.accepted
        assert "tblite" in fitness.rejection_reason


class TestWeighting:
    def test_with_nothing_measurable_the_total_is_the_quality_score(self, tmp_path):
        evaluator = make_evaluator(tmp_path)
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert fitness.components == {"quality": fitness.quality.score}
        assert fitness.total == pytest.approx(fitness.quality.score)

    def test_quality_regressions_lower_the_score(self, tmp_path):
        sloppy = FIXED.replace(
            "    except OSError:\n        return []\n", "    except:\n        pass\n"
        )
        evaluator = make_evaluator(tmp_path, require_bug_fix=False)
        fitness = evaluator.evaluate(BEFORE, sloppy)
        assert fitness.accepted
        assert fitness.total < 1.0

    def test_custom_weights_are_respected(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path,
            repro=FakeRepro(ReproStatus.PRESENT),
            require_bug_fix=False,
            weights=FitnessWeights(bug_fix=0.9, benchmark=0.0, quality=0.1),
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert fitness.total == pytest.approx(0.1)
        assert fitness.weights_used["bug_fix"] == 0.9


class TestOnDiskGuard:
    def test_scoring_a_candidate_that_is_not_applied_is_refused(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text(BEFORE)
        evaluator = make_evaluator(tmp_path, target=target)
        with pytest.raises(FitnessError, match="does not contain the candidate"):
            evaluator.evaluate(BEFORE, FIXED)

    def test_scoring_an_applied_candidate_is_allowed(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text(FIXED)
        evaluator = make_evaluator(tmp_path, target=target)
        assert evaluator.evaluate(BEFORE, FIXED).accepted


class TestBaseline:
    def test_snapshot_reports_green_tests_and_a_reproducing_bug(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.PRESENT))
        baseline = evaluator.snapshot_baseline(BEFORE)

        assert isinstance(baseline, BaselineSnapshot)
        assert baseline.tests_green
        assert baseline.bug_reproduces

    def test_snapshot_flags_a_bug_that_does_not_reproduce(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.FIXED))
        baseline = evaluator.snapshot_baseline(BEFORE)
        assert not baseline.bug_reproduces

    def test_snapshot_flags_a_red_baseline(self, tmp_path):
        evaluator = make_evaluator(tmp_path, pytest_runner=Recorder(failed()))
        assert not evaluator.snapshot_baseline(BEFORE).tests_green

    def test_snapshot_records_benchmark_baselines_for_later_comparison(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": passed("tblite", score=0.77)})
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite"]
        )
        baseline = evaluator.snapshot_baseline(BEFORE)

        assert baseline.benchmark_baselines() == {"tblite": 0.77}
        assert evaluator.benchmark_baselines["tblite"] == 0.77

    def test_snapshot_is_json_serialisable(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.PRESENT))
        blob = json.loads(json.dumps(evaluator.snapshot_baseline(BEFORE).to_dict()))
        assert blob["tests_green"] is True
        assert blob["bug_reproduces"] is True


class TestFitnessRecord:
    def test_to_dict_is_json_serialisable(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.FIXED))
        blob = json.loads(json.dumps(evaluator.evaluate(BEFORE, FIXED, "c01").to_dict()))

        assert blob["label"] == "c01"
        assert blob["accepted"] is True
        assert blob["safety"]["passed"] is True
        assert blob["pytest"]["status"] == "passed"
        assert blob["repro"]["status"] == "fixed"

    def test_rejected_is_the_inverse_of_accepted(self, tmp_path):
        evaluator = make_evaluator(tmp_path)
        assert evaluator.evaluate(BEFORE, UNSAFE).rejected
        assert not evaluator.evaluate(BEFORE, FIXED).rejected


# ──────────────────────────────────────────────────────────────────────────
# Repeated reproduction runs
# ──────────────────────────────────────────────────────────────────────────


class TestReproTrials:
    def test_one_clean_run_is_fixed_but_not_certain(self):
        measured = trials(ReproStatus.FIXED)
        assert measured.fixed
        assert measured.fix_rate == 1.0
        # One clean run cannot rule out a repro that fails four times in five.
        assert measured.interval().low == pytest.approx(0.2065, abs=1e-3)
        assert measured.interval().high == 1.0

    def test_a_partial_fix_rate_is_not_a_fix(self):
        measured = trials(
            ReproStatus.FIXED,
            ReproStatus.PRESENT,
            ReproStatus.FIXED,
            ReproStatus.FIXED,
            ReproStatus.PRESENT,
        )
        assert measured.fixes == 3
        assert measured.fix_rate == pytest.approx(0.6)
        assert measured.flaky
        assert not measured.fixed
        assert measured.status is ReproStatus.PRESENT

    def test_more_clean_runs_narrow_the_interval(self):
        assert trials(*[ReproStatus.FIXED] * 20).interval().low > (
            trials(*[ReproStatus.FIXED] * 3).interval().low
        )

    def test_one_bad_run_outranks_any_number_of_clean_ones(self):
        measured = trials(*([ReproStatus.FIXED] * 9), ReproStatus.PRESENT)
        assert measured.status is ReproStatus.PRESENT
        assert not measured.fixed
        assert measured.fix_rate == pytest.approx(0.9)

    def test_an_errored_run_is_not_counted_as_a_measurement(self):
        measured = trials(ReproStatus.FIXED, ReproStatus.ERROR)
        assert measured.n == 2
        assert measured.measured_runs == 1
        assert measured.status is ReproStatus.ERROR
        assert not measured.fixed

    def test_no_runs_is_unavailable_not_fixed(self):
        empty = ReproTrials()
        assert empty.status is ReproStatus.UNAVAILABLE
        assert not empty.fixed
        assert not empty.measured
        assert empty.describe() == "no verdict from 0 run(s)"

    def test_the_representative_run_matches_the_aggregate_verdict(self):
        measured = trials(ReproStatus.FIXED, ReproStatus.FIXED, ReproStatus.PRESENT)
        assert measured.representative.status is ReproStatus.PRESENT

    def test_reproduced_is_true_when_any_run_showed_the_bug(self):
        assert trials(ReproStatus.FIXED, ReproStatus.PRESENT).reproduced
        assert not trials(ReproStatus.FIXED, ReproStatus.FIXED).reproduced

    def test_the_power_note_names_what_clean_runs_did_not_rule_out(self):
        note = trials(ReproStatus.FIXED, ReproStatus.FIXED).power_note
        assert "2/2 clean run(s)" in note
        assert "as low as 34.2%" in note

    def test_a_failing_set_of_runs_has_no_power_note(self):
        assert trials(ReproStatus.PRESENT).power_note is None

    def test_describe_calls_a_flake_a_flake(self):
        text = trials(ReproStatus.FIXED, ReproStatus.PRESENT).describe()
        assert "fixed 1/2 run(s)" in text
        assert "flaky, not a fix" in text

    def test_to_dict_is_json_serialisable(self):
        blob = json.loads(json.dumps(trials(ReproStatus.FIXED, ReproStatus.PRESENT).to_dict()))
        assert blob["runs"] == 2
        assert blob["fixes"] == 1
        assert blob["flaky"] is True
        assert blob["fix_rate_ci"]["low"] < 0.5 < blob["fix_rate_ci"]["high"]


class TestRunMany:
    def test_a_deterministic_script_runs_the_requested_number_of_times(self, tmp_path):
        repro = BugReproduction(
            script=write_script(tmp_path, "repro.py", "import sys\nsys.exit(0)\n")
        )
        measured = repro.run_many(tmp_path, sys.executable, runs=3)

        assert measured.n == 3
        assert measured.fixes == 3
        assert measured.fixed

    def test_a_flaky_script_is_caught_instead_of_believed(self, tmp_path):
        counter = tmp_path / "runs.txt"
        repro = BugReproduction(
            script=write_script(
                tmp_path,
                "repro.py",
                "import pathlib\nimport sys\n"
                f"counter = pathlib.Path({str(counter)!r})\n"
                "count = int(counter.read_text() or 0) + 1 if counter.exists() else 1\n"
                "counter.write_text(str(count))\n"
                "sys.exit(0 if count % 2 else 1)\n",
            )
        )
        measured = repro.run_many(tmp_path, sys.executable, runs=4)

        assert measured.n == 4
        assert measured.fixes == 2
        assert measured.flaky
        assert not measured.fixed

    def test_a_missing_script_stops_after_one_attempt(self, tmp_path):
        measured = BugReproduction(script=tmp_path / "ghost.py").run_many(
            tmp_path, sys.executable, runs=5
        )
        assert measured.n == 1
        assert measured.status is ReproStatus.UNAVAILABLE

    def test_a_run_count_below_one_is_refused(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "pass\n"))
        with pytest.raises(ValueError, match="at least 1"):
            repro.run_many(tmp_path, sys.executable, runs=0)

    def test_a_subclass_that_overrides_one_run_still_aggregates(self, tmp_path):
        repro = ScriptedRepro(ReproStatus.FIXED, ReproStatus.PRESENT)
        measured = repro.run_many(tmp_path, runs=2)
        assert measured.fixes == 1
        assert repro.calls == 2


class TestReproRunsInTheEvaluator:
    def test_a_flaky_fix_is_rejected_with_the_rate_in_the_reason(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path,
            repro=ScriptedRepro(
                ReproStatus.FIXED, ReproStatus.FIXED, ReproStatus.PRESENT
            ),
            repro_runs=3,
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert not fitness.accepted
        assert "bug not fixed" in fitness.rejection_reason
        assert "fixed 2/3 run(s)" in fitness.rejection_reason
        assert fitness.repro_trials.n == 3

    def test_a_flaky_fix_scores_its_rate_when_the_fix_is_not_required(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path,
            repro=ScriptedRepro(
                ReproStatus.FIXED, ReproStatus.FIXED, ReproStatus.PRESENT
            ),
            repro_runs=3,
            require_bug_fix=False,
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert fitness.components["bug_fix"] == pytest.approx(2 / 3)

    def test_a_single_run_keeps_the_old_binary_component(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.FIXED))
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.components["bug_fix"] == 1.0
        assert fitness.repro_trials.n == 1
        assert fitness.repro.status is ReproStatus.FIXED

    def test_the_baseline_snapshot_records_every_run(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path, repro=FakeRepro(ReproStatus.PRESENT), repro_runs=4
        )
        baseline = evaluator.snapshot_baseline(BEFORE)

        assert baseline.repro_trials.n == 4
        assert baseline.bug_reproduces
        assert json.loads(json.dumps(baseline.to_dict()))["repro_trials"]["runs"] == 4

    def test_a_run_count_below_one_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="at least 1"):
            make_evaluator(tmp_path, repro_runs=0)


# ──────────────────────────────────────────────────────────────────────────
# Per-test outcomes
# ──────────────────────────────────────────────────────────────────────────


class TestParsePytestOutcomes:
    def test_the_short_summary_form_is_read(self):
        outcomes = parse_pytest_outcomes(
            "=== short test summary info ===\n"
            "PASSED tests/test_a.py::test_one\n"
            "FAILED tests/test_a.py::test_two - assert 1 == 2\n"
        )
        assert outcomes == {
            "tests/test_a.py::test_one": True,
            "tests/test_a.py::test_two": False,
        }

    def test_the_verbose_form_is_read(self):
        outcomes = parse_pytest_outcomes(
            "tests/test_a.py::test_one PASSED [ 50%]\n"
            "tests/test_a.py::test_two FAILED [100%]\n"
        )
        assert outcomes == {
            "tests/test_a.py::test_one": True,
            "tests/test_a.py::test_two": False,
        }

    def test_an_error_counts_as_a_failure(self):
        assert parse_pytest_outcomes("ERROR tests/test_a.py::test_one - fixture") == {
            "tests/test_a.py::test_one": False
        }

    def test_an_expected_failure_is_not_a_failure(self):
        outcomes = parse_pytest_outcomes(
            "XFAIL tests/test_a.py::test_one - known\nXPASS tests/test_a.py::test_two\n"
        )
        assert outcomes == {
            "tests/test_a.py::test_one": True,
            "tests/test_a.py::test_two": True,
        }

    def test_skipped_tests_carry_no_outcome(self):
        assert parse_pytest_outcomes(
            "SKIPPED [1] tests/test_a.py:3: needs git\n"
            "tests/test_b.py::test_two SKIPPED [100%]\n"
        ) == {}

    def test_a_collection_error_without_a_node_id_is_ignored(self):
        assert parse_pytest_outcomes("ERROR tests/test_a.py\n") == {}

    def test_a_parametrised_id_with_a_space_survives(self):
        assert parse_pytest_outcomes("PASSED tests/test_a.py::test_x[a b]\n") == {
            "tests/test_a.py::test_x[a b]": True
        }

    def test_conflicting_verdicts_resolve_to_the_failure(self):
        outcomes = parse_pytest_outcomes(
            "tests/test_a.py::test_one PASSED [100%]\n"
            "FAILED tests/test_a.py::test_one - flaked on rerun\n"
        )
        assert outcomes == {"tests/test_a.py::test_one": False}

    def test_noise_is_ignored(self):
        assert parse_pytest_outcomes("collecting ...\n2 passed in 0.10s\n") == {}


class TestOutcomeReader:
    def test_a_structured_result_is_preferred(self):
        result = PerTestGateResult(
            "pytest", GateStatus.PASSED, "ok", outcomes={"a::b": True}
        )
        assert pytest_outcomes_from_result(result) == {"a::b": True}

    def test_a_plain_result_falls_back_to_its_captured_output(self):
        result = GateResult(
            "pytest", GateStatus.FAILED, "1 failed",
            details="FAILED tests/test_a.py::test_one - boom",
        )
        assert pytest_outcomes_from_result(result) == {
            "tests/test_a.py::test_one": False
        }

    def test_a_result_with_nothing_to_read_reports_nothing(self):
        assert pytest_outcomes_from_result(GateResult("pytest", GateStatus.PASSED, "ok")) == {}

    def test_the_serialised_form_stays_compact(self):
        result = PerTestGateResult(
            "pytest",
            GateStatus.FAILED,
            "1 failed",
            outcomes={f"tests/test_a.py::test_{i}": i != 3 for i in range(50)},
        )
        blob = json.loads(json.dumps(result.to_dict()))
        assert blob["tests_measured"] == 50
        assert blob["failing_tests"] == ["tests/test_a.py::test_3"]


def write_suite(root, body, name="test_suite.py"):
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / name).write_text(body)
    return root


class TestPerTestPytestRunner:
    def test_a_green_run_records_every_test(self, tmp_path):
        write_suite(
            tmp_path,
            "def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n",
        )
        result = PerTestPytestRunner()(tmp_path, python=sys.executable)

        assert result.status is GateStatus.PASSED
        assert result.outcomes == {
            "tests/test_suite.py::test_one": True,
            "tests/test_suite.py::test_two": True,
        }

    def test_a_red_run_names_the_failing_test_and_still_fails(self, tmp_path):
        write_suite(
            tmp_path,
            "def test_one():\n    assert True\n\n\ndef test_two():\n    assert False\n",
        )
        result = PerTestPytestRunner()(tmp_path, python=sys.executable)

        assert result.status is GateStatus.FAILED
        assert result.outcomes["tests/test_suite.py::test_two"] is False
        assert result.outcomes["tests/test_suite.py::test_one"] is True

    def test_a_repo_without_tests_is_unavailable(self, tmp_path):
        result = PerTestPytestRunner()(tmp_path, python=sys.executable)
        assert result.status is GateStatus.UNAVAILABLE
        assert result.outcomes == {}

    def test_a_missing_repo_is_unavailable(self, tmp_path):
        result = PerTestPytestRunner()(tmp_path / "ghost", python=sys.executable)
        assert result.status is GateStatus.UNAVAILABLE

    def test_a_subset_still_narrows_the_run(self, tmp_path):
        write_suite(tmp_path, "def test_one():\n    assert True\n", name="test_a.py")
        write_suite(tmp_path, "def test_two():\n    assert True\n", name="test_b.py")
        result = PerTestPytestRunner()(
            tmp_path, subset=["tests/test_b.py"], python=sys.executable
        )
        assert list(result.outcomes) == ["tests/test_b.py::test_two"]


class TestCompareTestSuites:
    def test_tests_are_paired_by_node_id_not_by_position(self):
        comparison = compare_test_suites(
            {"a::one": True, "a::two": True},
            {"a::two": True, "a::one": False},
        )
        assert comparison.n == 2
        assert comparison.newly_failing == ("a::one",)

    def test_tests_only_one_run_knows_about_are_left_unpaired(self):
        comparison = compare_test_suites(
            {"a::one": True, "a::gone": True},
            {"a::one": True, "a::new": False},
        )
        assert comparison.n == 1
        assert comparison.added == ("a::new",)
        assert comparison.removed == ("a::gone",)
        # The shared test did not move, but the suite is not the same suite.
        assert comparison.paired.discordant == 0
        assert comparison.coverage_changed
        assert not comparison.unchanged
        assert comparison.verdict == "coverage changed"

    def test_a_vanishing_test_is_not_an_identical_run(self):
        """Fifty tests that stopped being collected are not fifty passes."""
        baseline = {f"a::t{i}": True for i in range(100)}
        candidate = {f"a::t{i}": True for i in range(50)}
        comparison = compare_test_suites(baseline, candidate)
        assert len(comparison.removed) == 50
        assert not comparison.unchanged
        assert comparison.verdict == "coverage changed"

    def test_no_shared_test_is_no_comparison(self):
        assert compare_test_suites({"a::one": True}, {"b::two": True}) is None

    def test_an_empty_side_is_no_comparison(self):
        assert compare_test_suites({}, {"a::one": True}) is None

    def test_a_wholesale_regression_is_significant(self):
        baseline = {f"a::t{i}": True for i in range(20)}
        candidate = {name: i >= 10 for i, name in enumerate(baseline)}
        comparison = compare_test_suites(baseline, candidate)

        assert comparison.significant_regression
        assert comparison.verdict == "significant regression"
        assert len(comparison.newly_failing) == 10

    def test_one_flipped_test_out_of_many_is_not_significant(self):
        baseline = {f"a::t{i}": True for i in range(40)}
        candidate = dict(baseline)
        candidate["a::t0"] = False
        comparison = compare_test_suites(baseline, candidate)

        assert not comparison.significant_regression
        assert comparison.newly_failing == ("a::t0",)
        assert comparison.paired.p_worse == pytest.approx(0.5)
        assert comparison.verdict == "no significant change"

    def test_a_wholesale_repair_is_a_significant_improvement(self):
        baseline = {f"a::t{i}": i >= 10 for i in range(20)}
        candidate = {name: True for name in baseline}
        comparison = compare_test_suites(baseline, candidate)

        assert comparison.significant_improvement
        assert len(comparison.newly_passing) == 10

    def test_describe_and_to_dict_carry_the_evidence(self):
        comparison = compare_test_suites(
            {"a::one": True, "a::two": True}, {"a::one": True, "a::two": False}
        )
        assert "newly failing" in comparison.describe()
        blob = json.loads(json.dumps(comparison.to_dict()))
        assert blob["paired"]["n"] == 2
        assert blob["newly_failing"] == ["a::two"]


class OutcomeRunner:
    """A pytest runner that answers with prepared per-test outcomes."""

    def __init__(self, *outcomes, status=GateStatus.PASSED):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, repo, *args, **kwargs):
        outcomes = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        status = GateStatus.PASSED if all(outcomes.values()) else GateStatus.FAILED
        return PerTestGateResult(
            "pytest",
            status,
            "ok" if status is GateStatus.PASSED else "1 failed",
            outcomes=dict(outcomes),
        )


class TestSuiteComparisonInTheEvaluator:
    def test_a_candidate_is_compared_against_the_baseline_run(self, tmp_path):
        green = {"a::one": True, "a::two": True}
        evaluator = make_evaluator(tmp_path, pytest_runner=OutcomeRunner(green))
        evaluator.snapshot_baseline(BEFORE)
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.suite is not None
        assert fitness.suite.n == 2
        assert fitness.suite.unchanged
        assert any("suite vs baseline" in note for note in fitness.notes)

    def test_without_a_baseline_there_is_no_comparison(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path, pytest_runner=OutcomeRunner({"a::one": True})
        )
        assert evaluator.evaluate(BEFORE, FIXED).suite is None

    def test_a_runner_with_no_per_test_detail_reports_no_comparison(self, tmp_path):
        evaluator = make_evaluator(tmp_path, pytest_runner=Recorder(passed()))
        evaluator.snapshot_baseline(BEFORE)
        assert evaluator.evaluate(BEFORE, FIXED).suite is None

    def test_a_fresh_baseline_replaces_outcomes_handed_in_from_elsewhere(self, tmp_path):
        # Pairing against a baseline measured in some other session is worse
        # than not pairing at all, so a snapshot that measured nothing clears it.
        evaluator = make_evaluator(
            tmp_path,
            pytest_runner=Recorder(passed()),
            baseline_test_outcomes={"a::stale": True},
        )
        evaluator.snapshot_baseline(BEFORE)

        assert evaluator.baseline_test_outcomes == {}
        assert evaluator.evaluate(BEFORE, FIXED).suite is None

    def test_a_newly_failing_test_is_still_an_outright_rejection(self, tmp_path):
        baseline = {f"a::t{i}": True for i in range(40)}
        candidate = dict(baseline)
        candidate["a::t0"] = False
        evaluator = make_evaluator(
            tmp_path, pytest_runner=OutcomeRunner(baseline, candidate)
        )
        evaluator.snapshot_baseline(BEFORE)
        fitness = evaluator.evaluate(BEFORE, FIXED)

        # One failure in forty is nowhere near significant (p = 0.5), and it
        # rejects the candidate anyway. The statistics inform, they do not vote.
        assert not fitness.suite.significant_regression
        assert not fitness.accepted
        assert "hard gate" in fitness.rejection_reason
        assert fitness.total == 0.0
        assert fitness.suite.newly_failing == ("a::t0",)

    def test_a_significant_improvement_cannot_rescue_a_red_suite(self, tmp_path):
        baseline = {f"a::t{i}": i >= 10 for i in range(20)}
        candidate = {name: i != 19 for i, name in enumerate(baseline)}
        evaluator = make_evaluator(
            tmp_path, pytest_runner=OutcomeRunner(baseline, candidate)
        )
        evaluator.snapshot_baseline(BEFORE)
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.suite.significant_improvement
        assert not fitness.accepted
        assert fitness.total == 0.0

    def test_the_comparison_is_serialised_with_the_candidate(self, tmp_path):
        green = {"a::one": True}
        evaluator = make_evaluator(tmp_path, pytest_runner=OutcomeRunner(green))
        evaluator.snapshot_baseline(BEFORE)
        blob = json.loads(json.dumps(evaluator.evaluate(BEFORE, FIXED).to_dict()))

        assert blob["suite"]["verdict"] == "identical outcomes"
        assert blob["pytest"]["tests_measured"] == 1


# ──────────────────────────────────────────────────────────────────────────
# Evidence coverage
# ──────────────────────────────────────────────────────────────────────────


class TestEvidenceCoverage:
    def test_quality_alone_covers_a_fifth_of_the_intended_weight(self, tmp_path):
        fitness = make_evaluator(tmp_path).evaluate(BEFORE, FIXED)

        assert fitness.total == pytest.approx(1.0)
        assert fitness.evidence_coverage == pytest.approx(0.2)
        assert fitness.missing_evidence == ["bug_fix", "benchmark"]

    def test_a_reproduction_raises_the_coverage(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.FIXED))
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.evidence_coverage == pytest.approx(0.7)
        assert fitness.missing_evidence == ["benchmark"]

    def test_every_component_measured_is_full_coverage(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": passed("tblite", score=0.9)})
        evaluator = make_evaluator(
            tmp_path,
            benchmark_runner=benchmarks,
            benchmarks=["tblite"],
            repro=FakeRepro(ReproStatus.FIXED),
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.evidence_coverage == pytest.approx(1.0)
        assert fitness.missing_evidence == []

    def test_two_identical_scores_are_told_apart_by_their_coverage(self, tmp_path):
        thin = make_evaluator(tmp_path).evaluate(BEFORE, FIXED)
        thick = make_evaluator(
            tmp_path, repro=FakeRepro(ReproStatus.FIXED)
        ).evaluate(BEFORE, FIXED)

        assert thin.total == pytest.approx(thick.total)
        assert thin.evidence_coverage < thick.evidence_coverage

    def test_the_score_line_carries_the_coverage(self, tmp_path):
        line = make_evaluator(tmp_path).evaluate(BEFORE, FIXED).score_line()
        assert line == "1.000 (evidence 20%, no bug_fix or benchmark)"

    def test_a_zeroed_component_is_not_reported_as_missing(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path, weights=FitnessWeights(bug_fix=0.0, benchmark=0.0, quality=1.0)
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.missing_evidence == []
        assert fitness.evidence_coverage == pytest.approx(1.0)

    def test_a_benchmark_comparison_admits_it_has_no_sample_size(self, tmp_path):
        benchmarks = NamedBenchmarks(
            {"tblite": GateResult(
                "tblite", GateStatus.PASSED, "held", score=0.88, baseline=0.90
            )}
        )
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite"]
        )
        note = next(
            n for n in evaluator.evaluate(BEFORE, FIXED).notes if n.startswith("tblite")
        )
        assert "90.0% -> 88.0%" in note
        assert "no sample size and no significance" in note

    def test_a_benchmark_without_a_baseline_makes_no_such_claim(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": passed("tblite", score=0.88)})
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite"]
        )
        assert not any(
            n.startswith("tblite") for n in evaluator.evaluate(BEFORE, FIXED).notes
        )

    def test_coverage_is_serialised(self, tmp_path):
        blob = json.loads(
            json.dumps(make_evaluator(tmp_path).evaluate(BEFORE, FIXED).to_dict())
        )
        assert blob["evidence_coverage"] == 0.2
        assert blob["missing_evidence"] == ["bug_fix", "benchmark"]


# ──────────────────────────────────────────────────────────────────────────
# Ranking
# ──────────────────────────────────────────────────────────────────────────


def scored(label, total, coverage=1.0, repro=None):
    return CodeFitness(
        label=label,
        accepted=True,
        total=total,
        safety=None,
        quality=None,
        pytest_result=passed(),
        evidence_coverage=coverage,
        repro_trials=repro,
    )


class TestRankCandidates:
    def test_nothing_to_rank_is_no_ranking(self):
        assert rank_candidates([]) is None

    def test_a_sole_survivor_has_no_margin(self):
        ranking = rank_candidates([scored("c01", 0.8)])
        assert ranking.winner == "c01"
        assert ranking.margin is None
        assert ranking.separated
        assert "nothing to rank it against" in ranking.describe()

    def test_a_clear_win_is_reported_as_one(self):
        ranking = rank_candidates([scored("c01", 0.9), scored("c02", 0.5)])
        assert ranking.winner == "c01"
        assert ranking.runner_up == "c02"
        assert ranking.margin == pytest.approx(0.4)
        assert ranking.separated
        assert ranking.tied == ("c01",)
        assert "ahead of c02 by 0.400" in ranking.describe()

    def test_a_hairline_lead_is_called_arbitrary(self):
        ranking = rank_candidates([scored("c01", 0.851), scored("c02", 0.850)])
        assert ranking.within_noise
        assert not ranking.separated
        assert ranking.tied == ("c01", "c02")
        assert "the pick is arbitrary" in ranking.describe()
        assert "nothing separates c01, c02" in ranking.describe()

    def test_the_winner_matches_what_max_would_have_picked(self):
        entries = [scored("c01", 0.8), scored("c02", 0.8), scored("c03", 0.4)]
        assert rank_candidates(entries).winner == max(
            entries, key=lambda f: f.total
        ).label

    def test_a_custom_resolution_is_respected(self):
        entries = [scored("c01", 0.9), scored("c02", 0.85)]
        assert rank_candidates(entries, resolution=0.01).separated
        assert not rank_candidates(entries, resolution=0.10).separated

    def test_overlapping_fix_rates_are_called_out(self):
        ranking = rank_candidates(
            [
                scored("c01", 0.9, repro=trials(*[ReproStatus.FIXED] * 3)),
                scored(
                    "c02",
                    0.5,
                    repro=trials(ReproStatus.FIXED, ReproStatus.FIXED, ReproStatus.PRESENT),
                ),
            ]
        )
        assert ranking.separated
        assert ranking.fix_rate_inconclusive
        assert "overlapping intervals" in ranking.describe()

    def test_identical_fix_rates_are_not_reported_as_inconclusive(self):
        ranking = rank_candidates(
            [
                scored("c01", 0.9, repro=trials(ReproStatus.FIXED)),
                scored("c02", 0.5, repro=trials(ReproStatus.FIXED)),
            ]
        )
        assert not ranking.fix_rate_inconclusive

    def test_a_winner_measured_more_thinly_than_the_runner_up_is_flagged(self):
        ranking = rank_candidates(
            [scored("c01", 0.95, coverage=0.2), scored("c02", 0.70, coverage=1.0)]
        )
        assert ranking.thinner_evidence
        assert "rests on less evidence" in ranking.describe()

    def test_equal_coverage_is_not_flagged(self):
        ranking = rank_candidates(
            [scored("c01", 0.95, coverage=0.7), scored("c02", 0.70, coverage=0.7)]
        )
        assert not ranking.thinner_evidence

    def test_the_default_resolution_is_two_points(self):
        assert DEFAULT_RANKING_RESOLUTION == 0.02

    def test_to_dict_is_json_serialisable(self):
        blob = json.loads(
            json.dumps(
                rank_candidates([scored("c01", 0.9), scored("c02", 0.895)]).to_dict()
            )
        )
        assert blob["winner"] == "c01"
        assert blob["within_noise"] is True
        assert blob["margin"] == pytest.approx(0.005)
        assert isinstance(blob["summary"], str)

    def test_a_ranking_is_a_dataclass_with_the_expected_shape(self):
        ranking = rank_candidates([scored("c01", 0.9)])
        assert isinstance(ranking, CandidateRanking)
        assert ranking.considered == 1
