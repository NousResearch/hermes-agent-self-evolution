"""Tests for the validation gate ladder.

Offline throughout. The pytest gate is exercised against throwaway repos
written into tmp_path, so it runs a real subprocess but never touches the
network or a model.
"""

import json
import sys

import pytest

from evolution.core.gates import (
    KNOWN_BENCHMARKS,
    GateChain,
    GateResult,
    GateStatus,
    find_benchmark,
    run_benchmark_gate,
    run_pytest_gate,
)
from evolution.core.gates import _parse_benchmark_score as parse_score


@pytest.fixture
def passing_repo(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    return tmp_path


@pytest.fixture
def failing_repo(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_bad.py").write_text(
        "def test_bad():\n    assert 1 == 2, 'deliberate'\n"
    )
    return tmp_path


class TestPytestGate:
    def test_clean_suite_passes(self, passing_repo):
        result = run_pytest_gate(passing_repo, python=sys.executable)
        assert result.status is GateStatus.PASSED
        assert result.passed
        assert not result.blocking
        assert "1 passed" in result.message

    def test_failing_suite_fails(self, failing_repo):
        result = run_pytest_gate(failing_repo, python=sys.executable)
        assert result.status is GateStatus.FAILED
        assert result.blocking
        assert "deliberate" in result.details or "1 failed" in result.message

    def test_missing_tests_dir_is_unavailable_not_pass(self, tmp_path):
        result = run_pytest_gate(tmp_path, python=sys.executable)
        assert result.status is GateStatus.UNAVAILABLE
        assert not result.passed

    def test_missing_repo_is_unavailable(self, tmp_path):
        result = run_pytest_gate(tmp_path / "nope", python=sys.executable)
        assert result.status is GateStatus.UNAVAILABLE

    def test_subset_narrows_the_run(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
        (tmp_path / "tests" / "test_b.py").write_text("def test_b():\n    assert False\n")

        narrowed = run_pytest_gate(
            tmp_path, subset=["tests/test_a.py"], python=sys.executable
        )
        assert narrowed.status is GateStatus.PASSED

        whole = run_pytest_gate(tmp_path, python=sys.executable)
        assert whole.status is GateStatus.FAILED

    def test_timeout_is_a_failure_not_a_pass(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_slow.py").write_text(
            "import time\ndef test_slow():\n    time.sleep(30)\n"
        )
        result = run_pytest_gate(tmp_path, timeout=1, python=sys.executable)
        assert result.status is GateStatus.FAILED
        assert "timed out" in result.message


class TestBenchmarkDiscovery:
    def test_absent_benchmark_resolves_to_none(self, tmp_path):
        assert find_benchmark(tmp_path, "tblite") is None

    def test_unknown_benchmark_name_is_none(self, tmp_path):
        assert find_benchmark(tmp_path, "not_a_benchmark") is None

    def test_present_benchmark_is_found(self, tmp_path):
        target = tmp_path / "environments" / "benchmarks" / "tblite"
        target.mkdir(parents=True)
        assert find_benchmark(tmp_path, "tblite") == target

    def test_env_override_wins(self, tmp_path, monkeypatch):
        elsewhere = tmp_path / "external_tblite"
        elsewhere.mkdir()
        monkeypatch.setenv("HERMES_BENCH_TBLITE", str(elsewhere))
        assert find_benchmark(tmp_path, "tblite") == elsewhere

    def test_env_override_pointing_nowhere_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_BENCH_TBLITE", str(tmp_path / "ghost"))
        assert find_benchmark(tmp_path, "tblite") is None

    def test_plan_benchmarks_are_all_known(self):
        assert {"tblite", "terminalbench2", "yc_bench"} <= set(KNOWN_BENCHMARKS)


class TestBenchmarkGate:
    def test_absent_benchmark_is_unavailable_not_passed(self, tmp_path):
        result = run_benchmark_gate(tmp_path, "tblite")
        assert result.status is GateStatus.UNAVAILABLE
        assert not result.passed
        assert "HERMES_BENCH_TBLITE" in result.message

    def test_absent_benchmark_never_reports_a_score(self, tmp_path):
        assert run_benchmark_gate(tmp_path, "tblite").score is None


class TestScoreParsing:
    @pytest.mark.parametrize(
        "stdout,expected",
        [
            ('{"score": 0.83}', 0.83),
            ('{"pass_rate": 0.5}', 0.5),
            ('{"accuracy": 1.0}', 1.0),
            ("running...\nscore: 82/100", 0.82),
            ("final: 76.5%", 0.765),
            ("noise\n{\"score\": 0.4}", 0.4),
            ("passed 82 / 100", 0.82),
            ("Accuracy: 76.5 %", 0.765),
        ],
    )
    def test_parses_known_shapes(self, stdout, expected):
        assert parse_score(stdout) == pytest.approx(expected)

    def test_unparseable_returns_none(self):
        assert parse_score("no score anywhere") is None

    def test_zero_denominator_does_not_crash(self):
        assert parse_score("score: 0/0") is None

    @pytest.mark.parametrize(
        "stdout",
        [
            "tests/test_tblite.py ......                              [100%]",
            "collecting tasks\nprocessing 12/20",
            "elapsed 45/60 seconds",
            "82/100",
        ],
    )
    def test_a_progress_line_is_not_a_score(self, stdout):
        """An unlabelled number is a number, not a result the benchmark claimed."""
        assert parse_score(stdout) is None

    def test_a_trailing_progress_line_cannot_outrank_a_real_result(self):
        """The scan runs backwards, so the fabricating line is seen first."""
        stdout = '{"score": 0.42}\nrunning suite ... [100%]'
        assert parse_score(stdout) == pytest.approx(0.42)


class TestGateChain:
    def _r(self, name, status):
        return GateResult(name, status, "msg")

    def test_stops_at_first_failure(self):
        chain = GateChain().run(
            lambda: self._r("a", GateStatus.PASSED),
            lambda: self._r("b", GateStatus.FAILED),
            lambda: pytest.fail("third gate must not run"),
        )
        assert not chain.passed
        assert [r.name for r in chain.results] == ["a", "b"]

    def test_all_passing_passes(self):
        chain = GateChain().run(
            lambda: self._r("a", GateStatus.PASSED),
            lambda: self._r("b", GateStatus.PASSED),
        )
        assert chain.passed
        assert chain.blockers == []

    def test_permissive_tolerates_unavailable(self):
        chain = GateChain(strict=False).run(
            lambda: self._r("a", GateStatus.PASSED),
            lambda: self._r("tblite", GateStatus.UNAVAILABLE),
        )
        assert chain.passed
        assert len(chain.results) == 2

    def test_strict_blocks_on_unavailable(self):
        chain = GateChain(strict=True).run(
            lambda: self._r("a", GateStatus.PASSED),
            lambda: self._r("tblite", GateStatus.UNAVAILABLE),
            lambda: pytest.fail("must not run after a strict blocker"),
        )
        assert not chain.passed
        assert [b.name for b in chain.blockers] == ["tblite"]

    def test_summary_marks_each_status(self):
        chain = GateChain().run(
            lambda: self._r("a", GateStatus.PASSED),
            lambda: self._r("b", GateStatus.UNAVAILABLE),
        )
        summary = chain.summary()
        assert "✓ a" in summary
        assert "○ b" in summary

    def test_to_dict_is_json_serialisable(self):
        chain = GateChain().run(lambda: self._r("a", GateStatus.PASSED))
        blob = json.loads(json.dumps(chain.to_dict()))
        assert blob["passed"] is True
        assert blob["results"][0]["name"] == "a"
        assert blob["results"][0]["status"] == "passed"


class TestEmptySelectionIsNotAFailure:
    """pytest exit code 5 means "nothing collected", not "tests failed".

    Treating it as FAILED turned a too-narrow -k expression into a red suite
    that blocked every candidate, while telling the operator their tests were
    broken. An empty selection verifies nothing, so it is UNAVAILABLE.
    """

    def test_a_selection_matching_nothing_is_unavailable(self, passing_repo):
        result = run_pytest_gate(
            passing_repo, subset=["tests/", "-k", "matches_nothing_at_all"],
            python=sys.executable,
        )
        assert result.status is GateStatus.UNAVAILABLE
        assert not result.passed
        assert not result.blocking
        assert "no tests matched" in result.message

    def test_permissive_tolerates_it_but_strict_blocks_it(self, passing_repo):
        result = run_pytest_gate(
            passing_repo, subset=["tests/", "-k", "nope"], python=sys.executable
        )
        assert GateChain(strict=False).run(lambda: result).passed
        assert not GateChain(strict=True).run(lambda: result).passed

    def test_a_real_failure_is_still_a_failure(self, failing_repo):
        result = run_pytest_gate(failing_repo, python=sys.executable)
        assert result.status is GateStatus.FAILED
