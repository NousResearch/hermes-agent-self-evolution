"""Tests for the Phase 4 admission gate.

The load-bearing property is the visible/hidden split: an optimizer that can
see every check it must satisfy will learn to satisfy those checks
specifically, and the gate stops measuring anything. These tests pin that the
held-out checks gate admission while leaking nothing back to the mutator.
"""

from __future__ import annotations

import os
import stat
import textwrap
from pathlib import Path

import pytest

from evolution.code.admission import (
    AdmissionGate,
    AdmissionVerdict,
    CheckResult,
    CommandCheck,
    PytestCheck,
    RecordedCommandCheck,
    build_default_gate,
    materialize_candidate,
)


class StubCheck:
    """A check with a predetermined outcome."""

    def __init__(self, name: str, passed: bool, detail: str = "", record: list | None = None):
        self.name = name
        self._passed = passed
        self._detail = detail
        self._record = record

    def run(self, repo: Path) -> CheckResult:
        if self._record is not None:
            self._record.append(self.name)
        return CheckResult(name=self.name, passed=self._passed, detail=self._detail)


@pytest.fixture
def baseline_repo(tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "agent").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "agent" / "tool.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "tests" / "test_tool.py").write_text(
        textwrap.dedent(
            """
            import sys
            sys.path.insert(0, ".")
            from agent.tool import add

            def test_add():
                # 2+3 not 2+2: the latter equals 2*2, so a candidate that
                # swapped + for * would pass and the fixture would prove nothing.
                assert add(2, 3) == 5
            """
        ).strip()
        + "\n"
    )
    # Noise that must not be copied into a sandbox.
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / "agent" / "__pycache__").mkdir()
    (repo / "agent" / "__pycache__" / "tool.cpython-312.pyc").write_bytes(b"\x00")
    return repo


# ── sandboxing ──────────────────────────────────────────────────────────


class TestMaterializeCandidate:
    def test_baseline_is_copied_and_overlaid(self, baseline_repo, tmp_path):
        sandbox = materialize_candidate(
            baseline_repo,
            {"agent/tool.py": "def add(a, b):\n    return b + a\n"},
            tmp_path / "sandbox",
        )
        assert (sandbox / "agent" / "tool.py").read_text() == "def add(a, b):\n    return b + a\n"
        # Untouched files come along.
        assert (sandbox / "tests" / "test_tool.py").is_file()

    def test_the_real_repo_is_never_modified(self, baseline_repo, tmp_path):
        original = (baseline_repo / "agent" / "tool.py").read_text()
        materialize_candidate(baseline_repo, {"agent/tool.py": "BROKEN"}, tmp_path / "sandbox")
        assert (baseline_repo / "agent" / "tool.py").read_text() == original

    def test_vcs_and_caches_are_excluded(self, baseline_repo, tmp_path):
        sandbox = materialize_candidate(baseline_repo, {}, tmp_path / "sandbox")
        assert not (sandbox / ".git").exists()
        assert not (sandbox / "agent" / "__pycache__").exists()

    def test_a_path_escaping_the_sandbox_is_refused(self, baseline_repo, tmp_path):
        with pytest.raises(ValueError, match="escapes the sandbox"):
            materialize_candidate(
                baseline_repo, {"../../etc/passwd": "pwned"}, tmp_path / "sandbox"
            )

    def test_an_absolute_path_is_refused(self, baseline_repo, tmp_path):
        with pytest.raises(ValueError, match="escapes the sandbox"):
            materialize_candidate(baseline_repo, {"/tmp/evil.py": "pwned"}, tmp_path / "sandbox")

    def test_a_new_nested_file_is_created(self, baseline_repo, tmp_path):
        sandbox = materialize_candidate(
            baseline_repo, {"agent/sub/new.py": "x = 1\n"}, tmp_path / "sandbox"
        )
        assert (sandbox / "agent" / "sub" / "new.py").read_text() == "x = 1\n"

    def test_a_stale_sandbox_is_replaced(self, baseline_repo, tmp_path):
        dest = tmp_path / "sandbox"
        dest.mkdir()
        (dest / "leftover.txt").write_text("from a previous run")
        materialize_candidate(baseline_repo, {}, dest)
        assert not (dest / "leftover.txt").exists()


# ── the visible / hidden split ──────────────────────────────────────────


class TestAdmissionSplit:
    def test_all_passing_is_admitted(self, tmp_path):
        gate = AdmissionGate(visible=[StubCheck("v", True)], hidden=[StubCheck("h", True)])
        assert gate.admit(tmp_path).admitted is True

    def test_a_visible_failure_rejects(self, tmp_path):
        gate = AdmissionGate(visible=[StubCheck("v", False, "boom")], hidden=[StubCheck("h", True)])
        verdict = gate.admit(tmp_path)
        assert verdict.admitted is False
        assert "v" in verdict.reason

    def test_a_hidden_failure_rejects_even_when_everything_visible_passed(self, tmp_path):
        gate = AdmissionGate(visible=[StubCheck("v", True)], hidden=[StubCheck("h", False, "secret")])
        verdict = gate.admit(tmp_path)
        assert verdict.admitted is False
        assert "held-out" in verdict.reason

    def test_hidden_checks_are_not_run_when_visible_already_failed(self, tmp_path):
        ran: list[str] = []
        gate = AdmissionGate(
            visible=[StubCheck("v", False, record=ran)],
            hidden=[StubCheck("h", True, record=ran)],
        )
        gate.admit(tmp_path)
        assert ran == ["v"], "the expensive held-out suite should not be spent on a dead candidate"

    def test_visible_checks_short_circuit_on_first_failure(self, tmp_path):
        ran: list[str] = []
        gate = AdmissionGate(
            visible=[StubCheck("a", True, record=ran), StubCheck("b", False, record=ran),
                     StubCheck("c", True, record=ran)],
        )
        gate.admit(tmp_path)
        assert ran == ["a", "b"]

    def test_a_missing_repo_is_rejected_not_crashed(self, tmp_path):
        gate = AdmissionGate(visible=[StubCheck("v", True)])
        verdict = gate.admit(tmp_path / "nope")
        assert verdict.admitted is False
        assert "does not exist" in verdict.reason


class TestFeedbackLeakage:
    """The mutator must learn from visible failures and nothing else."""

    def test_visible_failure_detail_is_returned(self, tmp_path):
        gate = AdmissionGate(visible=[StubCheck("lint", False, "line 12: unused import")])
        feedback = gate.admit(tmp_path).feedback_for_mutator()
        assert "lint" in feedback
        assert "unused import" in feedback

    def test_hidden_failure_names_and_details_are_withheld(self, tmp_path):
        gate = AdmissionGate(
            visible=[StubCheck("v", True)],
            hidden=[StubCheck("secret-canary-check", False, "the canary value is 8f3a2b")],
        )
        feedback = gate.admit(tmp_path).feedback_for_mutator()

        assert "secret-canary-check" not in feedback
        assert "8f3a2b" not in feedback
        # It must still be told it was rejected, or it cannot improve at all.
        assert "held-out" in feedback
        assert "1" in feedback

    def test_the_operator_can_still_see_hidden_results(self, tmp_path):
        gate = AdmissionGate(visible=[StubCheck("v", True)], hidden=[StubCheck("h", False, "why")])
        verdict = gate.admit(tmp_path)
        # Withheld from the mutator, not from the human reading the report.
        assert verdict.hidden[0].name == "h"
        assert verdict.hidden[0].detail == "why"

    def test_clean_feedback_when_everything_passes(self, tmp_path):
        gate = AdmissionGate(visible=[StubCheck("v", True)], hidden=[StubCheck("h", True)])
        assert gate.admit(tmp_path).feedback_for_mutator() == "All checks passed."


# ── real checks ─────────────────────────────────────────────────────────


class TestPytestCheck:
    def test_a_passing_suite_passes(self, baseline_repo, tmp_path):
        sandbox = materialize_candidate(baseline_repo, {}, tmp_path / "s")
        assert PytestCheck(timeout_s=120).run(sandbox).passed is True

    def test_a_broken_candidate_fails(self, baseline_repo, tmp_path):
        sandbox = materialize_candidate(
            baseline_repo, {"agent/tool.py": "def add(a, b):\n    return a * b\n"}, tmp_path / "s"
        )
        assert PytestCheck(timeout_s=120).run(sandbox).passed is False

    def test_deleting_the_tests_does_not_count_as_passing(self, baseline_repo, tmp_path):
        """pytest exits 5 on no-tests-collected; that must not read as success."""
        sandbox = materialize_candidate(baseline_repo, {}, tmp_path / "s")
        (sandbox / "tests" / "test_tool.py").unlink()

        result = PytestCheck(timeout_s=120).run(sandbox)
        assert result.passed is False
        assert "no tests were collected" in result.detail


class TestCommandCheck:
    def test_exit_zero_passes(self, tmp_path):
        assert CommandCheck("ok", ["python", "-c", "pass"]).run(tmp_path).passed is True

    def test_nonzero_fails_and_captures_output(self, tmp_path):
        result = CommandCheck("bad", ["python", "-c", "import sys; print('nope'); sys.exit(3)"]).run(tmp_path)
        assert result.passed is False
        assert result.exit_code == 3
        assert "nope" in result.detail

    def test_a_missing_binary_is_reported_not_raised(self, tmp_path):
        result = CommandCheck("ghost", ["definitely-not-a-real-binary-xyz"]).run(tmp_path)
        assert result.passed is False
        assert "could not run" in result.detail

    def test_a_hanging_command_times_out(self, tmp_path):
        result = CommandCheck("hang", ["python", "-c", "import time; time.sleep(30)"], timeout_s=1).run(tmp_path)
        assert result.passed is False
        assert "timed out" in result.detail


class TestSandboxEnvironment:
    def test_credentials_are_stripped_from_check_runs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")
        monkeypatch.setenv("HARMLESS_VAR", "fine")

        script = "import os; print(os.environ.get('ANTHROPIC_API_KEY','ABSENT')); " \
                 "print(os.environ.get('HARMLESS_VAR','ABSENT'))"
        result = CommandCheck("env", ["python", "-c", script + "; import sys; sys.exit(1)"]).run(tmp_path)

        assert "sk-ant-should-not-leak" not in result.detail
        assert "ABSENT" in result.detail
        assert "fine" in result.detail

    def test_the_sandbox_marker_is_set(self, tmp_path):
        result = CommandCheck(
            "marker",
            ["python", "-c", "import os,sys; print(os.environ.get('EVOLUTION_SANDBOX')); sys.exit(1)"],
        ).run(tmp_path)
        assert "1" in result.detail


class TestRecordedCommandCheck:
    def test_matching_the_recorded_exit_code_passes(self, tmp_path):
        assert RecordedCommandCheck("rec", "true").run(tmp_path).passed is True

    def test_diverging_from_the_recorded_outcome_fails(self, tmp_path):
        result = RecordedCommandCheck("rec", "false", expected_exit=0).run(tmp_path)
        assert result.passed is False
        assert "recorded 0" in result.detail


class TestDefaultGate:
    def test_the_full_suite_is_always_held_out(self, tmp_path):
        gate = build_default_gate(tmp_path, targeted_tests=["tests/test_tool.py"])
        assert [c.name for c in gate.visible] == ["targeted-tests"]
        assert "full-suite" in [c.name for c in gate.hidden]

    def test_recorded_checks_join_the_hidden_set(self, tmp_path):
        recorded = [RecordedCommandCheck("recorded:test:pytest", "pytest -q")]
        gate = build_default_gate(tmp_path, targeted_tests=[], recorded=recorded)
        assert any(c.name.startswith("recorded:") for c in gate.hidden)

    def test_a_gate_with_no_targeted_tests_still_gates(self, tmp_path):
        gate = build_default_gate(tmp_path)
        assert gate.visible == []
        assert gate.hidden, "the full suite must still be enforced"
