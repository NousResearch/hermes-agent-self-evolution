"""Tests for the oracle-based measurement gate (run_code_oracle_gate).

The campaign worktrees at fix_sha (fix-commit test present) with the buggy parent
tool written in; the gate verifies a repair against the upstream-fix oracle. These
tests use a two-function tool so a repair that fixes the bug but breaks a sibling
behavior (a regression the oracle preserves) is demonstrable — that is what the
oracle test-match catches. (It does NOT catch pure input-hardcoding of the bug
tests; that needs the fuzzed differential, deferred.)
"""

from pathlib import Path

from evolution.code.gate import run_code_oracle_gate
from evolution.code.repair import RepairResult, RoundRecord
from tests.code.conftest import StagedRepo

TOOL = "tools/calc.py"
TEST = "tests/tools/test_calc.py"
BUG_TESTS = (f"{TEST}::test_add",)

BUGGY = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
FIXED = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
# fixes the bug but breaks the sibling mul (a regression the oracle preserves)
REGRESSION = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a + b\n"
FULL_TEST = (
    "from tools.calc import add, mul\n\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n\n"
    "def test_mul():\n    assert mul(3, 4) == 12\n"
)


def _repo(tmp_path: Path) -> StagedRepo:
    repo = StagedRepo(tmp_path)
    repo.write(TOOL, BUGGY)
    repo.write(TEST, FULL_TEST)
    repo.git_init_commit()  # committed buggy state == the worktree base
    return repo


def _fixed(src: str) -> RepairResult:
    return RepairResult(fixed=True, fixed_round=2, final_source=src,
                        rounds=[RoundRecord(round=1, proposed=True),
                                RoundRecord(round=2, proposed=True, test_passed=True)])


def _gate(repo: StagedRepo, result: RepairResult, floor_paths=None):
    return run_code_oracle_gate(
        repo, tool_relpath=TOOL, test_relpath=TEST, bug_tests=BUG_TESTS,
        oracle_failures=frozenset(), base_src=BUGGY, repair_result=result,
        floor_paths=floor_paths)


class TestOracleGate:
    def test_correct_fix_is_correct(self, tmp_path):
        repo = _repo(tmp_path)
        repo.write_tool(TOOL, FIXED)
        res = _gate(repo, _fixed(FIXED))
        assert res.deploy, res.reason
        assert res.decision["decision"] == "correct"
        assert res.decision["guards"]["bug_tests_passed"]
        assert res.decision["guards"]["oracle_match"]["new_vs_oracle"] == []
        assert res.decision["guards"]["floor"] is None  # broad floor off by default

    def test_optional_broad_floor_runs_when_requested(self, tmp_path):
        repo = _repo(tmp_path)
        repo.write_tool(TOOL, FIXED)
        res = _gate(repo, _fixed(FIXED), floor_paths=("tests/tools",))
        assert res.deploy, res.reason
        assert res.decision["guards"]["floor"]["new_failures"] == []

    def test_regression_against_oracle_rejected(self, tmp_path):
        repo = _repo(tmp_path)
        repo.write_tool(TOOL, REGRESSION)
        res = _gate(repo, _fixed(REGRESSION))
        assert not res.deploy
        assert "upstream fix" in res.reason
        assert any("test_mul" in t for t in res.decision["guards"]["oracle_match"]["new_vs_oracle"])

    def test_repair_not_passing_bug_tests_rejected(self, tmp_path):
        repo = _repo(tmp_path)
        still_buggy = "def add(a, b):\n    return a * b\n\n\ndef mul(a, b):\n    return a * b\n"
        repo.write_tool(TOOL, still_buggy)
        res = _gate(repo, _fixed(still_buggy))
        assert not res.deploy
        assert "bug tests" in res.reason

    def test_signature_drift_rejected_by_freeze(self, tmp_path):
        repo = _repo(tmp_path)
        drifted = "def add(a, b, c=0):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
        repo.write_tool(TOOL, drifted)
        res = _gate(repo, _fixed(drifted))
        assert not res.deploy
        assert "freeze" in res.reason or "signature" in res.reason

    def test_unfixed_repair_rejected(self, tmp_path):
        repo = _repo(tmp_path)
        res = _gate(repo, RepairResult(fixed=False, fixed_round=None,
                                       final_source=None, rounds=[]))
        assert not res.deploy
        assert "did not produce a fix" in res.reason
