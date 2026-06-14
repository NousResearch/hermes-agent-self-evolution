"""Tests for the code-evolution deploy gate.

The decisive case is the held-out split: a fix that games the visible test must
be rejected. The gate runs against a real git repo (it derives the pre-repair
base from HEAD) with self-contained tools (real pytest, no venv).
"""

from pathlib import Path

from evolution.code.gate import run_code_gate
from evolution.code.repair import RepairResult, RoundRecord
from tests.code.conftest import (
    BUGGY_CALC,
    FIXED_CALC,
    HOLDOUT_TEST,
    VISIBLE_TEST,
    StagedRepo,
)

TOOL = "tools/calc.py"
VIS = "tests/tools/test_calc_visible.py"
HOLD = "tests/tools/test_calc_holdout.py"


def _repo(tmp_path: Path) -> StagedRepo:
    repo = StagedRepo(tmp_path)
    repo.write(TOOL, BUGGY_CALC)
    repo.write(VIS, VISIBLE_TEST)
    repo.write(HOLD, HOLDOUT_TEST)
    repo.git_init_commit()
    return repo


def _fixed_result(src: str) -> RepairResult:
    return RepairResult(fixed=True, fixed_round=1, final_source=src,
                        rounds=[RoundRecord(round=1, proposed=True, test_passed=True)])


def _gate(repo: StagedRepo, result: RepairResult):
    return run_code_gate(
        repo, tool_relpath=TOOL, visible_test_relpath=VIS, holdout_test_relpath=HOLD,
        repair_result=result, floor_paths=("tests/tools",),
    )


class TestCodeGate:
    def test_clean_fix_deploys(self, tmp_path):
        repo = _repo(tmp_path)
        repo.write_tool(TOOL, FIXED_CALC)  # repair engine wrote this
        res = _gate(repo, _fixed_result(FIXED_CALC))
        assert res.deploy, res.reason
        assert res.decision["decision"] == "deploy"
        assert res.decision["guards"]["holdout"]["passed"]
        assert res.decision["guards"]["floor"]["new_failures"] == []

    def test_floor_catches_introduced_regression(self, tmp_path):
        # A committed test encodes the OLD (buggy) behavior add(1,1)==0; it
        # passes on the buggy base but fails once add is correctly fixed. The
        # baseline-diff floor must flag it as an introduced regression even
        # though the visible+held-out splits pass.
        repo = StagedRepo(tmp_path)
        repo.write(TOOL, BUGGY_CALC)
        repo.write(VIS, VISIBLE_TEST)
        repo.write(HOLD, HOLDOUT_TEST)
        repo.write("tests/tools/test_calc_legacy.py",
                   "from tools.calc import add\n\n\ndef test_legacy():\n    assert add(1, 1) == 0\n")
        repo.git_init_commit()
        repo.write_tool(TOOL, FIXED_CALC)
        res = _gate(repo, _fixed_result(FIXED_CALC))
        assert not res.deploy
        assert "introduced" in res.reason
        assert any("test_legacy" in f for f in res.decision["guards"]["floor"]["new_failures"])

    def test_gaming_fix_rejected_by_holdout(self, tmp_path):
        # Passes the visible test (add(2,3)==5) by hard-coding it, but fails the
        # held-out test (add(10,20)==30). The anti-gaming core.
        repo = _repo(tmp_path)
        gaming = "def add(a, b):\n    return 5\n"
        repo.write_tool(TOOL, gaming)
        res = _gate(repo, _fixed_result(gaming))
        assert not res.deploy
        assert "held-out" in res.reason
        assert res.decision["guards"]["holdout"]["passed"] is False

    def test_signature_drift_rejected_by_freeze(self, tmp_path):
        # Adds a parameter (signature drift) while still passing the visible test.
        repo = _repo(tmp_path)
        drifted = "def add(a, b, c=0):\n    return a + b\n"
        repo.write_tool(TOOL, drifted)
        res = _gate(repo, _fixed_result(drifted))
        assert not res.deploy
        assert "freeze" in res.reason or "signature" in res.reason

    def test_touching_a_test_file_rejected_by_file_scope(self, tmp_path):
        repo = _repo(tmp_path)
        repo.write_tool(TOOL, FIXED_CALC)
        # Simulate the loop also having edited a test file.
        repo.write(HOLD, HOLDOUT_TEST + "\n# tampered\n")
        res = _gate(repo, _fixed_result(FIXED_CALC))
        assert not res.deploy
        assert "test file" in res.reason

    def test_unfixed_repair_rejected(self, tmp_path):
        repo = _repo(tmp_path)
        res = _gate(repo, RepairResult(fixed=False, fixed_round=None,
                                       final_source=None, rounds=[]))
        assert not res.deploy
        assert "did not produce a fix" in res.reason

    def test_holdout_equal_to_visible_rejected(self, tmp_path):
        # A held-out split equal to the visible split is a tautology — rejected
        # before any test runs (the anti-gaming check would prove nothing).
        repo = _repo(tmp_path)
        repo.write_tool(TOOL, FIXED_CALC)
        res = run_code_gate(
            repo, tool_relpath=TOOL, visible_test_relpath=VIS,
            holdout_test_relpath=VIS, repair_result=_fixed_result(FIXED_CALC),
            floor_paths=("tests/tools",),
        )
        assert not res.deploy
        assert "no anti-gaming signal" in res.reason

    def test_holdout_collecting_no_tests_rejected(self, tmp_path):
        # A held-out path that collects zero tests (pytest exit 5) gives no
        # anti-gaming signal and must not be allowed to look like a pass. The
        # empty file is part of the committed baseline (an untracked one would
        # be caught earlier by the file-scope guard).
        repo = StagedRepo(tmp_path)
        repo.write(TOOL, BUGGY_CALC)
        repo.write(VIS, VISIBLE_TEST)
        empty = "tests/tools/test_empty.py"
        repo.write(empty, "# no test functions here\n")
        repo.git_init_commit()
        repo.write_tool(TOOL, FIXED_CALC)
        res = run_code_gate(
            repo, tool_relpath=TOOL, visible_test_relpath=VIS,
            holdout_test_relpath=empty, repair_result=_fixed_result(FIXED_CALC),
            floor_paths=("tests/tools",),
        )
        assert not res.deploy
        assert "collected no tests" in res.reason


class TestIsTestPath:
    def test_detects_test_artifacts(self):
        from evolution.code.gate import _is_test_path

        assert _is_test_path("tests/tools/test_foo.py")
        assert _is_test_path("tools/conftest.py")
        assert _is_test_path("tools/foo_test.py")
        assert _is_test_path("pkg/tests/helper.py")

    def test_excludes_real_source_even_with_test_substring(self):
        from evolution.code.gate import _is_test_path

        assert not _is_test_path("tools/fuzzy_match.py")
        assert not _is_test_path("tools/attestation.py")  # 'test' is a substring, not a marker
        assert not _is_test_path("pkg/testing/runner.py")  # 'testing/' dir, not 'tests/'
