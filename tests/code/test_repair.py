"""Tests for the iterative test-feedback repair engine (no LLM)."""

from pathlib import Path

from evolution.code.repair import RepairEngine
from tests.code.conftest import BUGGY_CALC, FIXED_CALC, StagedRepo, VISIBLE_TEST


def _staged(tmp_path: Path) -> StagedRepo:
    repo = StagedRepo(tmp_path)
    repo.write("tools/calc.py", BUGGY_CALC)
    repo.write("tests/tools/test_calc.py", VISIBLE_TEST)
    return repo


def _scripted_proposer(*responses):
    """A proposer that returns each response in turn (None allowed)."""
    it = iter(responses)

    def _propose(module_path, current_source, failing_output):
        try:
            return next(it)
        except StopIteration:
            return None

    return _propose


class TestRepairEngine:
    def test_fixes_on_first_round(self, tmp_path):
        repo = _staged(tmp_path)
        engine = RepairEngine(_scripted_proposer(FIXED_CALC), max_rounds=5)
        result = engine.repair(repo, "tools/calc.py", "tests/tools/test_calc.py")
        assert result.fixed
        assert result.fixed_round == 1
        assert result.final_source == FIXED_CALC

    def test_climbs_over_a_failed_round(self, tmp_path):
        repo = _staged(tmp_path)
        # Round 1: a still-wrong fix (multiplies). Round 2: the correct fix.
        wrong = "def add(a, b):\n    return a * b\n"
        engine = RepairEngine(_scripted_proposer(wrong, FIXED_CALC), max_rounds=5)
        result = engine.repair(repo, "tools/calc.py", "tests/tools/test_calc.py")
        assert result.fixed
        assert result.fixed_round == 2

    def test_freeze_violation_is_rejected_before_test_then_recovers(self, tmp_path):
        repo = _staged(tmp_path)
        # Round 1: renames the public function (freeze violation) — rejected
        # before any test run. Round 2: the correct, surface-preserving fix.
        renamed = "def add_renamed(a, b):\n    return a + b\n"
        engine = RepairEngine(_scripted_proposer(renamed, FIXED_CALC), max_rounds=5)
        result = engine.repair(repo, "tools/calc.py", "tests/tools/test_calc.py")
        assert result.fixed
        assert result.fixed_round == 2
        assert result.rounds[0].freeze_violations  # round 1 flagged
        assert not result.rounds[0].test_passed     # and never tested

    def test_never_fixes_returns_unfixed(self, tmp_path):
        repo = _staged(tmp_path)
        wrong = "def add(a, b):\n    return a * b\n"
        engine = RepairEngine(_scripted_proposer(wrong, wrong, wrong), max_rounds=3)
        result = engine.repair(repo, "tools/calc.py", "tests/tools/test_calc.py")
        assert not result.fixed
        assert result.final_source is None
        assert len(result.rounds) == 3

    def test_unusable_proposer_output_is_handled(self, tmp_path):
        repo = _staged(tmp_path)
        engine = RepairEngine(_scripted_proposer(None, FIXED_CALC), max_rounds=3)
        result = engine.repair(repo, "tools/calc.py", "tests/tools/test_calc.py")
        assert result.fixed
        assert result.rounds[0].proposed is False
        assert result.fixed_round == 2
