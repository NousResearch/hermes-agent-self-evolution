"""Tests for the Phase 4 composite fitness scoring."""

from evolution.code.fitness import compute_fitness
from evolution.code.sandbox import SandboxResult
from evolution.code.signature_freeze import FreezeReport


def _ok_result(**overrides) -> SandboxResult:
    defaults = dict(
        command=["pytest"], returncode=0, stdout="", stderr="",
        duration_seconds=1.0, timed_out=False,
    )
    defaults.update(overrides)
    return SandboxResult(**defaults)


def _fail_result() -> SandboxResult:
    return SandboxResult(
        command=["pytest"], returncode=1, stdout="1 failed",
        stderr="", duration_seconds=1.0, timed_out=False,
    )


def _ok_freeze() -> FreezeReport:
    return FreezeReport(
        signature_violations=[], registry_violations=[],
        error_handling_baseline=2, error_handling_candidate=2,
    )


class TestComputeFitness:
    def test_all_green_no_bug_repro(self):
        fitness = compute_fitness(
            pytest_candidate=_ok_result(),
            ruff_candidate=_ok_result(),
            freeze=_ok_freeze(),
        )
        assert fitness.score == 0.5
        assert fitness.pytest_passed
        assert fitness.freeze_passed

    def test_pytest_fail_yields_zero(self):
        fitness = compute_fitness(
            pytest_candidate=_fail_result(),
            ruff_candidate=_ok_result(),
            freeze=_ok_freeze(),
        )
        assert fitness.score == 0.0
        assert not fitness.pytest_passed

    def test_freeze_fail_yields_zero(self):
        bad_freeze = FreezeReport(
            signature_violations=["sig changed"],
            registry_violations=[],
            error_handling_baseline=2, error_handling_candidate=2,
        )
        fitness = compute_fitness(
            pytest_candidate=_ok_result(),
            ruff_candidate=_ok_result(),
            freeze=bad_freeze,
        )
        assert fitness.score == 0.0
        assert not fitness.freeze_passed

    def test_bug_repro_transitioned_scores_1(self):
        fitness = compute_fitness(
            pytest_candidate=_ok_result(),
            ruff_candidate=_ok_result(),
            freeze=_ok_freeze(),
            bug_repro_baseline=_fail_result(),
            bug_repro_candidate=_ok_result(),
        )
        assert fitness.score == 1.0
        assert fitness.bug_repro_transitioned

    def test_bug_repro_not_transitioned(self):
        fitness = compute_fitness(
            pytest_candidate=_ok_result(),
            ruff_candidate=_ok_result(),
            freeze=_ok_freeze(),
            bug_repro_baseline=_fail_result(),
            bug_repro_candidate=_fail_result(),
        )
        assert fitness.score == 0.25
        assert not fitness.bug_repro_transitioned

    def test_ruff_dirty_penalty(self):
        ruff_dirty = _ok_result(returncode=1)
        fitness = compute_fitness(
            pytest_candidate=_ok_result(),
            ruff_candidate=ruff_dirty,
            freeze=_ok_freeze(),
        )
        assert fitness.score < 0.5
        assert not fitness.ruff_clean
