"""Tests for the Phase 4 gate evaluator."""

from pathlib import Path

from evolution.code.fitness import CompositeFitness
from evolution.code.signature_freeze import FreezeReport
from evolution.core.config import EvolutionConfig
from evolution.core.phase_gate import evaluate_phase4_gate
from evolution.core.reproducibility import build_code_reproducibility_manifest


def _make_manifest(tmp_path: Path):
    return build_code_reproducibility_manifest(
        tool_name="file_tools",
        tool_source_path=Path("/fake/tools/file_tools.py"),
        baseline_source="def x(): pass",
        config=EvolutionConfig(phase0_enforce=False),
        best_fitness_score=1.0,
        best_iteration=0,
        iterations=3,
        engine="internal",
        evolution_repo_path=tmp_path,
    )


def _ok_fitness() -> CompositeFitness:
    return CompositeFitness(
        score=1.0, pytest_passed=True, pytest_duration_seconds=1.0,
        ruff_clean=True, freeze_passed=True,
        bug_repro_transitioned=True, bug_repro_evaluated=True,
        notes=[],
    )


def _ok_freeze() -> FreezeReport:
    return FreezeReport(
        signature_violations=[], registry_violations=[],
        error_handling_baseline=2, error_handling_candidate=2,
    )


class TestEvaluatePhase4Gate:
    def test_passes_when_all_green(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        result = evaluate_phase4_gate(
            fitness=_ok_fitness(),
            freeze=_ok_freeze(),
            config=EvolutionConfig(phase0_enforce=False),
            manifest=manifest,
        )
        assert result.passed, result.failures

    def test_fails_on_pytest_failure(self, tmp_path):
        fitness = CompositeFitness(
            score=0.0, pytest_passed=False, pytest_duration_seconds=1.0,
            ruff_clean=True, freeze_passed=True,
            bug_repro_transitioned=False, bug_repro_evaluated=False,
            notes=["pytest failed"],
        )
        result = evaluate_phase4_gate(
            fitness=fitness,
            freeze=_ok_freeze(),
            config=EvolutionConfig(phase0_enforce=False),
            manifest=_make_manifest(tmp_path),
        )
        assert not result.passed
        assert any("pytest" in f for f in result.failures)

    def test_fails_on_signature_violation(self, tmp_path):
        bad_freeze = FreezeReport(
            signature_violations=["public function 'x' removed"],
            registry_violations=[],
            error_handling_baseline=2, error_handling_candidate=2,
        )
        result = evaluate_phase4_gate(
            fitness=_ok_fitness(),
            freeze=bad_freeze,
            config=EvolutionConfig(phase0_enforce=False),
            manifest=_make_manifest(tmp_path),
        )
        assert not result.passed

    def test_fails_on_bug_not_fixed(self, tmp_path):
        fitness = CompositeFitness(
            score=0.25, pytest_passed=True, pytest_duration_seconds=1.0,
            ruff_clean=True, freeze_passed=True,
            bug_repro_transitioned=False, bug_repro_evaluated=True,
            notes=["bug repro: candidate still failing"],
        )
        result = evaluate_phase4_gate(
            fitness=fitness,
            freeze=_ok_freeze(),
            config=EvolutionConfig(phase0_enforce=False),
            manifest=_make_manifest(tmp_path),
        )
        assert not result.passed
        assert any("Bug reproduction" in f for f in result.failures)
