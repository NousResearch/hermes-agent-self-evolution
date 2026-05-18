"""Phase gate evaluation for plan-driven execution flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from evolution.core.config import EvolutionConfig
from evolution.core.reproducibility import ReproducibilityManifest


REQUIRED_MANIFEST_FIELDS = {
    "generated_at",
    "skill_name",
    "skill_path",
    "skill_sha256",
    "evolution_repo_git_sha",
    "hermes_repo_git_sha",
    "optimizer_model",
    "eval_model",
    "judge_model",
    "iterations",
    "dataset_train_sha256",
    "dataset_val_sha256",
    "dataset_holdout_sha256",
    "dataset_counts",
    "baseline_score",
    "evolved_score",
    "improvement",
    "elapsed_seconds",
}


@dataclass(frozen=True)
class PhaseGateResult:
    """Outcome for phase gate checks."""

    passed: bool
    checks: dict[str, bool]
    failures: list[str]
    relative_gain: float

    def to_dict(self) -> dict:
        return asdict(self)


def _manifest_has_required_fields(manifest: ReproducibilityManifest | dict) -> bool:
    data = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    return REQUIRED_MANIFEST_FIELDS.issubset(set(data.keys()))


def evaluate_phase1_gate(
    *,
    baseline_score: float,
    evolved_score: float,
    improvement: float,
    config: EvolutionConfig,
    manifest: ReproducibilityManifest | dict | None,
    benchmark_regression: float | None = None,
    safety_incidents: int = 0,
) -> PhaseGateResult:
    """Evaluate Phase 1 criteria from the plan."""
    denominator = max(abs(baseline_score), 1e-9)
    relative_gain = improvement / denominator

    checks = {
        "relative_gain": relative_gain >= config.phase1_min_relative_gain,
        "absolute_gain": improvement >= config.phase1_min_absolute_gain,
        "score_direction": evolved_score >= baseline_score,
        "benchmark_regression": (
            True
            if benchmark_regression is None
            else benchmark_regression <= config.phase1_max_benchmark_regression
        ),
        "reproducibility_manifest": (
            (not config.phase1_require_reproducibility_manifest)
            or (manifest is not None and _manifest_has_required_fields(manifest))
        ),
        "safety_incidents": (
            (not config.phase1_require_zero_safety_incidents) or safety_incidents == 0
        ),
    }

    failures: list[str] = []
    if not checks["relative_gain"] and not checks["absolute_gain"]:
        failures.append(
            "Improvement gate failed: expected relative gain "
            f">= {config.phase1_min_relative_gain:.2%} or absolute gain "
            f">= {config.phase1_min_absolute_gain:.3f}, got "
            f"relative={relative_gain:.2%}, absolute={improvement:.3f}",
        )
    if not checks["score_direction"]:
        failures.append("Evolved score is lower than baseline")
    if not checks["benchmark_regression"]:
        failures.append(
            "Benchmark regression exceeded threshold "
            f"({benchmark_regression:.3f} > {config.phase1_max_benchmark_regression:.3f})",
        )
    if not checks["reproducibility_manifest"]:
        failures.append("Reproducibility manifest missing required fields")
    if not checks["safety_incidents"]:
        failures.append(f"Safety incident gate failed ({safety_incidents} incidents)")

    return PhaseGateResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
        relative_gain=relative_gain,
    )


def write_phase_gate_result(path: Path, result: PhaseGateResult) -> None:
    """Persist phase gate decision."""
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


REQUIRED_TOOL_MANIFEST_FIELDS = {
    "generated_at",
    "artifact_type",
    "tool_name",
    "tool_source_path",
    "baseline_description_sha256",
    "evolution_repo_git_sha",
    "hermes_repo_git_sha",
    "optimizer_model",
    "eval_model",
    "judge_model",
    "iterations",
    "dataset_train_sha256",
    "dataset_val_sha256",
    "dataset_holdout_sha256",
    "dataset_counts",
    "baseline_score",
    "evolved_score",
    "improvement",
    "elapsed_seconds",
}


def _tool_manifest_has_required_fields(manifest) -> bool:
    data = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    return REQUIRED_TOOL_MANIFEST_FIELDS.issubset(set(data.keys()))


REQUIRED_PROMPT_SECTION_MANIFEST_FIELDS = {
    "generated_at",
    "artifact_type",
    "section_name",
    "section_source_path",
    "baseline_text_sha256",
    "evolution_repo_git_sha",
    "hermes_repo_git_sha",
    "optimizer_model",
    "eval_model",
    "judge_model",
    "iterations",
    "dataset_train_sha256",
    "dataset_val_sha256",
    "dataset_holdout_sha256",
    "dataset_counts",
    "baseline_score",
    "evolved_score",
    "improvement",
    "elapsed_seconds",
}


def _prompt_section_manifest_has_required_fields(manifest) -> bool:
    data = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    return REQUIRED_PROMPT_SECTION_MANIFEST_FIELDS.issubset(set(data.keys()))


def evaluate_phase3_gate(
    *,
    baseline_score: float,
    evolved_score: float,
    improvement: float,
    config: EvolutionConfig,
    manifest,
    benchmark_regression: float | None = None,
    safety_incidents: int = 0,
    identity_traits: dict[str, bool] | None = None,
) -> PhaseGateResult:
    """Phase 3 gate: behavioral lift + zero benchmark regression + trait preservation.

    Identity sections must keep core traits (``helpful``/``direct``/``admits_uncertainty``).
    When ``identity_traits`` is provided, every required trait must remain present.
    """
    denominator = max(abs(baseline_score), 1e-9)
    relative_gain = improvement / denominator

    traits_ok = True if identity_traits is None else all(identity_traits.values())

    checks = {
        "relative_gain": relative_gain >= config.phase1_min_relative_gain,
        "absolute_gain": improvement >= config.phase1_min_absolute_gain,
        "score_direction": evolved_score >= baseline_score,
        "benchmark_regression": (
            True
            if benchmark_regression is None
            else benchmark_regression <= config.phase1_max_benchmark_regression
        ),
        "identity_traits_preserved": traits_ok,
        "reproducibility_manifest": (
            (not config.phase1_require_reproducibility_manifest)
            or (manifest is not None and _prompt_section_manifest_has_required_fields(manifest))
        ),
        "safety_incidents": (
            (not config.phase1_require_zero_safety_incidents) or safety_incidents == 0
        ),
    }

    failures: list[str] = []
    if not checks["relative_gain"] and not checks["absolute_gain"]:
        failures.append(
            "Improvement gate failed: expected relative gain "
            f">= {config.phase1_min_relative_gain:.2%} or absolute gain "
            f">= {config.phase1_min_absolute_gain:.3f}, got "
            f"relative={relative_gain:.2%}, absolute={improvement:.3f}",
        )
    if not checks["score_direction"]:
        failures.append("Evolved score is lower than baseline")
    if not checks["benchmark_regression"]:
        failures.append(
            "Benchmark regression exceeded threshold "
            f"({benchmark_regression:.3f} > {config.phase1_max_benchmark_regression:.3f})",
        )
    if not checks["identity_traits_preserved"]:
        missing = [k for k, v in (identity_traits or {}).items() if not v]
        failures.append(f"Identity traits not preserved: missing {missing}")
    if not checks["reproducibility_manifest"]:
        failures.append("Reproducibility manifest missing required fields")
    if not checks["safety_incidents"]:
        failures.append(f"Safety incident gate failed ({safety_incidents} incidents)")

    return PhaseGateResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
        relative_gain=relative_gain,
    )


REQUIRED_CODE_MANIFEST_FIELDS = {
    "generated_at",
    "artifact_type",
    "tool_name",
    "tool_source_path",
    "baseline_source_sha256",
    "evolution_repo_git_sha",
    "hermes_repo_git_sha",
    "optimizer_model",
    "iterations",
    "engine",
    "best_fitness_score",
    "best_iteration",
}


def _code_manifest_has_required_fields(manifest) -> bool:
    data = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    return REQUIRED_CODE_MANIFEST_FIELDS.issubset(set(data.keys()))


def evaluate_phase4_gate(
    *,
    fitness,
    freeze,
    config: EvolutionConfig,
    manifest,
    safety_incidents: int = 0,
) -> PhaseGateResult:
    """Phase 4 gate: pytest 100%, signature/registry/error-handling freeze, bug-repro transition.

    Unlike Phases 1-3 which gate on a relative improvement threshold, Phase 4
    gates on structural invariants: all tests pass, no public API changed, and
    (if a bug-repro test was supplied) the bug actually got fixed.
    """
    fitness_score = fitness.score if hasattr(fitness, "score") else float(fitness)
    pytest_passed = fitness.pytest_passed if hasattr(fitness, "pytest_passed") else fitness_score > 0
    freeze_passed = freeze.passed if hasattr(freeze, "passed") else bool(freeze)
    bug_transitioned = fitness.bug_repro_transitioned if hasattr(fitness, "bug_repro_transitioned") else False
    bug_evaluated = fitness.bug_repro_evaluated if hasattr(fitness, "bug_repro_evaluated") else False

    checks = {
        "pytest_passes_100pct": pytest_passed,
        "signature_frozen": freeze_passed,
        "registry_frozen": freeze_passed,
        "error_handling_not_decreased": freeze_passed,
        "bug_repro_transitioned": bug_transitioned if bug_evaluated else True,
        "reproducibility_manifest": (
            (not config.phase1_require_reproducibility_manifest)
            or (manifest is not None and _code_manifest_has_required_fields(manifest))
        ),
        "safety_incidents": (
            (not config.phase1_require_zero_safety_incidents) or safety_incidents == 0
        ),
    }

    failures: list[str] = []
    if not checks["pytest_passes_100pct"]:
        failures.append("pytest does not pass 100% on candidate")
    if not checks["signature_frozen"]:
        sig_violations = getattr(freeze, "signature_violations", ["(unknown)"])
        failures.append(f"Public function signatures changed: {sig_violations}")
    if not checks["registry_frozen"]:
        reg_violations = getattr(freeze, "registry_violations", ["(unknown)"])
        failures.append(f"Registry calls changed: {reg_violations}")
    if not checks["error_handling_not_decreased"]:
        failures.append("Error-handling coverage decreased")
    if not checks["bug_repro_transitioned"]:
        failures.append("Bug reproduction test did not transition FAIL -> PASS")
    if not checks["reproducibility_manifest"]:
        failures.append("Reproducibility manifest missing required fields")
    if not checks["safety_incidents"]:
        failures.append(f"Safety incident gate failed ({safety_incidents} incidents)")

    return PhaseGateResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
        relative_gain=float(fitness_score),
    )


def evaluate_phase2_gate(
    *,
    baseline_score: float,
    evolved_score: float,
    improvement: float,
    config: EvolutionConfig,
    manifest,
    per_tool_regression: dict[str, float] | None = None,
    benchmark_regression: float | None = None,
    safety_incidents: int = 0,
) -> PhaseGateResult:
    """Phase 2 gate: tool-selection lift + cross-tool non-regression.

    `per_tool_regression` maps peer-tool name to its selection-rate delta
    (evolved minus baseline); negative values indicate a peer regressed.
    Any regression worse than the benchmark-regression threshold fails the
    cross-tool check, matching the plan's "no individual tool's selection
    rate regresses" constraint.
    """
    denominator = max(abs(baseline_score), 1e-9)
    relative_gain = improvement / denominator

    if per_tool_regression:
        worst_peer_regression = min(per_tool_regression.values())
    else:
        worst_peer_regression = 0.0

    checks = {
        "relative_gain": relative_gain >= config.phase1_min_relative_gain,
        "absolute_gain": improvement >= config.phase1_min_absolute_gain,
        "score_direction": evolved_score >= baseline_score,
        "cross_tool_no_regression": worst_peer_regression >= -config.phase1_max_benchmark_regression,
        "benchmark_regression": (
            True
            if benchmark_regression is None
            else benchmark_regression <= config.phase1_max_benchmark_regression
        ),
        "reproducibility_manifest": (
            (not config.phase1_require_reproducibility_manifest)
            or (manifest is not None and _tool_manifest_has_required_fields(manifest))
        ),
        "safety_incidents": (
            (not config.phase1_require_zero_safety_incidents) or safety_incidents == 0
        ),
    }

    failures: list[str] = []
    if not checks["relative_gain"] and not checks["absolute_gain"]:
        failures.append(
            "Improvement gate failed: expected relative gain "
            f">= {config.phase1_min_relative_gain:.2%} or absolute gain "
            f">= {config.phase1_min_absolute_gain:.3f}, got "
            f"relative={relative_gain:.2%}, absolute={improvement:.3f}",
        )
    if not checks["score_direction"]:
        failures.append("Evolved score is lower than baseline")
    if not checks["cross_tool_no_regression"]:
        worst_tool = min(per_tool_regression or {}, key=lambda k: per_tool_regression[k]) if per_tool_regression else ""
        failures.append(
            f"Cross-tool regression detected (worst peer {worst_tool!r}: "
            f"{worst_peer_regression:+.3f} > -{config.phase1_max_benchmark_regression:.3f})",
        )
    if not checks["benchmark_regression"]:
        failures.append(
            "Benchmark regression exceeded threshold "
            f"({benchmark_regression:.3f} > {config.phase1_max_benchmark_regression:.3f})",
        )
    if not checks["reproducibility_manifest"]:
        failures.append("Reproducibility manifest missing required fields")
    if not checks["safety_incidents"]:
        failures.append(f"Safety incident gate failed ({safety_incidents} incidents)")

    return PhaseGateResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
        relative_gain=relative_gain,
    )
