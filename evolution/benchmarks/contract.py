"""Shared dry-run fixture benchmark adapter contract.

The Phase 3 benchmark adapters intentionally implement a deterministic,
read-only fixture contract first. They validate prompt artifacts, evaluate small
committed fixtures, and write a machine-readable benchmark report. They do not
call external services, mutate prompt/source files, or approve active apply.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

ADAPTER_VERSION = "phase3-benchmark-adapter-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = "output/phase3-system-prompt/"
ALLOWED_OUTPUT_SUFFIX = ".json"
PHASE3_OUTPUT_ROOT = REPO_ROOT / ALLOWED_OUTPUT_ROOT


@dataclass(frozen=True)
class FixtureBenchmarkCase:
    """One deterministic benchmark fixture case."""

    id: str
    category: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class PromptArtifact:
    """Prompt artifact metadata used by dry-run reports."""

    path: Path
    sha256: str
    bytes: int
    normalized_text: str
    normalized_text_sha256: str

    def report(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "normalized_text_sha256": self.normalized_text_sha256,
        }


def run_fixture_benchmark(
    *,
    benchmark: str,
    pass_condition: str,
    baseline_prompt: str | Path,
    candidate_prompt: str | Path,
    fixtures_jsonl: str | Path,
    output_json: str | Path,
    dry_run: bool,
    preset: str | None = None,
) -> dict[str, object]:
    """Run a deterministic read-only fixture benchmark and write its report.

    This adapter contract is deliberately fail-closed: only dry-run fixture mode
    is supported for now, and the only write target is ``output_json``.
    """

    if dry_run is not True:
        raise ValueError("Phase 3 benchmark adapters currently require --dry-run")

    baseline_path = Path(baseline_prompt)
    candidate_path = Path(candidate_prompt)
    fixtures_path = Path(fixtures_jsonl)
    output_path = _normalize_output_json_path(Path(output_json))
    _validate_output_json_path(output_path)
    _validate_distinct_output_path(output_path, (baseline_path, candidate_path, fixtures_path))

    baseline_artifact = load_prompt_artifact(baseline_path)
    candidate_artifact = load_prompt_artifact(candidate_path)
    cases = load_fixture_cases(fixtures_path)
    if not cases:
        raise ValueError(f"Fixture case file must contain at least one case: {fixtures_path}")

    case_reports = [
        _evaluate_case(case, baseline_artifact.normalized_text, candidate_artifact.normalized_text)
        for case in cases
    ]
    baseline_score = sum(_numeric_case_value(case, "baseline_score") for case in case_reports)
    candidate_score = sum(_numeric_case_value(case, "candidate_score") for case in case_reports)
    candidate_regressions = [case for case in case_reports if case["passed"] is not True]
    failed_checks = [
        f"case_regression {case['id']} candidate={case['candidate_score']} baseline={case['baseline_score']}"
        for case in candidate_regressions
    ]
    if candidate_score < baseline_score:
        failed_checks.append(f"aggregate_regression candidate_score {candidate_score:.4f} < baseline {baseline_score:.4f}")

    total_weight = sum(case.weight for case in cases)
    report: dict[str, object] = {
        "benchmark": benchmark,
        "adapter_version": ADAPTER_VERSION,
        "mode": "dry-run-fixture",
        "dry_run": True,
        "candidate_only": True,
        "read_only": True,
        "external_calls_performed": False,
        "apply_ready": False,
        "pass_condition": pass_condition,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "prompt_artifacts": {
            "baseline": baseline_artifact.report(),
            "candidate": candidate_artifact.report(),
        },
        "fixture_cases": {
            "path": str(fixtures_path),
            "case_count": len(cases),
            "categories": sorted({case.category for case in cases}),
        },
        "metrics": {
            "case_count": len(cases),
            "total_weight": total_weight,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "score_delta": candidate_score - baseline_score,
            "candidate_regression_count": len(candidate_regressions),
        },
        "cases": case_reports,
        "artifacts": {
            "output_json": str(output_path),
            "baseline_prompt": str(baseline_path),
            "candidate_prompt": str(candidate_path),
            "fixtures_jsonl": str(fixtures_path),
        },
        "write_targets": [str(output_path)],
        "output_constraints": {
            "allowed_root": ALLOWED_OUTPUT_ROOT,
            "suffix": ALLOWED_OUTPUT_SUFFIX,
            "fresh_output_required": True,
            "symlink_output_allowed": False,
            "hardlink_output_allowed": False,
            "input_output_overlap_allowed": False,
        },
    }
    if preset is not None:
        report["preset"] = preset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_prompt_artifact(path: str | Path) -> PromptArtifact:
    """Read a prompt artifact and return normalized metadata."""

    prompt_path = Path(path)
    raw_bytes = prompt_path.read_bytes()
    text = raw_bytes.decode("utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = text
    normalized_text = normalize_prompt_text(payload)
    return PromptArtifact(
        path=prompt_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        bytes=len(raw_bytes),
        normalized_text=normalized_text,
        normalized_text_sha256=hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
    )


def normalize_prompt_text(payload: object) -> str:
    """Extract deterministic prompt text from common JSON/string artifacts."""

    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        sections = payload.get("sections")
        if isinstance(sections, Mapping):
            lines: list[str] = []
            for key in sorted(sections):
                value = sections[key]
                if isinstance(key, str):
                    lines.append(f"## {key}")
                lines.append(normalize_prompt_text(value))
            return "\n".join(line for line in lines if line)
        system_prompt = payload.get("system_prompt")
        if isinstance(system_prompt, str):
            return system_prompt
        return "\n".join(normalize_prompt_text(value) for _, value in sorted(payload.items()) if value is not None)
    if isinstance(payload, Sequence) and not isinstance(payload, bytes | bytearray):
        return "\n".join(normalize_prompt_text(item) for item in payload)
    return str(payload)


def load_fixture_cases(path: str | Path) -> tuple[FixtureBenchmarkCase, ...]:
    """Load deterministic fixture cases from JSONL."""

    cases_path = Path(path)
    cases: list[FixtureBenchmarkCase] = []
    for line_number, line in enumerate(cases_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Fixture case must be an object at {cases_path}:{line_number}")
        cases.append(_fixture_case_from_mapping(raw, cases_path, line_number))
    return tuple(cases)


def _fixture_case_from_mapping(
    raw: Mapping[str, object],
    path: Path,
    line_number: int,
) -> FixtureBenchmarkCase:
    case_id = raw.get("id")
    category = raw.get("category")
    required_terms = raw.get("required_terms")
    forbidden_terms = raw.get("forbidden_terms", [])
    weight = raw.get("weight", 1.0)
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"Fixture case id must be a non-empty string at {path}:{line_number}")
    if not isinstance(category, str) or not category:
        raise ValueError(f"Fixture case category must be a non-empty string at {path}:{line_number}")
    if not _is_string_sequence(required_terms) or not required_terms:
        raise ValueError(f"Fixture required_terms must be a non-empty string list at {path}:{line_number}")
    if not _is_string_sequence(forbidden_terms):
        raise ValueError(f"Fixture forbidden_terms must be a string list at {path}:{line_number}")
    if not isinstance(weight, int | float) or weight <= 0:
        raise ValueError(f"Fixture weight must be a positive number at {path}:{line_number}")
    required_terms_s = cast(Sequence[str], required_terms)
    forbidden_terms_s = cast(Sequence[str], forbidden_terms)
    return FixtureBenchmarkCase(
        id=case_id,
        category=category,
        required_terms=tuple(required_terms_s),
        forbidden_terms=tuple(forbidden_terms_s),
        weight=float(weight),
    )


def _evaluate_case(
    case: FixtureBenchmarkCase,
    baseline_text: str,
    candidate_text: str,
) -> dict[str, object]:
    baseline_score, baseline_missing, baseline_forbidden = _score_text(case, baseline_text)
    candidate_score, candidate_missing, candidate_forbidden = _score_text(case, candidate_text)
    passed = candidate_score >= baseline_score and not candidate_forbidden
    return {
        "id": case.id,
        "category": case.category,
        "weight": case.weight,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "score_delta": candidate_score - baseline_score,
        "baseline_missing_terms": baseline_missing,
        "candidate_missing_terms": candidate_missing,
        "baseline_forbidden_hits": baseline_forbidden,
        "candidate_forbidden_hits": candidate_forbidden,
        "passed": passed,
    }


def _score_text(case: FixtureBenchmarkCase, text: str) -> tuple[float, list[str], list[str]]:
    normalized = text.casefold()
    required = list(case.required_terms)
    forbidden = list(case.forbidden_terms)
    missing = [term for term in required if term.casefold() not in normalized]
    forbidden_hits = [term for term in forbidden if term.casefold() in normalized]
    required_hits = len(required) - len(missing)
    raw_score = (required_hits - len(forbidden_hits)) / max(len(required), 1)
    return max(0.0, raw_score) * case.weight, missing, forbidden_hits


def _numeric_case_value(case: Mapping[str, object], key: str) -> float:
    value = case[key]
    if not isinstance(value, int | float):
        raise TypeError(f"Case value must be numeric: {key}")
    return float(value)


def _normalize_output_json_path(output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path
    return REPO_ROOT / output_path


def _validate_output_json_path(output_path: Path) -> None:
    if output_path.suffix.lower() != ALLOWED_OUTPUT_SUFFIX:
        raise ValueError(f"output-json must use a {ALLOWED_OUTPUT_SUFFIX} suffix: {output_path}")
    output_resolved = output_path.resolve()
    allowed_root_resolved = PHASE3_OUTPUT_ROOT.resolve()
    if not output_resolved.is_relative_to(allowed_root_resolved):
        raise ValueError(f"output-json must stay under {ALLOWED_OUTPUT_ROOT}: {output_path}")


def _validate_distinct_output_path(output_path: Path, input_paths: Sequence[Path]) -> None:
    if output_path.is_symlink():
        raise ValueError(f"output-json must not be a symlink: {output_path}")
    if output_path.exists():
        for input_path in input_paths:
            if output_path.samefile(input_path):
                raise ValueError(f"output-json must not overwrite input artifact: {input_path}")
        raise ValueError(f"output-json must not already exist: {output_path}")

    output_resolved = output_path.resolve()
    for input_path in input_paths:
        if output_resolved == input_path.resolve():
            raise ValueError(f"output-json must not overwrite input artifact: {input_path}")


def _is_string_sequence(value: object) -> bool:
    return isinstance(value, list | tuple) and all(isinstance(item, str) and item for item in value)
