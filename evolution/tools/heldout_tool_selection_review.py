"""Phase 2E held-out tool-selection review.

This module compares candidate-only tool descriptions against an explicit
holdout case set. It is intentionally read-only: it writes a review report only
and never patches Hermes Agent tool schemas, source files, or runtime config.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

import click
from rich.console import Console

from evolution.tools.evolve_tool_descriptions import load_inventory_from_json
from evolution.tools.tool_description_eval import (
    DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS,
    CrossToolGateThresholds,
    ToolDescriptionCandidate,
    ToolInventoryRecord,
    ToolSelectionCase,
    candidates_from_inventory,
    default_tool_selection_cases,
    evaluate_cross_tool_gate,
    load_tool_selection_cases,
)

console = Console()
MAX_DESCRIPTION_CHARS = 500


@dataclass(frozen=True)
class HoldoutReviewRunResult:
    """Paths and status for one Phase 2E holdout review run."""

    output_path: Path
    passed: bool
    failed_checks: tuple[str, ...]


def load_candidate_descriptions(path: str | Path) -> tuple[ToolDescriptionCandidate, ...]:
    """Load candidate description records from a Phase 2C/2D artifact."""

    candidates_path = Path(path)
    raw = json.loads(candidates_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Candidate descriptions JSON must contain a list: {candidates_path}")

    candidates: list[ToolDescriptionCandidate] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate description record must be an object at {candidates_path}:{index}")
        name = item.get("name")
        toolset = item.get("toolset")
        baseline_description = item.get("baseline_description")
        candidate_description = item.get("candidate_description")
        parameter_descriptions = item.get("parameter_descriptions", {})
        if not all(isinstance(value, str) for value in (name, toolset, baseline_description, candidate_description)):
            raise ValueError(f"Candidate record missing string name/toolset/descriptions at {candidates_path}:{index}")
        if not isinstance(parameter_descriptions, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in parameter_descriptions.items()
        ):
            raise ValueError(f"parameter_descriptions must be a string map at {candidates_path}:{index}")
        name_s = cast(str, name)
        toolset_s = cast(str, toolset)
        baseline_s = cast(str, baseline_description)
        candidate_s = cast(str, candidate_description)
        parameter_map = cast(dict[str, str], parameter_descriptions)
        candidates.append(
            ToolDescriptionCandidate(
                name=name_s,
                toolset=toolset_s,
                baseline_description=baseline_s,
                candidate_description=candidate_s,
                parameter_descriptions=dict(parameter_map),
            )
        )
    return tuple(candidates)


def build_holdout_review_report(
    baseline_candidates: Sequence[ToolDescriptionCandidate],
    candidate_candidates: Sequence[ToolDescriptionCandidate],
    cases: Sequence[ToolSelectionCase],
    *,
    holdout_source: str,
    default_gate_case_count: int | None = None,
) -> dict[str, object]:
    """Build a candidate-only Phase 2E holdout improvement/no-regression report."""

    eval_cases = tuple(cases)
    default_case_count = default_gate_case_count if default_gate_case_count is not None else len(default_tool_selection_cases())
    thresholds = CrossToolGateThresholds(
        min_case_count=max(len(eval_cases), 1),
        min_selection_accuracy=0.0,
        min_wrong_tool_avoidance=0.0,
        max_per_tool_regression=0.0,
    )
    gate = evaluate_cross_tool_gate(baseline_candidates, candidate_candidates, eval_cases, thresholds=thresholds)
    baseline_metrics = dict(gate.baseline_metrics)
    candidate_metrics = dict(gate.candidate_metrics)
    metric_deltas = _metric_deltas(baseline_metrics, candidate_metrics)
    failed_checks = list(gate.failed_checks)
    failed_checks.extend(_holdout_case_tool_coverage_failures(eval_cases, baseline_candidates, candidate_candidates))
    failed_checks.extend(_candidate_constraint_failures(candidate_candidates))
    for metric in ("selection_accuracy", "wrong_tool_avoidance"):
        if metric_deltas[metric] < 0:
            candidate_value = _float_metric(candidate_metrics, metric)
            baseline_value = _float_metric(baseline_metrics, metric)
            failed_checks.append(
                f"aggregate_regression {metric} "
                f"{candidate_value:.4f} < baseline {baseline_value:.4f}"
            )

    return {
        "phase": "2E",
        "mode": "candidate-only-heldout-review",
        "apply_ready": False,
        "summary": "Held-out tool-selection review for candidate descriptions; active Hermes tool schemas are not modified.",
        "candidate_only": True,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "holdout": {
            "source": holdout_source,
            "case_count": len(eval_cases),
            "default_gate_case_count": default_case_count,
            "included_in_default_gate": False,
            "promotion_decision": "holdout",
        },
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": metric_deltas,
        "per_tool_regressions": [asdict(regression) for regression in gate.per_tool_regressions],
    }


def run_holdout_review(
    *,
    inventory_json: str | Path,
    candidates_json: str | Path,
    cases_jsonl: str | Path,
    output_json: str | Path,
) -> HoldoutReviewRunResult:
    """Run the Phase 2E holdout review and write a JSON report."""

    records = load_inventory_from_json(inventory_json)
    baseline_candidates = candidates_from_inventory(records)
    candidate_candidates = load_candidate_descriptions(candidates_json)
    _validate_candidate_inventory_coverage(records, candidate_candidates)
    cases = load_tool_selection_cases(cases_jsonl)
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_holdout_review_report(
        baseline_candidates,
        candidate_candidates,
        cases,
        holdout_source=str(cases_jsonl),
    )
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    failed_checks_raw = report.get("failed_checks", ())
    failed_checks = tuple(str(check) for check in failed_checks_raw) if isinstance(failed_checks_raw, list) else ()
    return HoldoutReviewRunResult(
        output_path=output_path,
        passed=bool(report.get("passed")),
        failed_checks=failed_checks,
    )


def _validate_candidate_inventory_coverage(
    records: Sequence[ToolInventoryRecord],
    candidates: Sequence[ToolDescriptionCandidate],
) -> None:
    inventory_names = [record.name for record in records]
    candidate_names = [candidate.name for candidate in candidates]
    duplicate_inventory = sorted({name for name in inventory_names if inventory_names.count(name) > 1})
    duplicate_candidates = sorted({name for name in candidate_names if candidate_names.count(name) > 1})
    if duplicate_inventory or duplicate_candidates:
        raise ValueError(
            "Inventory and candidate descriptions must have unique tool names; "
            f"duplicate_inventory={duplicate_inventory}, duplicate_candidates={duplicate_candidates}"
        )

    inventory_map = {record.name: record for record in records}
    candidate_map = {candidate.name: candidate for candidate in candidates}
    if set(inventory_map) != set(candidate_map):
        missing = sorted(set(inventory_map) - set(candidate_map))
        extra = sorted(set(candidate_map) - set(inventory_map))
        raise ValueError(
            "Candidate descriptions must cover exactly the inventory tools; "
            f"missing={missing}, extra={extra}"
        )

    mismatched_toolsets = sorted(
        name for name, candidate in candidate_map.items() if candidate.toolset != inventory_map[name].toolset
    )
    mismatched_baselines = sorted(
        name
        for name, candidate in candidate_map.items()
        if candidate.baseline_description.strip() != inventory_map[name].description.strip()
    )
    if mismatched_toolsets or mismatched_baselines:
        raise ValueError(
            "Candidate descriptions must preserve inventory toolset and baseline descriptions; "
            f"toolset_mismatch={mismatched_toolsets}, baseline_mismatch={mismatched_baselines}"
        )


def _holdout_case_tool_coverage_failures(
    cases: Sequence[ToolSelectionCase],
    baseline_candidates: Sequence[ToolDescriptionCandidate],
    candidate_candidates: Sequence[ToolDescriptionCandidate],
) -> list[str]:
    expected_tools = {case.expected_tool for case in cases}
    confusing_tools = {tool for case in cases for tool in case.confusing_tools}
    failures: list[str] = []
    for label, candidates in (("baseline", baseline_candidates), ("candidate", candidate_candidates)):
        names = {candidate.name for candidate in candidates}
        missing_expected = sorted(expected_tools - names)
        missing_confusing = sorted(confusing_tools - names)
        if missing_expected or missing_confusing:
            failures.append(
                f"holdout_tool_coverage {label} missing expected={missing_expected} confusing={missing_confusing}"
            )
    return failures


def _candidate_constraint_failures(candidates: Sequence[ToolDescriptionCandidate]) -> list[str]:
    failures: list[str] = []
    for candidate in candidates:
        description_length = len(candidate.candidate_description.strip())
        if not 0 < description_length <= MAX_DESCRIPTION_CHARS:
            failures.append(
                f"candidate_constraint description_length {candidate.name} "
                f"{description_length} not in 1..{MAX_DESCRIPTION_CHARS}"
            )
        for param_name, param_description in candidate.parameter_descriptions.items():
            param_length = len(param_description.strip())
            if not 0 < param_length <= DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS:
                failures.append(
                    f"candidate_constraint parameter_description_length {candidate.name}.{param_name} "
                    f"{param_length} not in 1..{DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS}"
                )
    return failures


def _float_metric(metrics: Mapping[str, object], metric: str) -> float:
    value = metrics.get(metric, 0.0)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"Metric must be numeric: {metric}")


def _metric_deltas(
    baseline_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for metric in (
        "selection_accuracy",
        "wrong_tool_avoidance",
        "argument_cue_coverage",
        "constraint_pass_rate",
    ):
        baseline = _float_metric(baseline_metrics, metric)
        candidate = _float_metric(candidate_metrics, metric)
        deltas[metric] = round(candidate - baseline, 4)
    return deltas


@click.command()
@click.option("--inventory-json", required=True, type=click.Path(exists=True, dir_okay=False), help="Phase 2D inventory.json path")
@click.option(
    "--candidates-json",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Phase 2D candidate_descriptions.json path",
)
@click.option(
    "--cases-jsonl",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Held-out ToolSelectionCase JSONL fixture path",
)
@click.option("--output-json", required=True, type=click.Path(dir_okay=False), help="Held-out review JSON output path")
def main(inventory_json: str, candidates_json: str, cases_jsonl: str, output_json: str) -> None:
    """Compare baseline/candidate descriptions on a held-out tool-selection slice."""

    result = run_holdout_review(
        inventory_json=inventory_json,
        candidates_json=candidates_json,
        cases_jsonl=cases_jsonl,
        output_json=output_json,
    )
    console.print("[bold green]Phase 2E held-out review written[/bold green]")
    console.print(f"  report: {result.output_path}")
    console.print("  active Hermes tool schemas modified: no")
    if result.passed:
        console.print("  Phase 2E held-out review: passed")
        return

    console.print("  Phase 2E held-out review: failed")
    for failed_check in result.failed_checks:
        console.print(f"    - {failed_check}")
    raise click.ClickException(f"Phase 2E held-out review failed; inspect report: {result.output_path}")


if __name__ == "__main__":
    main()
