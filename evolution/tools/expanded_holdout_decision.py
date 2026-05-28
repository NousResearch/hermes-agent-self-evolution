"""Phase 2E expanded holdout closeout decision.

This module records whether the existing Phase 2D 45-case default gate plus the
SessionDB-derived holdout is enough for candidate-only Phase 2 closeout, or
whether a 100+ held-out quality slice must be built before closeout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import click
from rich.console import Console

from evolution.tools.tool_description_eval import ToolSelectionCase, default_tool_selection_cases, load_tool_selection_cases

console = Console()

DECISION_SUFFICIENT = "current_45_plus_9_sufficient_for_phase2_closeout"
DECISION_BUILD_100_PLUS = "build_100_plus_heldout_quality_slice_before_phase2_closeout"


@dataclass(frozen=True)
class ExpandedHoldoutDecisionRunResult:
    """Paths and decision status for one expanded holdout decision run."""

    output_json: Path
    output_markdown: Path
    decision: str
    requires_100_case_slice: bool


def build_expanded_holdout_decision(
    default_cases: Sequence[ToolSelectionCase],
    holdout_cases: Sequence[ToolSelectionCase],
    *,
    heldout_review: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the Phase 2E expanded-holdout decision report."""

    default_tuple = tuple(default_cases)
    holdout_tuple = tuple(holdout_cases)
    combined_tuple = default_tuple + holdout_tuple

    default_pairs = _confusion_pairs(default_tuple)
    holdout_pairs = _confusion_pairs(holdout_tuple)
    default_expected = {case.expected_tool for case in default_tuple}
    holdout_expected = {case.expected_tool for case in holdout_tuple}
    default_requests = {case.user_request for case in default_tuple}
    holdout_requests = {case.user_request for case in holdout_tuple}
    default_categories = {case.category for case in default_tuple}
    holdout_categories = {case.category for case in holdout_tuple}

    new_expected_tools = sorted(holdout_expected - default_expected)
    new_confusion_pairs = sorted(holdout_pairs - default_pairs)
    overlapping_requests = sorted(default_requests & holdout_requests)
    overlapping_categories = sorted(default_categories & holdout_categories)
    review = dict(heldout_review) if heldout_review else {}
    review_failed_checks_raw = review.get("failed_checks")
    review_failed_checks_contract_ok = isinstance(review_failed_checks_raw, list)
    review_failed_checks = list(review_failed_checks_raw) if review_failed_checks_contract_ok else []
    review_holdout = _dict_from_review(review, "holdout")
    review_passed = review.get("passed") is True
    review_contract_ok = (
        review.get("mode") == "candidate-only-heldout-review"
        and review.get("candidate_only") is True
        and review.get("apply_ready") is False
        and review_holdout.get("case_count") == len(holdout_tuple)
        and review_failed_checks_contract_ok
        and not review_failed_checks
    )

    default_case_count_ok = len(default_tuple) >= 45
    holdout_case_count_ok = len(holdout_tuple) >= 9
    holdout_disjoint = not overlapping_requests and not overlapping_categories
    no_new_stable_coverage = not new_expected_tools and not new_confusion_pairs
    current_slice_sufficient = (
        default_case_count_ok
        and holdout_case_count_ok
        and holdout_disjoint
        and review_passed
        and review_contract_ok
        and no_new_stable_coverage
    )
    decision = DECISION_SUFFICIENT if current_slice_sufficient else DECISION_BUILD_100_PLUS
    requires_100_plus = not current_slice_sufficient

    remaining_items = ["benchmark_gate_decision", "human_review_checkpoint"]
    if requires_100_plus:
        remaining_items.insert(0, "100_plus_heldout_quality_slice")

    return {
        "phase": "2E",
        "mode": "expanded-holdout-decision",
        "candidate_only": True,
        "apply_ready": False,
        "decision": decision,
        "requires_100_case_slice_before_phase2_closeout": requires_100_plus,
        "default_gate": _case_stats(default_tuple),
        "sessiondb_holdout": _case_stats(holdout_tuple),
        "combined_slice": _case_stats(combined_tuple),
        "coverage_delta": {
            "new_expected_tools_from_holdout": new_expected_tools,
            "new_confusion_pairs_from_holdout": [list(pair) for pair in new_confusion_pairs],
            "overlapping_user_requests": overlapping_requests,
            "overlapping_categories": overlapping_categories,
        },
        "evidence": {
            "default_case_count_ok": default_case_count_ok,
            "holdout_case_count_ok": holdout_case_count_ok,
            "holdout_disjoint_from_default": holdout_disjoint,
            "holdout_review_passed": review_passed,
            "holdout_review_contract_ok": review_contract_ok,
            "holdout_review_mode": review.get("mode"),
            "holdout_review_candidate_only": review.get("candidate_only"),
            "holdout_review_apply_ready": review.get("apply_ready"),
            "holdout_failed_checks_contract_ok": review_failed_checks_contract_ok,
            "holdout_failed_checks": review_failed_checks,
            "holdout_metric_deltas": _dict_from_review(review, "metric_deltas"),
        },
        "policy": {
            "phase2_closeout_policy": "45-case default gate plus 9-case SessionDB holdout is sufficient for candidate-only Phase 2 closeout when the holdout adds no new expected tools/confusion pairs and the heldout review passes.",
            "defer_100_plus_slice_until": "before any default-gate promotion, active tool-schema apply, or broader Phase 3/benchmark expansion that needs lexical diversity beyond the current candidate-only gate.",
            "reason": "The SessionDB holdout adds sanitized real-session variants but no new expected-tool or confusion-pair coverage beyond the default gate; current evidence supports closeout without turning Phase 2E into a 100+ case expansion project.",
        },
        "remaining_phase2_closeout_items": remaining_items,
    }


def render_decision_markdown(report: Mapping[str, object]) -> str:
    """Render a deterministic Markdown decision artifact."""

    default_gate = _mapping(report["default_gate"])
    holdout = _mapping(report["sessiondb_holdout"])
    combined = _mapping(report["combined_slice"])
    coverage_delta = _mapping(report["coverage_delta"])
    evidence = _mapping(report["evidence"])
    policy = _mapping(report["policy"])
    requires_100 = bool(report["requires_100_case_slice_before_phase2_closeout"])
    decision_line = (
        "Decision: build a 100+ held-out quality slice before Phase 2 closeout."
        if requires_100
        else "Decision: current 45+9 slice is sufficient for Phase 2 closeout."
    )
    remaining = report["remaining_phase2_closeout_items"]
    assert isinstance(remaining, list)

    lines = [
        "# Phase 2E Expanded Holdout Decision",
        "",
        decision_line,
        "",
        f"100+ held-out quality slice required before Phase 2 closeout: {'yes' if requires_100 else 'no'}",
        "Candidate-only/no-apply: yes",
        "",
        "## Coverage snapshot",
        "",
        f"- Default gate cases: {default_gate['case_count']}",
        f"- SessionDB holdout cases: {holdout['case_count']}",
        f"- Combined slice cases: {combined['case_count']}",
        f"- Combined expected tools: {combined['expected_tool_count']}",
        f"- Combined confusion pairs: {combined['confusion_pair_count']}",
        "",
        "## Coverage delta from holdout",
        "",
        f"- New expected tools from holdout: {_format_list(coverage_delta['new_expected_tools_from_holdout'])}",
        f"- New confusion pairs from holdout: {_format_pairs(coverage_delta['new_confusion_pairs_from_holdout'])}",
        f"- Overlapping user requests: {_format_list(coverage_delta['overlapping_user_requests'])}",
        f"- Overlapping categories: {_format_list(coverage_delta['overlapping_categories'])}",
        "",
        "## Evidence",
        "",
        f"- Default case count OK: {str(evidence['default_case_count_ok']).lower()}",
        f"- Holdout case count OK: {str(evidence['holdout_case_count_ok']).lower()}",
        f"- Holdout disjoint from default: {str(evidence['holdout_disjoint_from_default']).lower()}",
        f"- Holdout review passed: {str(evidence['holdout_review_passed']).lower()}",
        f"- Holdout review contract OK: {str(evidence['holdout_review_contract_ok']).lower()}",
        f"- Holdout failed-checks contract OK: {str(evidence['holdout_failed_checks_contract_ok']).lower()}",
        f"- Holdout metric deltas: `{json.dumps(evidence['holdout_metric_deltas'], sort_keys=True)}`",
        "",
        "## Policy",
        "",
        f"- {policy['phase2_closeout_policy']}",
        f"- Defer 100+ slice until: {policy['defer_100_plus_slice_until']}",
        f"- Rationale: {policy['reason']}",
        "",
        "## Remaining Phase 2 closeout items",
        "",
        *[f"- `{item}`" for item in remaining],
        "",
    ]
    return "\n".join(lines)


def run_expanded_holdout_decision(
    *,
    holdout_jsonl: str | Path,
    heldout_review_json: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
) -> ExpandedHoldoutDecisionRunResult:
    """Run the expanded holdout decision and write JSON + Markdown artifacts."""

    holdout_path = Path(holdout_jsonl)
    review_path = Path(heldout_review_json)
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    heldout_review = json.loads(review_path.read_text())
    if not isinstance(heldout_review, Mapping):
        raise ValueError(f"Heldout review JSON must contain an object: {review_path}")
    report = build_expanded_holdout_decision(
        default_tool_selection_cases(),
        load_tool_selection_cases(holdout_path),
        heldout_review=heldout_review,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_decision_markdown(report))
    return ExpandedHoldoutDecisionRunResult(
        output_json=json_path,
        output_markdown=markdown_path,
        decision=str(report["decision"]),
        requires_100_case_slice=bool(report["requires_100_case_slice_before_phase2_closeout"]),
    )


def _case_stats(cases: Sequence[ToolSelectionCase]) -> dict[str, object]:
    expected_tools = sorted({case.expected_tool for case in cases})
    confusion_pairs = sorted(_confusion_pairs(cases))
    return {
        "case_count": len(cases),
        "expected_tool_count": len(expected_tools),
        "confusion_pair_count": len(confusion_pairs),
        "category_count": len({case.category for case in cases}),
        "expected_tools": expected_tools,
        "confusion_pairs": [list(pair) for pair in confusion_pairs],
    }


def _confusion_pairs(cases: Sequence[ToolSelectionCase]) -> set[tuple[str, str]]:
    return {(case.expected_tool, confusing_tool) for case in cases for confusing_tool in case.confusing_tools}


def _dict_from_review(review: Mapping[str, object] | None, key: str) -> dict[str, object]:
    value = review.get(key) if review else {}
    return dict(value) if isinstance(value, dict) else {}


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected mapping in decision report")
    return value


def _format_list(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _format_pairs(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(f"{pair[0]}→{pair[1]}" for pair in value if isinstance(pair, list) and len(pair) == 2)


@click.command()
@click.option(
    "--holdout-jsonl",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="SessionDB-derived held-out ToolSelectionCase JSONL fixture path",
)
@click.option(
    "--heldout-review-json",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Phase 2E heldout_review.json path",
)
@click.option("--output-json", required=True, type=click.Path(dir_okay=False), help="Decision JSON output path")
@click.option("--output-md", required=True, type=click.Path(dir_okay=False), help="Decision Markdown output path")
def main(holdout_jsonl: str, heldout_review_json: str, output_json: str, output_md: str) -> None:
    """Record whether the current Phase 2E holdout coverage is enough for closeout."""

    result = run_expanded_holdout_decision(
        holdout_jsonl=holdout_jsonl,
        heldout_review_json=heldout_review_json,
        output_json=output_json,
        output_markdown=output_md,
    )
    console.print("[bold green]Phase 2E expanded holdout decision written[/bold green]")
    console.print(f"  json: {result.output_json}")
    console.print(f"  markdown: {result.output_markdown}")
    console.print(
        f"  requires 100+ heldout before Phase 2 closeout: "
        f"{'yes' if result.requires_100_case_slice else 'no'}"
    )


if __name__ == "__main__":
    main()
