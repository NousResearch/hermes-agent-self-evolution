"""Phase 2C/2D candidate-only tool description generation and gating.

This module deliberately does not patch Hermes Agent tool schemas. It reads a
Hermes tool inventory, generates deterministic candidate descriptions from the
Phase 2B golden tool-selection cases, evaluates them with the existing
candidate-only scaffold, applies the formal Phase 2D cross-tool gate, and writes
review artifacts.
"""

from __future__ import annotations

import difflib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import click
from rich.console import Console

from evolution.core.config import get_hermes_agent_path
from evolution.tools.tool_description_eval import (
    ToolDescriptionCandidate,
    ToolInventoryRecord,
    ToolSelectionCase,
    build_candidate_only_report,
    candidates_from_inventory,
    default_tool_selection_cases,
    evaluate_cross_tool_gate,
)

console = Console()


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Paths and formal-gate status from one candidate-only Phase 2C/2D run."""

    output_dir: Path
    inventory_path: Path
    candidates_path: Path
    report_path: Path
    diff_path: Path
    phase2d_gate_passed: bool
    phase2d_failed_checks: tuple[str, ...]


def collect_hermes_tool_inventory(hermes_repo: str | Path | None = None) -> list[ToolInventoryRecord]:
    """Collect a read-only inventory from a Hermes Agent checkout.

    Importing Hermes tool modules is used only to inspect registry metadata. This
    function does not call tool handlers, edit Hermes source files, or write
    configuration. Because discovery imports Python modules, point this only at
    trusted Hermes Agent checkouts.
    """

    repo = Path(hermes_repo).expanduser() if hermes_repo else get_hermes_agent_path()
    tools_dir = repo / "tools"
    if not tools_dir.exists():
        raise FileNotFoundError(f"Hermes tools directory not found: {tools_dir}")

    repo_string = str(repo)
    inserted = False
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
        inserted = True
    try:
        from tools.registry import discover_builtin_tools, registry  # type: ignore

        discover_builtin_tools(tools_dir)
        records: list[ToolInventoryRecord] = []
        for name in registry.get_all_tool_names():
            entry = registry.get_entry(name)
            if entry is None:
                continue
            records.append(
                ToolInventoryRecord(
                    name=entry.name,
                    toolset=entry.toolset,
                    description=entry.description or entry.schema.get("description", ""),
                    schema=dict(entry.schema),
                )
            )
        return sorted(records, key=lambda record: (record.toolset, record.name))
    finally:
        if inserted:
            try:
                sys.path.remove(repo_string)
            except ValueError:
                pass


def load_inventory_from_json(path: str | Path) -> list[ToolInventoryRecord]:
    """Load a tool inventory JSON file.

    Accepted shapes:
    - a list of tool records;
    - {"tools": [...]} / {"records": [...]} / {"inventory": [...]}.
    """

    inventory_path = Path(path)
    raw = json.loads(inventory_path.read_text())
    if isinstance(raw, dict):
        records_raw = None
        for key in ("tools", "records", "inventory"):
            if key in raw:
                records_raw = raw[key]
                break
    else:
        records_raw = raw
    if not isinstance(records_raw, list):
        raise ValueError(f"Inventory JSON must contain a list of tool records: {inventory_path}")

    records: list[ToolInventoryRecord] = []
    for item in records_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Inventory record must be an object: {item!r}")
        name = item.get("name")
        toolset = item.get("toolset")
        description = item.get("description")
        schema = item.get("schema", {})
        if not isinstance(name, str) or not isinstance(toolset, str) or not isinstance(description, str):
            raise ValueError(f"Inventory record missing string name/toolset/description: {item!r}")
        if not isinstance(schema, dict):
            raise ValueError(f"Inventory record schema must be an object: {name}")
        records.append(ToolInventoryRecord(name=name, toolset=toolset, description=description, schema=schema))
    return records


def generate_candidate_descriptions(
    records: Sequence[ToolInventoryRecord],
    cases: Sequence[ToolSelectionCase] | None = None,
    *,
    max_description_chars: int = 500,
) -> list[ToolDescriptionCandidate]:
    """Generate deterministic Phase 2C candidate descriptions from golden cases."""

    eval_cases = tuple(cases) if cases is not None else default_tool_selection_cases()
    candidates: list[ToolDescriptionCandidate] = []
    for record in records:
        relevant_cases = tuple(case for case in eval_cases if case.expected_tool == record.name)
        baseline = record.description.strip()
        candidate = _candidate_description_for_tool(baseline, relevant_cases, max_description_chars)
        candidates.append(
            ToolDescriptionCandidate(
                name=record.name,
                toolset=record.toolset,
                baseline_description=baseline,
                candidate_description=candidate,
                parameter_descriptions=_extract_parameter_descriptions(record.schema),
            )
        )
    return candidates


def run_candidate_generation(
    *,
    inventory_json: str | Path | None = None,
    hermes_repo: str | Path | None = None,
    output_dir: str | Path | None = None,
    cases: Sequence[ToolSelectionCase] | None = None,
) -> CandidateGenerationResult:
    """Run Phase 2C/2D in candidate-only mode and write review artifacts."""

    records = load_inventory_from_json(inventory_json) if inventory_json else collect_hermes_tool_inventory(hermes_repo)
    eval_cases = tuple(cases) if cases is not None else default_tool_selection_cases()
    candidates = generate_candidate_descriptions(records, eval_cases)

    target_dir = Path(output_dir) if output_dir else Path("output") / "tool-description" / datetime.now().strftime("%Y%m%d_%H%M%S_phase2d")
    target_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = target_dir / "inventory.json"
    candidates_path = target_dir / "candidate_descriptions.json"
    report_path = target_dir / "candidate_only_report.json"
    diff_path = target_dir / "candidate.diff"

    inventory_path.write_text(json.dumps([asdict(record) for record in records], indent=2, sort_keys=True) + "\n")
    candidates_path.write_text(
        json.dumps(
            [asdict(candidate) | {"description_delta": candidate.description_delta} for candidate in candidates],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    report = build_candidate_only_report(candidates, eval_cases)
    phase2d_gate = evaluate_cross_tool_gate(candidates_from_inventory(records), candidates, eval_cases)
    report.update(
        {
            "phase": "2D",
            "summary": "Candidate-only tool description generation plus formal Phase 2D cross-tool gate; active Hermes tool schemas are not modified.",
            "phase_index_executed": ["2A", "2B", "2C", "2D"],
            "phase2d_gate": phase2d_gate.to_dict(),
            "artifacts": {
                "inventory": str(inventory_path),
                "candidates": str(candidates_path),
                "diff": str(diff_path),
            },
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    diff_path.write_text(_candidate_diff(candidates))

    return CandidateGenerationResult(
        output_dir=target_dir,
        inventory_path=inventory_path,
        candidates_path=candidates_path,
        report_path=report_path,
        diff_path=diff_path,
        phase2d_gate_passed=phase2d_gate.passed,
        phase2d_failed_checks=phase2d_gate.failed_checks,
    )


def _candidate_description_for_tool(
    baseline: str,
    cases: Sequence[ToolSelectionCase],
    max_description_chars: int,
) -> str:
    if not cases:
        return _fit_description(baseline, max_description_chars)

    confusing_tools = _unique_in_order(tool for case in cases for tool in case.confusing_tools)
    cues = _unique_in_order(cue for case in cases for cue in (*case.required_cues, *case.required_arguments))
    cue_text = ", ".join(cues[:8])
    confusing_text = ", ".join(confusing_tools[:6])

    suffix_parts = []
    if cue_text:
        suffix_parts.append(f"Use for cues: {cue_text}")
    if confusing_text:
        suffix_parts.append(f"Prefer over {confusing_text}")

    suffix = "; ".join(suffix_parts)
    if not suffix:
        return _fit_description(baseline, max_description_chars)
    return _fit_description_with_suffix(baseline, f"{suffix}.", max_description_chars)


def _fit_description_with_suffix(description: str, suffix: str, max_chars: int) -> str:
    """Fit a description while preserving generated disambiguation cues."""

    normalized = " ".join(description.split())
    normalized_suffix = " ".join(suffix.split())
    candidate = f"{normalized.rstrip()} {normalized_suffix}".strip()
    if len(candidate) <= max_chars:
        return candidate

    suffix_with_separator = f" {normalized_suffix}"
    if len(suffix_with_separator) >= max_chars:
        return _fit_description(normalized_suffix, max_chars)

    baseline_budget = max_chars - len(suffix_with_separator)
    return f"{_fit_description(normalized, baseline_budget)}{suffix_with_separator}"


def _fit_description(description: str, max_chars: int) -> str:
    normalized = " ".join(description.split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 1:
        return normalized[:max_chars]
    return normalized[: max_chars - 1].rstrip(" ,;.") + "…"


def _candidate_diff(candidates: Sequence[ToolDescriptionCandidate]) -> str:
    chunks: list[str] = []
    for candidate in candidates:
        if candidate.baseline_description == candidate.candidate_description:
            continue
        chunks.extend(
            difflib.unified_diff(
                [candidate.baseline_description + "\n"],
                [candidate.candidate_description + "\n"],
                fromfile=f"{candidate.name}/baseline_description",
                tofile=f"{candidate.name}/candidate_description",
                lineterm="",
            )
        )
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n" if chunks else "No candidate description changes.\n"


def _extract_parameter_descriptions(schema: Mapping[str, object]) -> dict[str, str]:
    parameters = schema.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return {}
    descriptions: dict[str, str] = {}
    for name, spec in properties.items():
        if isinstance(name, str) and isinstance(spec, dict) and isinstance(spec.get("description"), str):
            descriptions[name] = spec["description"]
    return descriptions


def _unique_in_order(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


@click.command()
@click.option("--inventory-json", default=None, type=click.Path(exists=True, dir_okay=False), help="Read a saved inventory JSON instead of importing Hermes tools")
@click.option(
    "--hermes-repo",
    default=None,
    type=click.Path(file_okay=False),
    help="Trusted hermes-agent checkout for read-only inventory collection; tool modules are imported",
)
@click.option("--output-dir", default=None, type=click.Path(file_okay=False), help="Directory for candidate-only artifacts")
def main(inventory_json: str | None, hermes_repo: str | None, output_dir: str | None):
    """Generate Phase 2C candidates and Phase 2D gate artifacts without applying them."""

    result = run_candidate_generation(
        inventory_json=inventory_json,
        hermes_repo=hermes_repo,
        output_dir=output_dir,
    )
    console.print("[bold green]Phase 2D candidate-only artifacts written[/bold green]")
    console.print(f"  output: {result.output_dir}")
    console.print(f"  report: {result.report_path}")
    console.print("  active Hermes tool schemas modified: no")
    if result.phase2d_gate_passed:
        console.print("  Phase 2D gate: passed")
        return

    console.print("  Phase 2D gate: failed")
    for failed_check in result.phase2d_failed_checks:
        console.print(f"    - {failed_check}")
    raise click.ClickException(f"Phase 2D gate failed; inspect report: {result.report_path}")


if __name__ == "__main__":
    main()
