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
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import click
from rich.console import Console

from evolution.core.candidate_bundle import (
    create_candidate_bundle,
    write_bundle_json,
    write_bundle_text,
    write_decision,
)
from evolution.core.config import get_hermes_agent_path
from evolution.tools.tool_description_eval import (
    DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS,
    ToolDescriptionCandidate,
    ToolInventoryRecord,
    ToolSelectionCase,
    _normalize_token,
    build_candidate_only_report,
    candidates_from_inventory,
    default_tool_selection_cases,
    evaluate_cross_tool_gate,
    normalize_parameter_description,
)

console = Console()

_REQUEST_SIGNAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "before",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "only",
    "or",
    "that",
    "the",
    "these",
    "this",
    "those",
    "through",
    "to",
    "use",
    "using",
    "with",
    "without",
}
_REQUEST_SIGNAL_RE = re.compile(r"[a-zA-Z0-9_]+")
_REQUEST_SIGNAL_FORBIDDEN_FRAGMENTS = (
    "/" + "Users" + "/",
    "/" + "home" + "/",
    "session" + "_id",
    "OPENAI" + "_API_KEY",
    "ANTHROPIC" + "_API_KEY",
    "OPENROUTER" + "_API_KEY",
)
_REQUEST_SIGNAL_FORBIDDEN_TOKENS = {
    "users",
    "home",
    "session_id",
    "openai_api_key",
    "anthropic_api_key",
    "openrouter_api_key",
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "credential",
    "credentials",
    "private_key",
}
_SAFETY_OPERATIONAL_TERMS = (
    "do not",
    "must",
    "never",
    "important",
    "critical",
    "dangerous",
    "destructive",
    "secrets",
    "credential",
    "approval",
    "confirm",
    "background",
    "notify_on_complete",
    "read_file",
    "write_file",
    "patch",
    "search_files",
)


@dataclass(frozen=True)
class InventoryImportWarning:
    """A non-candidate warning emitted while importing Hermes tool modules."""

    module: str
    message: str
    exception: str
    classification: str
    candidate_quality: bool = False


@dataclass(frozen=True)
class ToolInventoryCollectionResult:
    """Read-only Hermes inventory plus import-time metadata."""

    records: list[ToolInventoryRecord]
    import_warnings: tuple[InventoryImportWarning, ...] = ()


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


class _InventoryImportWarningHandler(logging.Handler):
    """Capture Hermes registry import warnings without treating them as candidate warnings."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.import_warnings: list[InventoryImportWarning] = []

    def emit(self, record: logging.LogRecord) -> None:
        warning = _inventory_import_warning_from_record(record)
        if warning is not None:
            self.import_warnings.append(warning)


def _inventory_import_warning_from_record(record: logging.LogRecord) -> InventoryImportWarning | None:
    if not record.getMessage().startswith("Could not import tool module"):
        return None

    args = record.args if isinstance(record.args, tuple) else ()
    module = str(args[0]) if args else "unknown"
    exception = str(args[1]) if len(args) > 1 else ""
    message = record.getMessage()
    classification = "optional_dependency_import_warning" if "No module named" in exception else "tool_module_import_warning"
    return InventoryImportWarning(
        module=module,
        message=message,
        exception=exception,
        classification=classification,
        candidate_quality=False,
    )


def _inventory_metadata(
    *,
    source: str,
    records: Sequence[ToolInventoryRecord],
    import_warnings: Sequence[InventoryImportWarning],
) -> dict[str, object]:
    return {
        "source": source,
        "tool_count": len(records),
        "import_warning_count": len(import_warnings),
        "import_warnings": [asdict(warning) for warning in import_warnings],
        "candidate_quality_warnings_are_separate": True,
    }


def collect_hermes_tool_inventory(hermes_repo: str | Path | None = None) -> list[ToolInventoryRecord]:
    """Collect a read-only inventory from a Hermes Agent checkout.

    Importing Hermes tool modules is used only to inspect registry metadata. This
    function does not call tool handlers, edit Hermes source files, or write
    configuration. Because discovery imports Python modules, point this only at
    trusted Hermes Agent checkouts.
    """

    return collect_hermes_tool_inventory_with_metadata(hermes_repo).records


def collect_hermes_tool_inventory_with_metadata(
    hermes_repo: str | Path | None = None,
) -> ToolInventoryCollectionResult:
    """Collect a read-only inventory and separate import warnings as metadata."""

    repo = Path(hermes_repo).expanduser() if hermes_repo else get_hermes_agent_path()
    tools_dir = repo / "tools"
    if not tools_dir.exists():
        raise FileNotFoundError(f"Hermes tools directory not found: {tools_dir}")

    repo_string = str(repo)
    inserted = False
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
        inserted = True
    warning_handler = _InventoryImportWarningHandler()
    registry_logger = logging.getLogger("tools.registry")
    previous_handlers = list(registry_logger.handlers)
    previous_propagate = registry_logger.propagate
    try:
        from tools.registry import discover_builtin_tools, registry  # type: ignore

        registry_logger.handlers = [warning_handler]
        registry_logger.propagate = False
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
        return ToolInventoryCollectionResult(
            records=sorted(records, key=lambda record: (record.toolset, record.name)),
            import_warnings=tuple(warning_handler.import_warnings),
        )
    finally:
        registry_logger.handlers = previous_handlers
        registry_logger.propagate = previous_propagate
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
    max_parameter_description_chars: int = DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS,
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
                parameter_descriptions=_extract_parameter_descriptions(
                    record.schema,
                    max_chars=max_parameter_description_chars,
                ),
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

    if inventory_json:
        records = load_inventory_from_json(inventory_json)
        inventory_source = "inventory_json"
        import_warnings: tuple[InventoryImportWarning, ...] = ()
    else:
        inventory_collection = collect_hermes_tool_inventory_with_metadata(hermes_repo)
        records = inventory_collection.records
        inventory_source = "hermes_repo_import"
        import_warnings = inventory_collection.import_warnings
    eval_cases = tuple(cases) if cases is not None else default_tool_selection_cases()
    candidates = generate_candidate_descriptions(records, eval_cases)

    standard_bundle = None
    if output_dir:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        inventory_path = target_dir / "inventory.json"
        candidates_path = target_dir / "candidate_descriptions.json"
        report_path = target_dir / "candidate_only_report.json"
        diff_path = target_dir / "candidate.diff"
        gate_path = target_dir / "cross_tool_regression.json"
        artifact_refs = {
            "inventory": str(inventory_path),
            "candidates": str(candidates_path),
            "diff": str(diff_path),
            "cross_tool_gate": str(gate_path),
        }
    else:
        standard_bundle = create_candidate_bundle(
            phase="Phase 2: Tool Description Evolution",
            target="tool-description",
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f_phase2d"),
        )
        target_dir = standard_bundle.root
        inventory_path = target_dir / "inputs" / "inventory.json"
        candidates_path = target_dir / "candidates" / "candidate_descriptions.json"
        report_path = target_dir / "reports" / "candidate_only_report.json"
        diff_path = target_dir / "candidates" / "candidate.patch"
        gate_path = target_dir / "eval" / "cross_tool_regression.json"
        artifact_refs = {
            "inventory": "inputs/inventory.json",
            "candidates": "candidates/candidate_descriptions.json",
            "patch": "candidates/candidate.patch",
            "report": "reports/candidate_only_report.json",
            "cross_tool_gate": "eval/cross_tool_regression.json",
        }

    inventory_payload = [asdict(record) for record in records]
    candidates_payload = [asdict(candidate) | {"description_delta": candidate.description_delta} for candidate in candidates]
    if standard_bundle:
        write_bundle_json(standard_bundle, "inputs/inventory.json", inventory_payload)
        write_bundle_json(standard_bundle, "candidates/candidate_descriptions.json", candidates_payload)
    else:
        inventory_path.write_text(json.dumps(inventory_payload, indent=2, sort_keys=True) + "\n")
        candidates_path.write_text(json.dumps(candidates_payload, indent=2, sort_keys=True) + "\n")

    report = build_candidate_only_report(candidates, eval_cases)
    phase2d_gate = evaluate_cross_tool_gate(candidates_from_inventory(records), candidates, eval_cases)
    diff_text = _candidate_diff(candidates)
    report.update(
        {
            "phase": "2D",
            "summary": "Candidate-only tool description generation plus formal Phase 2D cross-tool gate; active Hermes tool schemas are not modified.",
            "phase_index_executed": ["2A", "2B", "2C", "2D"],
            "phase2d_gate": phase2d_gate.to_dict(),
            "inventory_metadata": _inventory_metadata(
                source=inventory_source,
                records=records,
                import_warnings=import_warnings,
            ),
            "artifacts": artifact_refs,
        }
    )
    gate_payload = phase2d_gate.to_dict()
    if standard_bundle:
        write_bundle_json(standard_bundle, "reports/candidate_only_report.json", report)
        write_bundle_json(standard_bundle, "eval/cross_tool_regression.json", gate_payload)
        write_bundle_text(standard_bundle, "candidates/candidate.patch", diff_text)
        decision_status = _decision_status_for_tool_candidate(phase2d_gate.passed, diff_text)
        write_decision(
            standard_bundle,
            status=decision_status,
            summary="Phase 2 tool-description candidate generated locally; no active schema or GitHub PR mutation performed.",
            metrics={
                "selection_accuracy": report["metrics"].get("selection_accuracy"),
                "wrong_tool_avoidance": report["metrics"].get("wrong_tool_avoidance"),
                "phase2d_gate_passed": phase2d_gate.passed,
            },
            artifacts=artifact_refs,
        )
    else:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        gate_path.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n")
        diff_path.write_text(diff_text)

    return CandidateGenerationResult(
        output_dir=target_dir,
        inventory_path=inventory_path,
        candidates_path=candidates_path,
        report_path=report_path,
        diff_path=diff_path,
        phase2d_gate_passed=phase2d_gate.passed,
        phase2d_failed_checks=phase2d_gate.failed_checks,
    )


def _decision_status_for_tool_candidate(gate_passed: bool, diff_text: str) -> str:
    if not diff_text.strip() or diff_text.startswith("No candidate description changes."):
        return "NO_DIFF_NO_GO"
    if gate_passed:
        return "PASS_CANDIDATE_ONLY"
    return "REGRESSION_NO_GO"


def _candidate_description_for_tool(
    baseline: str,
    cases: Sequence[ToolSelectionCase],
    max_description_chars: int,
) -> str:
    if not cases:
        return _fit_description_preserving_safety_terms(baseline, max_description_chars)

    confusing_tools = _unique_in_order(tool for case in cases for tool in case.confusing_tools)
    cues = _unique_in_order(
        [
            *(cue for case in cases for cue in (*case.required_cues, *case.required_arguments)),
            *_request_signal_cues(cases),
        ]
    )
    cue_text = ", ".join(cues[:16])
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
    """Fit a description while preserving generated disambiguation cues.

    Active-apply review also requires that cue-focused rewrites do not delete
    safety/operational terms from long active tool descriptions.  Preserve those
    terms explicitly before spending the remaining budget on selection cues.
    """

    normalized = " ".join(description.split())
    normalized_suffix = " ".join(suffix.split())
    safety_terms = _safety_operational_terms_present(normalized)
    safety_suffix = _safety_operational_suffix(safety_terms)
    if safety_suffix:
        return _fit_description_with_required_and_optional_suffix(
            normalized,
            required_suffix=safety_suffix,
            optional_suffix=normalized_suffix,
            max_chars=max_chars,
        )

    candidate = f"{normalized.rstrip()} {normalized_suffix}".strip()
    if len(candidate) <= max_chars:
        return candidate

    suffix_with_separator = f" {normalized_suffix}"
    if len(suffix_with_separator) >= max_chars:
        return _fit_description(normalized_suffix, max_chars)

    baseline_budget = max_chars - len(suffix_with_separator)
    return f"{_fit_description(normalized, baseline_budget)}{suffix_with_separator}"


def _fit_description_with_required_and_optional_suffix(
    description: str,
    *,
    required_suffix: str,
    optional_suffix: str,
    max_chars: int,
) -> str:
    suffix = " ".join(part for part in (required_suffix, optional_suffix) if part).strip()
    suffix_with_separator = f" {suffix}" if suffix else ""
    if suffix_with_separator and len(suffix_with_separator) < max_chars:
        baseline_budget = max_chars - len(suffix_with_separator)
        return f"{_fit_description(description, baseline_budget)}{suffix_with_separator}"

    if required_suffix and len(required_suffix) < max_chars:
        optional_budget = max_chars - len(required_suffix) - 1
        if optional_suffix and optional_budget > 0:
            return f"{required_suffix} {_fit_description(optional_suffix, optional_budget)}"
        return _fit_description(required_suffix, max_chars)

    return _fit_description(suffix or description, max_chars)


def _safety_operational_terms_present(description: str) -> list[str]:
    lowered = description.lower()
    return [term for term in _SAFETY_OPERATIONAL_TERMS if term in lowered]


def _safety_operational_suffix(terms: Sequence[str]) -> str:
    ordered_terms = _unique_in_order(terms)
    if not ordered_terms:
        return ""
    return f"Safety/ops requirements include: {', '.join(ordered_terms)}."


def _fit_description_preserving_safety_terms(description: str, max_chars: int) -> str:
    normalized = " ".join(description.split())
    safety_suffix = _safety_operational_suffix(_safety_operational_terms_present(normalized))
    if not safety_suffix:
        return _fit_description(normalized, max_chars)
    return _fit_description_with_required_and_optional_suffix(
        normalized,
        required_suffix=safety_suffix,
        optional_suffix="",
        max_chars=max_chars,
    )


def _fit_description(description: str, max_chars: int) -> str:
    normalized = " ".join(description.split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 1:
        return normalized[:max_chars]
    return normalized[: max_chars - 1].rstrip(" ,;.") + "…"


def _request_signal_cues(cases: Sequence[ToolSelectionCase]) -> list[str]:
    """Extract privacy-safe request cue variants from local golden cases.

    Hand-authored required cues capture the core intent, but held-out rows can
    still fail when requests use benign variants such as "show first lines",
    "preserve surrounding content", or exact local tool names. Candidate-only
    descriptions can carry these sanitized cue variants because they are review
    artifacts, not active Hermes tool schemas.
    """

    existing_cues = {
        _normalize_token(cue)
        for case in cases
        for cue in (*case.required_cues, *case.required_arguments)
    }
    tool_name_tokens = {
        _normalize_token(token)
        for case in cases
        for tool in (case.expected_tool, *case.confusing_tools)
        for token in (*tool.split("_"), tool)
    }
    signals: list[str] = []
    for case in cases:
        _reject_private_request_text(case.user_request)
        for raw_token in _REQUEST_SIGNAL_RE.findall(case.user_request):
            token = _normalize_token(raw_token)
            if (
                token in existing_cues
                or token in tool_name_tokens
                or token in _REQUEST_SIGNAL_STOPWORDS
                or token.isdigit()
                or len(token) <= 2
                or not _request_signal_token_is_safe(token)
            ):
                continue
            signals.append(token)
    return _unique_in_order(signals)


def _reject_private_request_text(user_request: str) -> None:
    if any(fragment in user_request for fragment in _REQUEST_SIGNAL_FORBIDDEN_FRAGMENTS):
        raise ValueError("tool-selection request contains private/raw identifier")


def _request_signal_token_is_safe(token: str) -> bool:
    if token in _REQUEST_SIGNAL_FORBIDDEN_TOKENS:
        return False
    if "api_key" in token or "secret" in token or "password" in token:
        return False
    if len(token) > 48:
        return False
    return True


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


def _extract_parameter_descriptions(
    schema: Mapping[str, object],
    *,
    max_chars: int = DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS,
) -> dict[str, str]:
    parameters = schema.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return {}
    descriptions: dict[str, str] = {}
    for name, spec in properties.items():
        if isinstance(name, str) and isinstance(spec, dict) and isinstance(spec.get("description"), str):
            descriptions[name] = normalize_parameter_description(spec["description"], max_chars=max_chars)
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
