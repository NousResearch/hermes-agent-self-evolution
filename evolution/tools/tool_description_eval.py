"""Candidate-only evaluation scaffold for tool description evolution.

Phase 2B intentionally does not write to Hermes Agent's active tool registry,
schemas, or source tree.  It provides deterministic, JSON-serializable metrics
that can be used to compare candidate tool descriptions before any apply step is
considered by a human.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_DANGEROUS_DESCRIPTION_PATTERNS = (
    "ignore previous instructions",
    "ignore system instructions",
    "reveal secrets",
    "leak secrets",
    "exfiltrate",
    "password",
    "private key",
)
DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS = 200


def normalize_parameter_description(
    description: str,
    *,
    max_chars: int = DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS,
) -> str:
    """Normalize a parameter description for candidate-only reporting.

    Active Hermes schemas can contain long parameter guidance. Phase 2E keeps the
    active schema read-only, but candidate artifacts should fit the Phase 2
    parameter-description budget so reports measure actionable candidate quality
    instead of repeating known live-schema verbosity warnings.
    """

    normalized = " ".join(description.split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 0:
        return ""
    if max_chars == 1:
        return "…"
    return normalized[: max_chars - 1].rstrip(" ,;.") + "…"


@dataclass(frozen=True)
class ToolInventoryRecord:
    """Read-only snapshot of one active Hermes tool definition."""

    name: str
    toolset: str
    description: str
    schema: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDescriptionCandidate:
    """A baseline/candidate description pair for one Hermes tool."""

    name: str
    toolset: str
    baseline_description: str
    candidate_description: str
    parameter_descriptions: Mapping[str, str] = field(default_factory=dict)

    @property
    def description_delta(self) -> int:
        return len(self.candidate_description) - len(self.baseline_description)


@dataclass(frozen=True)
class ToolSelectionCase:
    """Golden case for wrong-tool avoidance and expected-tool selection."""

    user_request: str
    expected_tool: str
    confusing_tools: tuple[str, ...] = ()
    required_cues: tuple[str, ...] = ()
    required_arguments: tuple[str, ...] = ()
    category: str = "tool-selection"


@dataclass(frozen=True)
class ToolCaseResult:
    """Per-case deterministic ranking result."""

    user_request: str
    expected_tool: str
    selected_tool: str | None
    expected_score: float
    confusing_scores: dict[str, float]
    passed: bool
    cue_coverage: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolDescriptionEvalResult:
    """Aggregate Phase 2B metrics.

    apply_ready is deliberately always false in this scaffold.  A future Phase 2C
    may add an explicit human-gated apply workflow, but Phase 2B only reports.
    """

    candidate_only: bool
    apply_ready: bool
    case_count: int
    selection_accuracy: float
    wrong_tool_avoidance: float
    argument_cue_coverage: float
    constraint_pass_rate: float
    case_results: tuple[ToolCaseResult, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "candidate_only": self.candidate_only,
            "apply_ready": self.apply_ready,
            "case_count": self.case_count,
            "selection_accuracy": self.selection_accuracy,
            "wrong_tool_avoidance": self.wrong_tool_avoidance,
            "argument_cue_coverage": self.argument_cue_coverage,
            "constraint_pass_rate": self.constraint_pass_rate,
            "case_results": [asdict(case) for case in self.case_results],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CrossToolGateThresholds:
    """Formal Phase 2D cross-tool gate thresholds.

    The gate is intentionally candidate-only: it can reject a candidate or mark it
    review-worthy, but it never creates an apply payload.
    """

    min_case_count: int = 45
    min_selection_accuracy: float = 0.70
    min_wrong_tool_avoidance: float = 0.70
    max_per_tool_regression: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CrossToolRegression:
    """Per expected-tool pass-rate comparison against the baseline."""

    expected_tool: str
    case_count: int
    baseline_pass_count: int
    candidate_pass_count: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta: float
    passed: bool


@dataclass(frozen=True)
class CrossToolGateResult:
    """Formal Phase 2D pass/fail result for a candidate description set."""

    phase: str
    candidate_only: bool
    passed: bool
    thresholds: CrossToolGateThresholds
    baseline_metrics: Mapping[str, object]
    candidate_metrics: Mapping[str, object]
    per_tool_regressions: tuple[CrossToolRegression, ...]
    failed_checks: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "candidate_only": self.candidate_only,
            "passed": self.passed,
            "thresholds": self.thresholds.to_dict(),
            "baseline_metrics": dict(self.baseline_metrics),
            "candidate_metrics": dict(self.candidate_metrics),
            "per_tool_regressions": [asdict(regression) for regression in self.per_tool_regressions],
            "failed_checks": list(self.failed_checks),
        }


def default_tool_selection_cases() -> tuple[ToolSelectionCase, ...]:
    """Small hand-curated golden set for common Hermes tool confusions."""

    return (
        ToolSelectionCase(
            user_request="Show the first 40 lines of README.md without using a shell pager.",
            expected_tool="read_file",
            confusing_tools=("terminal", "search_files"),
            required_cues=("read", "file", "line", "cat", "head", "tail"),
            category="file-read-vs-shell",
        ),
        ToolSelectionCase(
            user_request="Find Python files mentioning browser_navigate in the tools directory.",
            expected_tool="search_files",
            confusing_tools=("terminal", "read_file"),
            required_cues=("search", "files", "grep", "rg", "find"),
            category="search-vs-shell",
        ),
        ToolSelectionCase(
            user_request="Run the focused pytest target for the new tool evaluation tests.",
            expected_tool="terminal",
            confusing_tools=("execute_code", "read_file"),
            required_cues=("pytest", "command", "shell", "build", "test"),
            category="shell-execution",
        ),
        ToolSelectionCase(
            user_request="Make a targeted replacement in one Python file and preserve surrounding content.",
            expected_tool="patch",
            confusing_tools=("write_file", "terminal"),
            required_cues=("replace", "targeted", "edit", "diff", "sed"),
            category="edit-vs-overwrite",
        ),
        ToolSelectionCase(
            user_request="Overwrite a new Markdown report with complete content.",
            expected_tool="write_file",
            confusing_tools=("patch", "terminal"),
            required_cues=("write", "overwrite", "complete", "file"),
            category="write-vs-patch",
        ),
        ToolSelectionCase(
            user_request="Search prior conversations about release update policy.",
            expected_tool="session_search",
            confusing_tools=("web_search", "search_files"),
            required_cues=("past", "sessions", "conversation", "history"),
            category="session-vs-web-search",
        ),
        ToolSelectionCase(
            user_request="Find the previous chat where we discussed the HSE Phase 2B evaluator and summarize where we left off.",
            expected_tool="session_search",
            confusing_tools=("browser_navigate", "web_search", "search_files"),
            required_cues=("past", "previous", "session", "conversation", "left"),
            category="session-vs-browser-history",
        ),
        ToolSelectionCase(
            user_request="Open the web app and click through the login-free settings screen to see what changes dynamically.",
            expected_tool="browser_navigate",
            confusing_tools=("session_search", "web_search", "read_file"),
            required_cues=("navigate", "open", "click", "dynamic", "web"),
            category="browser-vs-session-search",
        ),
        ToolSelectionCase(
            user_request="After navigating, capture the current page structure and available buttons.",
            expected_tool="browser_snapshot",
            confusing_tools=("browser_vision", "read_file"),
            required_cues=("snapshot", "page", "buttons", "structure"),
            category="browser-snapshot-vs-vision",
        ),
        ToolSelectionCase(
            user_request="Click the Save button already visible in the browser snapshot.",
            expected_tool="browser_click",
            confusing_tools=("browser_type", "computer_use", "terminal"),
            required_cues=("click", "button", "snapshot", "ref"),
            category="browser-click-vs-desktop",
        ),
        ToolSelectionCase(
            user_request="Check the browser console for JavaScript errors after the failed form submission.",
            expected_tool="browser_console",
            confusing_tools=("terminal", "search_files", "browser_snapshot"),
            required_cues=("console", "javascript", "errors", "page"),
            category="browser-console-vs-shell",
        ),
        ToolSelectionCase(
            user_request="Use the macOS app window in the background to press the visible Export button.",
            expected_tool="computer_use",
            confusing_tools=("browser_click", "terminal"),
            required_cues=("macos", "background", "window", "click", "desktop"),
            category="desktop-vs-browser",
        ),
        ToolSelectionCase(
            user_request="Calculate the exact total cost from these numeric line items.",
            expected_tool="execute_code",
            confusing_tools=("terminal", "read_file"),
            required_cues=("calculate", "math", "python", "exact"),
            category="calculation-vs-shell",
        ),
        ToolSelectionCase(
            user_request="Run git status and the focused pytest command in this repository.",
            expected_tool="terminal",
            confusing_tools=("execute_code", "search_files"),
            required_cues=("git", "pytest", "command", "repository"),
            category="repo-command-vs-python",
        ),
        ToolSelectionCase(
            user_request="Find every file named config.yaml below the project without shell find.",
            expected_tool="search_files",
            confusing_tools=("terminal", "read_file"),
            required_cues=("find", "files", "glob", "name"),
            category="file-search-vs-shell-find",
        ),
        ToolSelectionCase(
            user_request="Read only lines 120 through 180 of gateway/run.py.",
            expected_tool="read_file",
            confusing_tools=("terminal", "search_files"),
            required_cues=("read", "lines", "offset", "limit"),
            category="file-read-range-vs-shell",
        ),
        ToolSelectionCase(
            user_request="Apply a small unique string replacement in SKILL.md and show the diff.",
            expected_tool="patch",
            confusing_tools=("write_file", "terminal"),
            required_cues=("replace", "unique", "diff", "patch"),
            category="patch-vs-overwrite",
        ),
        ToolSelectionCase(
            user_request="Create a brand new complete Markdown artifact with the supplied full text.",
            expected_tool="write_file",
            confusing_tools=("patch", "terminal"),
            required_cues=("create", "complete", "markdown", "overwrite"),
            category="write-complete-artifact",
        ),
        ToolSelectionCase(
            user_request="Track these four implementation steps and mark the current one in progress.",
            expected_tool="todo",
            confusing_tools=("write_file", "session_search"),
            required_cues=("track", "steps", "in_progress", "task"),
            category="todo-vs-notes",
        ),
        ToolSelectionCase(
            user_request="Split three independent code inspection subtasks across isolated workers and summarize their findings.",
            expected_tool="delegate_task",
            confusing_tools=("terminal", "todo", "cronjob"),
            required_cues=("subtasks", "workers", "parallel", "isolated"),
            category="delegation-vs-shell",
        ),
        ToolSelectionCase(
            user_request="Schedule a self-contained reminder to check the release feed every two hours.",
            expected_tool="cronjob",
            confusing_tools=("todo", "terminal", "send_message"),
            required_cues=("schedule", "every", "recurring", "job"),
            category="cron-vs-todo",
        ),
        ToolSelectionCase(
            user_request="Analyze the screenshot file and tell me what text is visible.",
            expected_tool="vision_analyze",
            confusing_tools=("read_file", "browser_vision", "browser_snapshot"),
            required_cues=("image", "screenshot", "visible", "analyze"),
            category="vision-vs-file-read",
        ),
        ToolSelectionCase(
            user_request="From the current browser page, take a screenshot and inspect the visual layout.",
            expected_tool="browser_vision",
            confusing_tools=("vision_analyze", "browser_snapshot"),
            required_cues=("browser", "screenshot", "visual", "page"),
            category="browser-vision-vs-image-file",
        ),
        ToolSelectionCase(
            user_request="Send this final status update to the configured Discord home channel.",
            expected_tool="send_message",
            confusing_tools=("cronjob", "write_file"),
            required_cues=("send", "discord", "message", "channel"),
            category="message-send-vs-file",
        ),
        ToolSelectionCase(
            user_request="Remember that the user prefers release-tag updates as a durable preference.",
            expected_tool="memory",
            confusing_tools=("write_file", "session_search", "todo"),
            required_cues=("remember", "durable", "preference", "user"),
            category="memory-vs-session-search",
        ),
        ToolSelectionCase(
            user_request="Use the visible native macOS menu bar item in Safari's window, without relying on browser DOM refs.",
            expected_tool="computer_use",
            confusing_tools=("browser_click", "browser_snapshot", "terminal"),
            required_cues=("macos", "native", "menu", "window", "background"),
            category="computer-use-vs-browser-dom",
        ),
        ToolSelectionCase(
            user_request="From this pasted JSON array, compute grouped totals and return exact percentages.",
            expected_tool="execute_code",
            confusing_tools=("terminal", "read_file"),
            required_cues=("json", "compute", "grouped", "totals", "percentages"),
            category="execute-code-vs-terminal-calculation",
        ),
        ToolSelectionCase(
            user_request="Install the project dependencies and run the npm build script.",
            expected_tool="terminal",
            confusing_tools=("execute_code", "patch"),
            required_cues=("install", "dependencies", "npm", "build", "shell"),
            category="terminal-package-manager-vs-python",
        ),
        ToolSelectionCase(
            user_request="Find the prior conversation where I defined what '안전모드로 진행해' means.",
            expected_tool="session_search",
            confusing_tools=("search_files", "memory", "browser_navigate"),
            required_cues=("prior", "conversation", "defined", "means", "session"),
            category="session-search-vs-memory-file-search",
        ),
        ToolSelectionCase(
            user_request="Inspect the current loaded browser page's accessible structure before deciding what to click.",
            expected_tool="browser_snapshot",
            confusing_tools=("computer_use", "vision_analyze", "browser_click"),
            required_cues=("browser", "accessible", "structure", "snapshot", "click"),
            category="browser-snapshot-vs-computer-use",
        ),
        ToolSelectionCase(
            user_request="Type the supplied email into the focused input field from the current browser snapshot.",
            expected_tool="browser_type",
            confusing_tools=("computer_use", "browser_click", "terminal"),
            required_cues=("type", "input", "field", "browser", "text", "ref"),
            category="browser-type-vs-desktop",
        ),
        ToolSelectionCase(
            user_request="Press Enter in the browser page to submit the focused search form.",
            expected_tool="browser_press",
            confusing_tools=("browser_click", "computer_use", "terminal"),
            required_cues=("press", "enter", "key", "browser", "focused"),
            category="browser-press-vs-click",
        ),
        ToolSelectionCase(
            user_request="Scroll the current browser page down to reveal more content before taking another snapshot.",
            expected_tool="browser_scroll",
            confusing_tools=("computer_use", "browser_snapshot", "terminal"),
            required_cues=("scroll", "browser", "page", "down", "reveal"),
            category="browser-scroll-vs-desktop",
        ),
        ToolSelectionCase(
            user_request="Go back to the previous page in the already-open browser session.",
            expected_tool="browser_back",
            confusing_tools=("browser_navigate", "session_search", "terminal"),
            required_cues=("back", "previous", "browser", "history"),
            category="browser-back-vs-navigate",
        ),
        ToolSelectionCase(
            user_request="List all image URLs and alt text from the current browser page.",
            expected_tool="browser_get_images",
            confusing_tools=("browser_vision", "vision_analyze", "browser_snapshot"),
            required_cues=("images", "urls", "alt", "page", "browser"),
            category="browser-images-vs-vision",
        ),
        ToolSelectionCase(
            user_request="Evaluate document.title in the current page context using the browser console.",
            expected_tool="browser_console",
            confusing_tools=("execute_code", "terminal", "browser_snapshot"),
            required_cues=("console", "javascript", "document", "page", "evaluate"),
            category="browser-console-expression-vs-python",
        ),
        ToolSelectionCase(
            user_request="Capture the native macOS Finder window in the background and identify available controls.",
            expected_tool="computer_use",
            confusing_tools=("browser_snapshot", "browser_vision", "terminal"),
            required_cues=("macos", "native", "background", "capture", "window"),
            category="computer-use-capture-vs-browser-snapshot",
        ),
        ToolSelectionCase(
            user_request="Start the local development server as a tracked background process.",
            expected_tool="terminal",
            confusing_tools=("process", "execute_code", "cronjob"),
            required_cues=("server", "background", "process", "start", "terminal"),
            category="terminal-background-vs-process",
        ),
        ToolSelectionCase(
            user_request="Poll the tracked background test process for new output without starting another command.",
            expected_tool="process",
            confusing_tools=("terminal", "execute_code", "cronjob"),
            required_cues=("poll", "background", "process", "output", "session"),
            category="process-poll-vs-terminal",
        ),
        ToolSelectionCase(
            user_request="Process these CSV rows with Python data structures and print a compact JSON summary.",
            expected_tool="execute_code",
            confusing_tools=("terminal", "read_file"),
            required_cues=("python", "process", "csv", "json", "summary"),
            category="execute-code-data-reduction-vs-terminal",
        ),
        ToolSelectionCase(
            user_request="Scroll forward inside the previously found conversation window around message 12345.",
            expected_tool="session_search",
            confusing_tools=("search_files", "memory", "terminal"),
            required_cues=("session", "scroll", "conversation", "message", "window"),
            category="session-search-scroll-vs-file-search",
        ),
        ToolSelectionCase(
            user_request="Ask me to choose between two deployment targets before taking the irreversible action.",
            expected_tool="clarify",
            confusing_tools=("terminal", "todo", "send_message"),
            required_cues=("ask", "choose", "clarification", "before", "decision"),
            category="clarify-vs-guessing-action",
        ),
        ToolSelectionCase(
            user_request="Generate a square illustration of the proposed architecture from this prompt.",
            expected_tool="image_generate",
            confusing_tools=("vision_analyze", "browser_vision", "write_file"),
            required_cues=("generate", "image", "illustration", "prompt", "square"),
            category="image-generation-vs-vision-analysis",
        ),
        ToolSelectionCase(
            user_request="Convert this final announcement text into an audio voice memo.",
            expected_tool="text_to_speech",
            confusing_tools=("send_message", "write_file", "terminal"),
            required_cues=("audio", "voice", "speech", "text", "memo"),
            category="tts-vs-message-send",
        ),
        ToolSelectionCase(
            user_request="Analyze the local MP4 clip and summarize what happens in the video.",
            expected_tool="video_analyze",
            confusing_tools=("vision_analyze", "browser_vision", "read_file"),
            required_cues=("video", "mp4", "clip", "analyze", "summarize"),
            category="video-analysis-vs-image-analysis",
        ),
    )


def evaluate_candidate_descriptions(
    candidates: Sequence[ToolDescriptionCandidate],
    cases: Sequence[ToolSelectionCase],
    *,
    max_description_chars: int = 500,
    max_parameter_description_chars: int = 200,
) -> ToolDescriptionEvalResult:
    """Evaluate candidates against golden tool-selection cases.

    The scoring is intentionally deterministic and cheap: it is not a substitute
    for model-in-the-loop evaluation, but it provides a stable baseline and
    catches obvious regressions before GEPA/DSPy optimization is introduced.
    """

    candidate_map = {candidate.name: candidate for candidate in candidates}
    case_results: list[ToolCaseResult] = []
    warnings: list[str] = []

    for case in cases:
        result, result_warnings = _evaluate_case(candidate_map, case)
        case_results.append(result)
        warnings.extend(result_warnings)

    if not cases:
        warnings.append("No evaluation cases supplied")

    selection_accuracy = _mean(1.0 if result.passed else 0.0 for result in case_results)
    wrong_tool_avoidance = _mean(
        0.0
        if "missing all confusing tools" in result.notes
        else 1.0
        if all(result.expected_score > score for score in result.confusing_scores.values())
        else 0.0
        for result in case_results
    )
    argument_cue_coverage = _mean(result.cue_coverage for result in case_results)
    candidate_constraint_rate = _constraint_pass_rate(
        candidates,
        max_description_chars=max_description_chars,
        max_parameter_description_chars=max_parameter_description_chars,
        warnings=warnings,
    )
    expected_tool_coverage = _expected_tool_coverage(candidate_map, cases)
    constraint_pass_rate = _mean([candidate_constraint_rate, expected_tool_coverage])

    return ToolDescriptionEvalResult(
        candidate_only=True,
        apply_ready=False,
        case_count=len(cases),
        selection_accuracy=selection_accuracy,
        wrong_tool_avoidance=wrong_tool_avoidance,
        argument_cue_coverage=argument_cue_coverage,
        constraint_pass_rate=constraint_pass_rate,
        case_results=tuple(case_results),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_candidate_only_report(
    candidates: Sequence[ToolDescriptionCandidate],
    cases: Sequence[ToolSelectionCase] | None = None,
) -> dict:
    """Build a JSON-serializable Phase 2B report with no apply payload."""

    eval_cases = tuple(cases) if cases is not None else default_tool_selection_cases()
    result = evaluate_candidate_descriptions(candidates, eval_cases)
    return {
        "phase": "2B",
        "mode": "candidate-only",
        "apply_ready": False,
        "summary": "Tool description evaluation scaffold report; active tool schemas are not modified.",
        "candidate_count": len(candidates),
        "metrics": result.to_dict(),
        "candidates": [asdict(candidate) | {"description_delta": candidate.description_delta} for candidate in candidates],
    }


def evaluate_cross_tool_gate(
    baseline_candidates: Sequence[ToolDescriptionCandidate],
    candidate_candidates: Sequence[ToolDescriptionCandidate],
    cases: Sequence[ToolSelectionCase] | None = None,
    *,
    thresholds: CrossToolGateThresholds | None = None,
) -> CrossToolGateResult:
    """Evaluate the formal Phase 2D cross-tool gate.

    Phase 2D turns the golden tool-selection set into an explicit pass/fail
    barrier. It compares candidate descriptions to baseline descriptions and
    fails closed when aggregate accuracy, wrong-tool avoidance, minimum dataset
    size, or per-tool regression constraints are violated.
    """

    eval_cases = tuple(cases) if cases is not None else default_tool_selection_cases()
    gate_thresholds = thresholds or CrossToolGateThresholds()
    baseline_result = evaluate_candidate_descriptions(baseline_candidates, eval_cases)
    candidate_result = evaluate_candidate_descriptions(candidate_candidates, eval_cases)

    per_tool_regressions = _per_tool_regressions(baseline_result, candidate_result, gate_thresholds)
    failed_checks: list[str] = []
    if candidate_result.case_count < gate_thresholds.min_case_count:
        failed_checks.append(f"case_count {candidate_result.case_count} < {gate_thresholds.min_case_count}")
    if candidate_result.selection_accuracy < gate_thresholds.min_selection_accuracy:
        failed_checks.append(
            f"selection_accuracy {candidate_result.selection_accuracy:.4f} < {gate_thresholds.min_selection_accuracy:.4f}"
        )
    if candidate_result.wrong_tool_avoidance < gate_thresholds.min_wrong_tool_avoidance:
        failed_checks.append(
            f"wrong_tool_avoidance {candidate_result.wrong_tool_avoidance:.4f} < {gate_thresholds.min_wrong_tool_avoidance:.4f}"
        )
    for regression in per_tool_regressions:
        if not regression.passed:
            failed_checks.append(
                "per_tool_regression "
                f"{regression.expected_tool} {regression.delta:.4f} < -{gate_thresholds.max_per_tool_regression:.4f}"
            )
    for warning in candidate_result.warnings:
        if warning.startswith("Dangerous wording constraint failed"):
            failed_checks.append(f"candidate_safety {warning}")

    return CrossToolGateResult(
        phase="2D",
        candidate_only=True,
        passed=not failed_checks,
        thresholds=gate_thresholds,
        baseline_metrics=_metric_snapshot(baseline_result),
        candidate_metrics=_metric_snapshot(candidate_result),
        per_tool_regressions=per_tool_regressions,
        failed_checks=tuple(failed_checks),
    )


def candidates_from_inventory(records: Sequence[ToolInventoryRecord]) -> list[ToolDescriptionCandidate]:
    """Create baseline-equals-candidate entries from a read-only inventory.

    Optimizers can replace candidate_description later, but Phase 2B starts from
    an identity candidate so baseline metrics are measurable without mutation.
    """

    return [
        ToolDescriptionCandidate(
            name=record.name,
            toolset=record.toolset,
            baseline_description=record.description,
            candidate_description=record.description,
            parameter_descriptions=_extract_parameter_descriptions(record.schema),
        )
        for record in records
    ]


def write_default_golden_cases(path: str | Path) -> Path:
    """Write the default tool-selection golden set as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for case in default_tool_selection_cases():
            handle.write(json.dumps(asdict(case), sort_keys=True) + "\n")
    return output_path


def write_candidate_only_report(
    path: str | Path,
    candidates: Sequence[ToolDescriptionCandidate],
    cases: Sequence[ToolSelectionCase] | None = None,
) -> Path:
    """Write a candidate-only report to disk.

    This function writes only the requested report path.  It never writes Hermes
    Agent source, registry, schema, or configuration files.
    """

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_candidate_only_report(candidates, cases)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


def _per_tool_regressions(
    baseline_result: ToolDescriptionEvalResult,
    candidate_result: ToolDescriptionEvalResult,
    thresholds: CrossToolGateThresholds,
) -> tuple[CrossToolRegression, ...]:
    baseline_stats = _pass_stats_by_expected_tool(baseline_result)
    candidate_stats = _pass_stats_by_expected_tool(candidate_result)
    expected_tools = sorted(set(baseline_stats) | set(candidate_stats))
    regressions: list[CrossToolRegression] = []
    for expected_tool in expected_tools:
        baseline_pass_count, baseline_case_count = baseline_stats.get(expected_tool, (0, 0))
        candidate_pass_count, candidate_case_count = candidate_stats.get(expected_tool, (0, 0))
        case_count = max(baseline_case_count, candidate_case_count)
        baseline_rate = _rate(baseline_pass_count, baseline_case_count)
        candidate_rate = _rate(candidate_pass_count, candidate_case_count)
        delta = round(candidate_rate - baseline_rate, 4)
        regressions.append(
            CrossToolRegression(
                expected_tool=expected_tool,
                case_count=case_count,
                baseline_pass_count=baseline_pass_count,
                candidate_pass_count=candidate_pass_count,
                baseline_pass_rate=baseline_rate,
                candidate_pass_rate=candidate_rate,
                delta=delta,
                passed=delta >= -thresholds.max_per_tool_regression,
            )
        )
    return tuple(regressions)


def _pass_stats_by_expected_tool(result: ToolDescriptionEvalResult) -> dict[str, tuple[int, int]]:
    stats: dict[str, list[int]] = {}
    for case_result in result.case_results:
        passed_count, case_count = stats.setdefault(case_result.expected_tool, [0, 0])
        stats[case_result.expected_tool] = [passed_count + int(case_result.passed), case_count + 1]
    return {tool: (counts[0], counts[1]) for tool, counts in stats.items()}


def _metric_snapshot(result: ToolDescriptionEvalResult) -> dict[str, object]:
    return {
        "case_count": result.case_count,
        "selection_accuracy": result.selection_accuracy,
        "wrong_tool_avoidance": result.wrong_tool_avoidance,
        "argument_cue_coverage": result.argument_cue_coverage,
        "constraint_pass_rate": result.constraint_pass_rate,
        "warning_count": len(result.warnings),
    }


def _rate(passed_count: int, case_count: int) -> float:
    if case_count <= 0:
        return 0.0
    return round(passed_count / case_count, 4)


def _evaluate_case(
    candidate_map: Mapping[str, ToolDescriptionCandidate],
    case: ToolSelectionCase,
) -> tuple[ToolCaseResult, list[str]]:
    warnings: list[str] = []
    expected = candidate_map.get(case.expected_tool)
    if expected is None:
        warnings.append(f"Missing expected tool candidate: {case.expected_tool}")
        return ToolCaseResult(
            user_request=case.user_request,
            expected_tool=case.expected_tool,
            selected_tool=None,
            expected_score=0.0,
            confusing_scores={name: 0.0 for name in case.confusing_tools},
            passed=False,
            cue_coverage=0.0,
            notes=("missing expected tool",),
        ), warnings

    expected_score = _rank_score(case.user_request, expected)
    confusing_scores = {
        name: _rank_score(case.user_request, candidate_map[name])
        for name in case.confusing_tools
        if name in candidate_map
    }
    notes: list[str] = []
    for missing in sorted(set(case.confusing_tools) - set(confusing_scores)):
        warnings.append(f"Missing confusing tool candidate: {missing}")
    if case.confusing_tools and not confusing_scores:
        warnings.append(f"No confusing tool candidates available for: {case.expected_tool}")
        notes.append("missing all confusing tools")

    selected_tool = case.expected_tool
    selected_score = expected_score
    for name, score in confusing_scores.items():
        if score > selected_score:
            selected_tool = name
            selected_score = score

    cue_coverage = _cue_coverage(expected, case.required_cues, case.required_arguments)
    passed = (
        selected_tool == case.expected_tool
        and expected_score > max(confusing_scores.values(), default=-1.0)
        and "missing all confusing tools" not in notes
    )
    return ToolCaseResult(
        user_request=case.user_request,
        expected_tool=case.expected_tool,
        selected_tool=selected_tool,
        expected_score=round(expected_score, 4),
        confusing_scores={name: round(score, 4) for name, score in confusing_scores.items()},
        passed=passed,
        cue_coverage=round(cue_coverage, 4),
        notes=tuple(notes),
    ), warnings


def _rank_score(user_request: str, candidate: ToolDescriptionCandidate) -> float:
    request_tokens = _tokens(user_request)
    name_tokens = set(candidate.name.lower().split("_")) | {candidate.name.lower()}
    description_tokens = _tokens(candidate.candidate_description)
    baseline_tokens = _tokens(candidate.baseline_description)

    if not request_tokens:
        return 0.0

    description_overlap = len(request_tokens & description_tokens) / len(request_tokens)
    baseline_overlap = len(request_tokens & baseline_tokens) / len(request_tokens)
    name_overlap = len(request_tokens & name_tokens) / len(request_tokens)
    return (0.65 * description_overlap) + (0.2 * baseline_overlap) + (0.15 * name_overlap)


def _cue_coverage(
    candidate: ToolDescriptionCandidate,
    required_cues: Sequence[str],
    required_arguments: Sequence[str],
) -> float:
    checks = list(required_cues) + list(required_arguments)
    if not checks:
        return 1.0
    haystack = " ".join(
        [candidate.candidate_description, *candidate.parameter_descriptions.keys(), *candidate.parameter_descriptions.values()]
    ).lower()
    covered = sum(1 for cue in checks if cue.lower() in haystack)
    return covered / len(checks)


def _constraint_pass_rate(
    candidates: Sequence[ToolDescriptionCandidate],
    *,
    max_description_chars: int,
    max_parameter_description_chars: int,
    warnings: list[str],
) -> float:
    checks: list[bool] = []
    for candidate in candidates:
        description = candidate.candidate_description.strip()
        length_ok = 0 < len(description) <= max_description_chars
        checks.append(length_ok)
        if not length_ok:
            warnings.append(f"Description length constraint failed: {candidate.name}")

        safety_ok = not any(pattern in description.lower() for pattern in _DANGEROUS_DESCRIPTION_PATTERNS)
        checks.append(safety_ok)
        if not safety_ok:
            warnings.append(f"Dangerous wording constraint failed: {candidate.name}")

        for param_name, param_description in candidate.parameter_descriptions.items():
            param_ok = 0 < len(param_description.strip()) <= max_parameter_description_chars
            checks.append(param_ok)
            if not param_ok:
                warnings.append(f"Parameter description length constraint failed: {candidate.name}.{param_name}")

    return _mean(1.0 if check else 0.0 for check in checks)


def _expected_tool_coverage(
    candidate_map: Mapping[str, ToolDescriptionCandidate],
    cases: Sequence[ToolSelectionCase],
) -> float:
    """Return whether golden expected tools are present in the candidate set."""

    if not cases:
        return 0.0
    return _mean(1.0 if case.expected_tool in candidate_map else 0.0 for case in cases)


def _extract_parameter_descriptions(
    schema: Mapping[str, object],
    *,
    max_chars: int = DEFAULT_MAX_PARAMETER_DESCRIPTION_CHARS,
) -> dict[str, str]:
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        return {}

    descriptions: dict[str, str] = {}
    for name, raw_spec in properties.items():
        if not isinstance(name, str) or not isinstance(raw_spec, Mapping):
            continue
        description = raw_spec.get("description")
        if isinstance(description, str) and description.strip():
            descriptions[name] = normalize_parameter_description(description, max_chars=max_chars)
    return descriptions


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 4)
