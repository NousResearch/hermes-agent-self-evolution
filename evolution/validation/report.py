"""ValidationReport — JSON schema + Rich console rendering.

Schema mirrors the existing ``gate_decision.json`` shape pattern so
downstream calibration scripts can use the same parsers.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.table import Table

from evolution.validation.agent_runner import AgentRunResult


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    passed: bool
    abstained: bool
    tool_calls_seq: list[str]
    duration_seconds: float
    model_name: Optional[str] = None
    error: Optional[str] = None
    pass_rate: Optional[float] = None

    def __post_init__(self) -> None:
        # Single-run (reps=1) callers construct with `passed` only; derive
        # the rate so win/loss can compare pass_rate uniformly. At reps=1
        # the rate is exactly 0.0 or 1.0, so passed == (pass_rate >= 0.5).
        if self.pass_rate is None:
            object.__setattr__(self, "pass_rate", 1.0 if self.passed else 0.0)


@dataclass(frozen=True)
class PhaseResult:
    pass_rate: float
    n_passed: int
    n_failed: int
    n_abstained: int
    tasks: list[TaskResult]


@dataclass(frozen=True)
class WinLoss:
    n_wins: int
    n_losses: int
    n_ties: int
    pass_rate_change: float


def score_task(
    *,
    expected_tools: tuple[str, ...],
    forbidden_tools: tuple[str, ...],
    run: AgentRunResult,
    test_command: Optional[str] = None,
    fixture_dir: Optional[Path] = None,
    test_command_timeout_seconds: float = 60.0,
    layer2_judge_fn: Optional[Callable[[list[dict]], float]] = None,
    layer2_threshold: float = 0.7,
    expected_action: Optional[str] = None,
    target_skill: Optional[str] = None,
    stale_token: Optional[str] = None,
    required_cmd_substr: tuple[str, ...] = (),
    forbidden_cmd_substr: tuple[str, ...] = (),
    command_tool: str = "Bash",
) -> tuple[bool, bool]:
    """Return (passed, abstained).

    Abstention takes precedence over pass/fail: a task that errored out
    in the runner is not evidence of the artifact's quality either way.

    When ``expected_action == "patch"`` (with ``target_skill`` and
    ``stale_token`` set), the verdict is action-level: pass iff the agent
    called ``skill_manage`` with ``action in {patch, edit}`` on
    ``target_skill`` and the call touched the stale token (for ``patch``:
    ``stale_token in old_string``; for ``edit``: ``stale_token not in
    content``, meaning the replacement was applied). All other paths are
    ignored in this mode.

    When ``test_command`` is set (skill-side suites), the verdict is
    "did the planted test pass after the agent's edits": the command
    runs in ``fixture_dir`` with the given timeout, and passes iff exit
    code is zero. ``expected_tools`` / ``forbidden_tools`` are ignored
    in this mode. Command failure modes (nonzero exit, timeout,
    FileNotFoundError) all map to ``(False, False)`` — "the test did
    not pass," which is the meaningful verdict regardless of cause.

    When ``expected_action == "convention"``, the verdict is convention
    adherence: pass iff some ``Bash`` call's command contains one of
    ``required_cmd_substr`` (the agent used the repo's wrapper) AND no
    ``Bash`` command contains any of ``forbidden_cmd_substr`` (it did not
    fall back to the default tool). All other paths are ignored in this mode.

    Layer 2 (compound verdict, prompt-section suites): when
    ``layer2_judge_fn`` is provided, a task passes only if Layer 1
    (trigger membership) passes AND the judge returns a score
    ``>= layer2_threshold``. The judge receives the subset of
    ``run.tool_calls_with_args`` whose name is ``memory`` (each item the
    call's ``arguments`` dict). Layer 2 is short-circuited when Layer 1
    fails — the judge is never called, so no LLM cost is spent on a task
    that already failed the trigger test. ``test_command`` mode ignores
    Layer 2.
    """
    if run.error is not None:
        return False, True
    if expected_action == "patch":
        return _score_action_patch(run, target_skill=target_skill, stale_token=stale_token), False
    if expected_action == "convention":
        return _score_convention(
            run,
            required_cmd_substr=required_cmd_substr,
            forbidden_cmd_substr=forbidden_cmd_substr,
            command_tool=command_tool,
        ), False
    if test_command is not None:
        if fixture_dir is None:
            raise ValueError(
                "score_task: fixture_dir is required when test_command is set"
            )
        return _run_test_command(test_command, fixture_dir, test_command_timeout_seconds), False
    invoked = set(run.tool_calls_seq)
    if forbidden_tools and (invoked & set(forbidden_tools)):
        return False, False
    if expected_tools and not (invoked & set(expected_tools)):
        return False, False
    if layer2_judge_fn is not None:
        memory_calls = [
            c["arguments"]
            for c in run.tool_calls_with_args
            if c.get("name") == "memory"
        ]
        if layer2_judge_fn(memory_calls) < layer2_threshold:
            return False, False
    return True, False


def _run_test_command(command: str, cwd: Path, timeout_seconds: float) -> bool:
    """Run ``command`` in ``cwd``. Return True iff exit code is zero.

    Uses ``shlex.split`` (no shell) so suite-controlled commands don't
    accidentally pick up shell metacharacters. All failure modes
    (nonzero exit, timeout, FileNotFoundError, OSError) return False —
    the test did not pass is the verdict the caller needs.
    """
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# Execution-aware convention matching. The convention verdict is judge-free and
# treated as ground truth, so it must distinguish a wrapper being *invoked* from one
# merely *mentioned* (`cat bin/check`, `echo 'bin/check'`, `./bin/check --help`).
_CONV_PREFIX_WRAPPERS = frozenset({"sudo", "env", "time", "nohup", "exec", "command", "nice"})
_CONV_INTERPRETERS = frozenset({"bash", "sh", "zsh", "dash"})
# Read-only / echo commands: a substring appearing as their ARGUMENT is a mention,
# not an execution of that tool.
_CONV_INSPECTORS = frozenset({
    "cat", "echo", "printf", "grep", "egrep", "fgrep", "less", "more",
    "head", "tail", "ls", "find", "stat", "file", "wc", "true", ":",
})
_CONV_HELP_FLAGS = frozenset({"--help", "-h", "--version", "-V"})
_CONV_OPERATORS = frozenset({"&&", "||", ";", "|", "&"})
_CONV_ENV_ASSIGN = re.compile(r"^\w+=")
_CONV_TAIL = r"(?![A-Za-z0-9_.\-])"


def _conv_tokenize(text: str) -> list[str]:
    try:
        return shlex.split(text, posix=True)
    except ValueError:  # unbalanced quotes etc. — degrade to whitespace split
        return text.split()


def _conv_segments(command: str) -> list[list[str]]:
    """Split a command into segments (token lists) on shell operators, quote-aware
    (operators inside quotes stay part of their token)."""
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in _conv_tokenize(command):
        if tok in _CONV_OPERATORS:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def _conv_base(prog: str) -> str:
    return prog.rsplit("/", 1)[-1]


def _conv_invoked_programs(tokens: list[str]) -> set[str]:
    """Best-effort set of programs actually executed in one segment.

    Strips env-assignments and command wrappers (sudo/env/...); for an interpreter
    (`bash X`, `sh -c "..."`) and `python -m <mod>` it also includes the script /
    module being run. An argument to e.g. ``cat`` is NOT an invoked program.
    """
    i = 0
    while i < len(tokens) and (tokens[i] in _CONV_PREFIX_WRAPPERS or _CONV_ENV_ASSIGN.match(tokens[i])):
        i += 1
    rest = tokens[i:]
    if not rest:
        return set()
    progs = {rest[0]}
    base = _conv_base(rest[0])
    if base in _CONV_INTERPRETERS:
        if "-c" in rest:  # bash -c "<script>": recurse one level into the body
            ci = rest.index("-c")
            if ci + 1 < len(rest):
                for seg in _conv_segments(rest[ci + 1]):
                    progs |= _conv_invoked_programs(seg)
        else:  # bash <script>: the first non-flag token is the invoked script
            for t in rest[1:]:
                if not t.startswith("-"):
                    progs.add(t)
                    break
    if base in {"python", "python3"} and "-m" in rest:
        mi = rest.index("-m")
        if mi + 1 < len(rest):
            progs.add(rest[mi + 1])
    return progs


def _conv_matches(substr: str, text: str) -> bool:
    return re.search(re.escape(substr) + _CONV_TAIL, text) is not None


def _score_convention(
    run: AgentRunResult,
    *,
    required_cmd_substr: tuple[str, ...],
    forbidden_cmd_substr: tuple[str, ...],
    command_tool: str = "Bash",
) -> bool:
    """Return True iff the agent INVOKED a required wrapper and never ran a forbidden
    default tool.

    Used to score adherence to a repo-specific convention (e.g. "run tests with
    ./bin/check, never pytest"). Agent-agnostic: reads only the ``command_tool`` calls
    (default ``Bash``) in ``tool_calls_with_args``.

    Execution-aware (not raw substring): each command is split into shell segments and
    each segment's *invoked programs* identified. A required substring counts as **used**
    only when it matches an invoked program (so ``./bin/check`` / ``bash bin/check`` count
    but ``cat bin/check`` / ``echo 'bin/check'`` / ``./bin/check --help`` do not). A
    forbidden substring counts as a **bypass** only when it appears in a segment whose
    program is not a read-only inspector and which isn't a help/version call (so
    ``python -m pytest`` / ``python app.py`` count but ``cat pytest.ini`` /
    ``echo 'pytest'`` do not). Substring matching is trailing-boundary aware (``pytest``
    does not match ``pytest.ini``). A forbidden default run *anywhere* fails the task —
    the convention is "never use the default", so explore-then-comply fails by design.
    """
    commands = [
        (call.get("arguments") or {}).get("command", "")
        for call in run.tool_calls_with_args
        if call.get("name") == command_tool
    ]
    used = False
    bypassed = False
    for command in commands:
        for seg in _conv_segments(command):
            if any(t in _CONV_HELP_FLAGS for t in seg):
                continue  # --help/--version: program named but didn't do its job
            progs = _conv_invoked_programs(seg)
            if not progs:
                continue
            if any(_conv_matches(req, p) for req in required_cmd_substr for p in progs):
                used = True
            bases = {_conv_base(p) for p in progs}
            if not (bases <= _CONV_INSPECTORS):  # not a pure mention/inspection segment
                seg_text = " ".join(seg)
                if any(_conv_matches(forb, seg_text) for forb in forbidden_cmd_substr):
                    bypassed = True
    return used and not bypassed


def _score_action_patch(
    run: AgentRunResult,
    *,
    target_skill: Optional[str],
    stale_token: Optional[str],
) -> bool:
    """Return True iff any skill_manage call on target_skill touched stale_token.

    Accepts both ``action='patch'`` (stale_token must appear in old_string) and
    ``action='edit'`` (stale_token must be absent from content, meaning it was
    replaced).  Any other action, wrong skill, or missing token evidence → False.
    """
    for call in run.tool_calls_with_args:
        if call.get("name") != "skill_manage":
            continue
        args = call.get("arguments") or {}
        if args.get("name") != target_skill:
            continue
        action = args.get("action")
        if action == "patch":
            old_string = args.get("old_string", "")
            if stale_token is not None and stale_token in old_string:
                return True
        elif action == "edit":
            content = args.get("content", "")
            if stale_token is not None and stale_token not in content:
                return True
    return False


def summarize_phase(task_results: list[TaskResult]) -> PhaseResult:
    n_passed = sum(1 for r in task_results if r.passed and not r.abstained)
    n_abstained = sum(1 for r in task_results if r.abstained)
    n_failed = sum(1 for r in task_results if not r.passed and not r.abstained)
    scored = n_passed + n_failed
    pass_rate = (n_passed / scored) if scored else 0.0
    return PhaseResult(
        pass_rate=pass_rate,
        n_passed=n_passed,
        n_failed=n_failed,
        n_abstained=n_abstained,
        tasks=task_results,
    )


def compute_win_loss(
    baseline: PhaseResult,
    evolved: PhaseResult,
    *,
    per_task_tolerance: Optional[dict[str, float]] = None,
    default_tolerance: float = 0.0,
) -> WinLoss:
    """Per-task win/loss: how the evolved phase moved vs baseline on
    each task_id. Abstentions on either side are ties.

    A per-task movement counts only if it exceeds the task's noise tolerance:
    win iff ``e - b > tol``, loss iff ``b - e > tol``, else tie. ``tol`` is
    ``per_task_tolerance.get(task_id, default_tolerance)``. The default
    tolerance of 0.0 reduces ``>`` to the legacy ``e.pass_rate > b.pass_rate``,
    so the no-tolerance gate is byte-for-byte unchanged. Tolerances come from
    the A/A noise floor (a task's measured spurious flip rate) so stochastic
    movement smaller than the floor isn't scored as signal.
    """
    by_id_baseline = {r.task_id: r for r in baseline.tasks}
    by_id_evolved = {r.task_id: r for r in evolved.tasks}
    n_wins = n_losses = n_ties = 0
    for task_id in by_id_baseline.keys() | by_id_evolved.keys():
        b = by_id_baseline.get(task_id)
        e = by_id_evolved.get(task_id)
        if b is None or e is None or b.abstained or e.abstained:
            n_ties += 1
            continue
        tol = (
            per_task_tolerance.get(task_id, default_tolerance)
            if per_task_tolerance is not None
            else default_tolerance
        )
        if e.pass_rate - b.pass_rate > tol:
            n_wins += 1
        elif b.pass_rate - e.pass_rate > tol:
            n_losses += 1
        else:
            n_ties += 1
    return WinLoss(
        n_wins=n_wins,
        n_losses=n_losses,
        n_ties=n_ties,
        pass_rate_change=evolved.pass_rate - baseline.pass_rate,
    )


def decide(
    baseline: PhaseResult,
    evolved: PhaseResult,
    wl: WinLoss,
    *,
    aggregate_tolerance: float = 0.0,
) -> tuple[str, list[str]]:
    """Two-condition decision rule.

    1) ``evolved.pass_rate >= baseline.pass_rate - aggregate_tolerance``
       (aggregate no-regression, within a noise tolerance).
    2) ``n_losses == 0`` OR ``n_wins >= 2 * n_losses`` (no per-task
       regression unless offset 2:1 by wins).

    Ties (equal pass-rate, no per-task losses) decide as ``pass`` —
    same semantics as the ``--benchmark-cmd`` no-regression rule.
    ``aggregate_tolerance`` defaults to 0.0 (strict, legacy behavior); when set
    from the A/A noise floor, an aggregate dip smaller than the floor isn't
    scored as a regression.
    """
    reasons: list[str] = []
    aggregate_ok = evolved.pass_rate >= baseline.pass_rate - aggregate_tolerance
    tol_note = (
        f" (tolerance {aggregate_tolerance:.2f})" if aggregate_tolerance else ""
    )
    if aggregate_ok:
        reasons.append(
            f"evolved pass_rate {evolved.pass_rate:.2f} >= baseline "
            f"{baseline.pass_rate:.2f}{tol_note}"
        )
    else:
        reasons.append(
            f"evolved pass_rate {evolved.pass_rate:.2f} < baseline "
            f"{baseline.pass_rate:.2f}{tol_note}"
        )
    per_task_ok = (wl.n_losses == 0) or (wl.n_wins >= 2 * wl.n_losses)
    if per_task_ok:
        reasons.append(
            f"per-task: {wl.n_wins} wins, {wl.n_losses} losses, {wl.n_ties} ties"
        )
    else:
        reasons.append(
            f"per-task regression: {wl.n_losses} losses, {wl.n_wins} wins "
            f"(need wins >= 2 * losses)"
        )
    return ("pass" if (aggregate_ok and per_task_ok) else "regression", reasons)


@dataclass(frozen=True)
class ValidationReport:
    schema_version: str
    tool: str
    task_suite_path: str
    task_suite_sha256: str
    baseline: PhaseResult
    evolved: PhaseResult
    delta: WinLoss
    decision: str
    decision_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool": self.tool,
            "task_suite_path": self.task_suite_path,
            "task_suite_sha256": self.task_suite_sha256,
            "baseline": asdict(self.baseline),
            "evolved": asdict(self.evolved),
            "delta": asdict(self.delta),
            "decision": self.decision,
            "decision_reasons": self.decision_reasons,
        }

    def write_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2))

    def render_console(self, console: Optional[Console] = None) -> None:
        c = console or Console()
        table = Table(title=f"Closed-loop validation: {self.tool}")
        table.add_column("Task", style="bold")
        table.add_column("Baseline", justify="center")
        table.add_column("Evolved", justify="center")
        table.add_column("Δ", justify="center")
        b_by_id = {r.task_id: r for r in self.baseline.tasks}
        e_by_id = {r.task_id: r for r in self.evolved.tasks}
        for task_id in b_by_id.keys() | e_by_id.keys():
            b = b_by_id.get(task_id)
            e = e_by_id.get(task_id)
            table.add_row(
                task_id,
                _cell(b),
                _cell(e),
                _delta_cell(b, e),
            )
        table.add_section()
        table.add_row(
            "[bold]aggregate[/bold]",
            f"{self.baseline.n_passed}/{self.baseline.n_passed + self.baseline.n_failed} "
            f"({self.baseline.pass_rate:.0%})",
            f"{self.evolved.n_passed}/{self.evolved.n_passed + self.evolved.n_failed} "
            f"({self.evolved.pass_rate:.0%})",
            f"{self.delta.pass_rate_change:+.2f}",
        )
        c.print(table)
        c.print(
            f"\n[bold]Decision:[/bold] "
            f"[{'green' if self.decision == 'pass' else 'red'}]{self.decision}[/]"
        )
        for r in self.decision_reasons:
            c.print(f"  • {r}")


def _cell(result: Optional[TaskResult]) -> str:
    if result is None:
        return "—"
    if result.abstained:
        return "[yellow]abstain[/yellow]"
    return "[green]✓[/green]" if result.passed else "[red]✗[/red]"


def _delta_cell(b: Optional[TaskResult], e: Optional[TaskResult]) -> str:
    if b is None or e is None or b.abstained or e.abstained:
        return ""
    if e.passed and not b.passed:
        return "[green]win[/green]"
    if b.passed and not e.passed:
        return "[red]loss[/red]"
    return ""
