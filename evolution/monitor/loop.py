"""The continuous self-improvement loop: check, triage, dispatch, propose.

This is the piece that turns four one-shot optimizers into a system. One cycle:

    1. run the scheduled checks  - benchmarks, scored and logged over time
    2. consult triage            - what is worst, weighted by how much it is used
    3. dispatch the winner       - to Phase 1/2/3/4 as a subprocess
    4. record the outcome        - back into the same history triage reads
    5. stop                      - a human reads the diff and merges the PR

Three decisions in here are deliberate and worth stating, because each one is a
place where a more automated design would have been easy and wrong.

**The clock is cron, not a daemon.** ``--emit-cron`` prints a line for the user
to install. It does not touch a crontab. Writing to someone's crontab because
they ran a CLI once is the kind of helpfulness nobody asked for, and a long-lived
background process on a developer machine is worse. PLAN.md says "wire up to the
cron scheduler"; the honest reading of that is to hand over a line, not to take
the pen.

**The loop stops at proposing.** PLAN.md is explicit that a human merges every
PR, so nothing here merges, pushes, or deploys. The phase entry points produce
branches and diffs; this module decides which one to run and when.

**A skip is not a success.** If the API key is missing, or the hermes-agent
checkout is absent, or a phase module has not been built yet, the cycle records
a skip with the reason. It never reports a cycle it did not run. The same rule
the gates module applies to absent benchmarks applies here: unavailable is its
own status, not a quiet pass. On this repo today that matters, because Phases
2-4 may or may not be installed alongside this one, and TBLite and YC-Bench do
not exist in hermes-agent at all.

CLI:

    python -m evolution.monitor.loop --once
    python -m evolution.monitor.loop --once --dry-run --max-targets 3
    python -m evolution.monitor.loop --emit-cron --threshold 0.25
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.gates import (
    KNOWN_BENCHMARKS,
    GateResult,
    GateStatus,
    run_benchmark_gate,
)
from evolution.monitor.metrics import (
    BENCHMARK_SCORE,
    OPTIMIZATION_RUN,
    MetricStore,
)
from evolution.monitor.triage import AutoTriage, TargetType, TriageConfig, TriageEntry

__all__ = [
    "DEFAULT_BENCHMARKS",
    "DEFAULT_SCHEDULE",
    "API_KEY_ENV_VARS",
    "PHASE_ENV_PASSTHROUGH",
    "PHASE_ENV_PREFIXES",
    "phase_environment",
    "PhaseEntry",
    "PHASE_DISPATCH",
    "LoopConfig",
    "CycleStatus",
    "DispatchStatus",
    "Dispatch",
    "CheckOutcome",
    "CycleReport",
    "ProcessResult",
    "default_history_path",
    "looks_like_hermes_repo",
    "phase_module_available",
    "preflight",
    "build_command",
    "run_cycle",
    "cron_line",
    "main",
]

console = Console()

# PLAN.md: "Weekly: Run TBLite + YC-Bench fast_test, log scores."
DEFAULT_BENCHMARKS: tuple[str, ...] = ("tblite", "yc_bench")

# Monday 03:00. Weekly, and outside working hours, because a full benchmark run
# is measured in hours and dollars.
DEFAULT_SCHEDULE = "0 3 * * 1"

# One of these must be set before a phase entry point is worth spawning. Every
# phase drives an LLM through DSPy, and a run without a key burns a cycle to
# produce a stack trace.
API_KEY_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
)

# What a dispatched phase inherits from this process's environment - and the
# complete list of it. The monitor used to hand its children a full copy of
# os.environ, which forwarded every token and secret in the operator's shell
# to a subprocess that ultimately drives contributor-influenced code; a
# scheduled loop is exactly the process most likely to be running in a shell
# full of credentials it never needed. Phases get what a phase needs: the
# interpreter and its search paths, locale and TLS basics, the model keys and
# endpoints DSPy reads, the evolver override, and the repo pointer. A phase
# that needs one more variable gets it added here by name, with the reason.
PHASE_ENV_PASSTHROUGH: tuple[str, ...] = (
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "TZ",
    "TERM",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ANTHROPIC_BASE_URL",
    "OPENROUTER_BASE_URL",
    "DARWINIAN_EVOLVER_CMD",
    "HERMES_AGENT_REPO",
    *API_KEY_ENV_VARS,
)

# Locale variables come as a family (LC_ALL, LC_CTYPE, LC_MESSAGES, ...);
# matching the prefix keeps the list above finite without dropping any.
PHASE_ENV_PREFIXES: tuple[str, ...] = ("LC_",)


def phase_environment(source: Optional[dict] = None) -> dict:
    """The allowlisted child environment for a dispatched phase.

    Built by naming what crosses, never by copying what happens to be there.
    """
    parent = os.environ if source is None else source
    env = {
        name: value
        for name, value in parent.items()
        if name in PHASE_ENV_PASSTHROUGH
        or any(name.startswith(prefix) for prefix in PHASE_ENV_PREFIXES)
    }
    return env


@dataclass(frozen=True)
class PhaseEntry:
    """How to invoke one phase's optimizer for one target."""

    phase: int
    module: str
    flag: str
    label: str
    # Phases 2 and 3 measure and report by default and write nothing. Dispatching
    # them without --write meant the loop could never produce the branch PLAN.md's
    # Phase 5 "Done when" requires ("detect problem -> optimize -> PR"), while
    # still reporting the cycle as proposed. Phase 1 has no write path at all and
    # Phase 4 always produces a branch, so neither takes the flag.
    write_flag: str = ""


# The dispatch table PLAN.md Phase 5 implies: skills to Phase 1, tools to
# Phase 2, prompts to Phase 3, code to Phase 4. Module paths and flags match the
# invocation examples in PLAN.md's "How It's Invoked" section.
PHASE_DISPATCH: dict[TargetType, PhaseEntry] = {
    TargetType.SKILL: PhaseEntry(
        phase=1,
        module="evolution.skills.evolve_skill",
        flag="--skill",
        label="skill evolution",
    ),
    TargetType.TOOL: PhaseEntry(
        phase=2,
        module="evolution.tools.evolve_tool_descriptions",
        flag="--tool",
        label="tool description evolution",
        write_flag="--write",
    ),
    TargetType.PROMPT: PhaseEntry(
        phase=3,
        module="evolution.prompts.evolve_prompt_section",
        flag="--section",
        label="prompt section evolution",
        write_flag="--write",
    ),
    TargetType.CODE: PhaseEntry(
        phase=4,
        module="evolution.code.evolve_tool_code",
        flag="--tool",
        label="tool code evolution",
    ),
}


def default_history_path(output_dir: Optional[Path] = None) -> Path:
    """Where the metric history lives by default: ``<output_dir>/monitor/``."""
    base = Path(output_dir) if output_dir is not None else EvolutionConfig().output_dir
    return Path(base) / "monitor" / "metrics.jsonl"


# ──────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────


class CycleStatus(str, Enum):
    """How one monitor cycle ended."""
    PROPOSED = "proposed"
    SKIPPED = "skipped"
    FAILED = "failed"
    # Targets were ranked, dispatched, and every phase ran cleanly without
    # producing a branch. Distinct from NO_TARGETS, where nothing was ranked
    # at all: "the phases rejected every candidate" and "there was nothing to
    # try" call for different operator responses.
    NO_CHANGE = "no_change"
    NO_TARGETS = "no_targets"
    DRY_RUN = "dry_run"


class DispatchStatus(str, Enum):
    """How one dispatched phase run ended."""
    PROPOSED = "proposed"
    # A phase can exit 0 having decided nothing was deployable: the guard
    # rejected the candidate, the gates blocked it, or the rewrite came back
    # identical. Reading exit 0 as "proposed" told the operator a human had a PR
    # to review when no branch existed anywhere.
    NO_CHANGE = "no_change"
    SKIPPED = "skipped"
    FAILED = "failed"
    DRY_RUN = "dry_run"


@dataclass
class ProcessResult:
    """What a dispatch runner reports back. Mocked wholesale in the tests."""

    returncode: int
    output: str = ""


@dataclass
class CheckOutcome:
    """One scheduled check and whether its score made it into history."""

    name: str
    status: str
    message: str
    score: Optional[float] = None
    baseline: Optional[float] = None
    recorded: bool = False

    def to_dict(self) -> dict:
        """Serialise one gate outcome from a dispatched run."""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "score": self.score,
            "baseline": self.baseline,
            "recorded": self.recorded,
        }


@dataclass
class Dispatch:
    """One optimization attempt, run or declined."""

    target: str
    target_type: TargetType
    phase: Optional[int]
    module: str
    command: list[str]
    status: DispatchStatus
    reason: str = ""
    returncode: Optional[int] = None
    duration_s: float = 0.0
    output_tail: str = ""

    @property
    def command_line(self) -> str:
        """The dispatched command, shell-quoted so it can be copied and rerun."""
        return " ".join(shlex.quote(part) for part in self.command)

    def to_dict(self) -> dict:
        """Serialise the dispatch and everything it reported."""
        return {
            "target": self.target,
            "target_type": self.target_type.value,
            "phase": self.phase,
            "module": self.module,
            "command": self.command,
            "status": self.status.value,
            "reason": self.reason,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 2),
            "output_tail": self.output_tail,
        }


@dataclass
class CycleReport:
    """Everything one cycle did, in a shape a human or a script can read."""

    started_at: float
    status: CycleStatus
    dry_run: bool = False
    hermes_repo: Optional[str] = None
    history_path: Optional[str] = None
    checks: list[CheckOutcome] = field(default_factory=list)
    ranked: list[TriageEntry] = field(default_factory=list)
    dispatches: list[Dispatch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def proposed(self) -> list[Dispatch]:
        """Dispatches that produced a proposal for review."""
        return [d for d in self.dispatches if d.status is DispatchStatus.PROPOSED]

    @property
    def skipped(self) -> list[Dispatch]:
        """Dispatches that declined to run."""
        return [d for d in self.dispatches if d.status is DispatchStatus.SKIPPED]

    @property
    def failed(self) -> list[Dispatch]:
        """Dispatches that exited non-zero."""
        return [d for d in self.dispatches if d.status is DispatchStatus.FAILED]

    def to_dict(self) -> dict:
        """Serialise the whole cycle for the run log."""
        return {
            "started_at": self.started_at,
            "status": self.status.value,
            "dry_run": self.dry_run,
            "hermes_repo": self.hermes_repo,
            "history_path": self.history_path,
            "checks": [c.to_dict() for c in self.checks],
            "ranked": [e.to_dict() for e in self.ranked],
            "dispatches": [d.to_dict() for d in self.dispatches],
            "notes": list(self.notes),
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class LoopConfig:
    """Everything one cycle needs that is not the metric store itself."""

    hermes_repo: Optional[Path] = None
    triage: TriageConfig = field(default_factory=TriageConfig)
    max_targets: int = 1
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS
    # A proposal already waiting on a human should not be re-proposed every
    # week. Two weeks is long enough for a review to land.
    cooldown_days: float = 14.0
    iterations: Optional[int] = None
    python: str = field(default_factory=lambda: sys.executable)
    dispatch_timeout: int = 7200


# ──────────────────────────────────────────────────────────────────────────
# Preflight
# ──────────────────────────────────────────────────────────────────────────


def looks_like_hermes_repo(path: Optional[Path]) -> bool:
    """True when *path* plausibly holds a hermes-agent checkout.

    Checks for the markers that actually exist in the repo today: the root
    ``batch_runner.py``, the ``tools`` package, the ``agent`` package.
    """
    if path is None:
        return False
    path = Path(path)
    if not path.is_dir():
        return False
    markers = ("batch_runner.py", "tools", "agent", "skills")
    return any((path / marker).exists() for marker in markers)


def phase_module_available(module: str) -> bool:
    """True when *module* can be imported in this environment.

    Phases 2-4 live in sibling directories of this one and may simply not be
    installed yet. Asking the import system is cheaper and more honest than
    spawning a subprocess to watch it fail.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def has_api_key(env: Optional[dict] = None) -> bool:
    """True when any recognised model-provider key is set and non-empty."""
    source = os.environ if env is None else env
    return any(str(source.get(name, "")).strip() for name in API_KEY_ENV_VARS)


def preflight(
    target_type: TargetType,
    hermes_repo: Optional[Path],
    *,
    env: Optional[dict] = None,
    module_available: Callable[[str], bool] = phase_module_available,
) -> tuple[bool, str, Optional[PhaseEntry]]:
    """Can this target actually be optimized right now?

    Returns ``(ok, reason, entry)``. A false *ok* is a legitimate skip, not an
    error: the reason is recorded and reported so the next cycle, or the human
    reading the log, knows exactly what is missing.
    """
    entry = PHASE_DISPATCH.get(target_type)
    if entry is None:
        return (
            False,
            f"no phase entry point optimizes a '{target_type.value}' target",
            None,
        )
    if not module_available(entry.module):
        return (
            False,
            f"phase {entry.phase} module '{entry.module}' is not installed",
            entry,
        )
    if not looks_like_hermes_repo(hermes_repo):
        location = str(hermes_repo) if hermes_repo else "unset"
        return (
            False,
            f"no hermes-agent checkout at {location} "
            "(set HERMES_AGENT_REPO or pass --hermes-repo)",
            entry,
        )
    if not has_api_key(env):
        return (
            False,
            "no API key in the environment (" + ", ".join(API_KEY_ENV_VARS) + ")",
            entry,
        )
    return True, "", entry


def build_command(
    entry: PhaseEntry,
    target: str,
    *,
    python: Optional[str] = None,
    iterations: Optional[int] = None,
) -> list[str]:
    """The subprocess argv for one phase run.

    Only the target flag and, optionally, ``--iterations`` are passed. The
    hermes-agent path travels in the environment as ``HERMES_AGENT_REPO``
    instead of on the command line, because every phase resolves it through
    :mod:`evolution.core.config` and not every phase is guaranteed to expose the
    same flag.
    """
    command = [python or sys.executable, "-m", entry.module, entry.flag, target]
    if iterations is not None:
        command += ["--iterations", str(iterations)]
    if entry.write_flag:
        command.append(entry.write_flag)
    return command


def _evolve_branches(repo: Optional[Path]) -> set[str]:
    """The ``evolve/`` branches present in *repo* right now, or an empty set.

    Never raises: a checkout that is not a git repository, or a git that is not
    installed, simply yields nothing to compare and the dispatch falls back to
    reporting no change rather than crashing a scheduled run.
    """
    if repo is None:
        return set()
    try:
        proc = subprocess.run(
            ["git", "branch", "--list", "evolve/*", "--format=%(refname:short)"],
            cwd=str(repo), capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _subprocess_runner(
    command: Sequence[str], env: dict, timeout: int
) -> ProcessResult:
    """Default dispatcher. Never imported by tests, always mocked there."""
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ProcessResult(returncode=124, output=f"timed out after {timeout}s")
    except OSError as exc:
        return ProcessResult(returncode=127, output=str(exc))

    combined = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(combined.strip().splitlines()[-25:])
    return ProcessResult(returncode=proc.returncode, output=tail)


# ──────────────────────────────────────────────────────────────────────────
# The cycle
# ──────────────────────────────────────────────────────────────────────────


def _step(out: Console, text: str) -> None:
    out.print(f"\n[bold]── {text} ─────────────────────────[/bold]")


def _run_scheduled_checks(
    config: LoopConfig,
    store: MetricStore,
    now: float,
    benchmark_runner: Callable[..., GateResult],
    out: Console,
) -> tuple[list[CheckOutcome], list[str]]:
    """Run the periodic benchmarks and log whatever they actually scored."""
    outcomes: list[CheckOutcome] = []
    notes: list[str] = []

    if not looks_like_hermes_repo(config.hermes_repo):
        location = str(config.hermes_repo) if config.hermes_repo else "unset"
        notes.append(f"scheduled checks skipped: no hermes-agent checkout at {location}")
        out.print(f"  [yellow]○ skipped[/yellow]: {notes[-1]}")
        return outcomes, notes

    for name in config.benchmarks:
        previous = store.latest(BENCHMARK_SCORE, name)
        baseline = previous.value if previous else None
        result = benchmark_runner(
            config.hermes_repo,
            name,
            baseline=baseline,
            fast=True,
        )

        recorded = False
        if result.score is not None:
            # A benchmark score summarizes however many tasks ran, and triage
            # weighs a target by how much evidence stands behind it. Recording
            # samples=1 would make every benchmark look like a single anecdote
            # next to a tool measured over hundreds of turns.
            spec = KNOWN_BENCHMARKS.get(name)
            tasks = spec.fast_task_count if spec else 1

            # A real number goes into history even when the gate failed on a
            # regression. The regression is exactly what the trend should see.
            store.record(
                BENCHMARK_SCORE,
                name,
                result.score,
                source="benchmark",
                samples=tasks,
                timestamp=now,
                metadata={
                    "gate_status": result.status.value,
                    "baseline": baseline,
                    "tasks": tasks,
                    "fast": True,
                },
            )
            recorded = True

        outcomes.append(
            CheckOutcome(
                name=name,
                status=result.status.value,
                message=result.message,
                score=result.score,
                baseline=baseline,
                recorded=recorded,
            )
        )

        icon = {
            GateStatus.PASSED: "[green]✓[/green]",
            GateStatus.FAILED: "[red]✗[/red]",
            GateStatus.UNAVAILABLE: "[yellow]○[/yellow]",
            GateStatus.SKIPPED: "[dim]-[/dim]",
        }.get(result.status, "?")
        suffix = "" if recorded else " [dim](nothing recorded)[/dim]"
        out.print(f"  {icon} {name}: {result.message}{suffix}")

    return outcomes, notes


def _in_cooldown(
    store: MetricStore, target: str, now: float, cooldown_days: float
) -> Optional[float]:
    """Seconds-ago of the last proposal for *target*, if it is still cooling."""
    if cooldown_days <= 0:
        return None
    recent = store.window(
        cooldown_days, now=now, metric=OPTIMIZATION_RUN, target=target
    )
    proposals = [p for p in recent if p.metadata.get("status") == "proposed"]
    if not proposals:
        return None
    return now - proposals[-1].timestamp


def _render_ranking(entries: Sequence[TriageEntry], out: Console) -> None:
    table = Table(title="Optimization targets, ranked")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Target", style="bold")
    table.add_column("Type")
    table.add_column("Score", justify="right")
    table.add_column("Now", justify="right")
    table.add_column("Uses", justify="right")
    table.add_column("Trend")
    # A rank is only as trustworthy as the trend that lifted it, so the
    # evidence rides in its own column instead of hiding inside the score.
    table.add_column("p")
    table.add_column("R²", justify="right")
    table.add_column("Why")

    for index, entry in enumerate(entries, start=1):
        value = "-" if entry.current_value is None else f"{entry.current_value:.2f}"
        trend = entry.trend.direction.value if entry.trend else "-"
        if entry.trend is not None and entry.trend.is_deterioration:
            trend = f"[red]{trend}[/red]"
        if entry.trend_p_value is None:
            p_cell = "[dim]-[/dim]"
        elif entry.trend_p_value < entry.trend.alpha:
            p_cell = f"{entry.trend_p_value:.3f}"
        else:
            # Above alpha the movement is not distinguishable from noise, and
            # the reader should see that before approving a run.
            p_cell = f"[dim]{entry.trend_p_value:.3f} (ns)[/dim]"
        r2_cell = "-" if entry.trend_r_squared is None else f"{entry.trend_r_squared:.2f}"
        target = entry.target if entry.actionable else f"{entry.target} [dim](advisory)[/dim]"
        table.add_row(
            str(index),
            target,
            entry.target_type.value,
            f"{entry.score:.3f}",
            value,
            str(entry.usage_samples),
            trend,
            p_cell,
            r2_cell,
            # Escaped, not interpolated: an explanation is prose, and rich
            # reads a lowercase "[trend p=...]" or "[advisory: ...]" as a style
            # tag and silently drops the whole bracket if the style is unknown.
            escape(entry.explain()),
        )
    out.print(table)


def run_cycle(
    config: LoopConfig,
    store: MetricStore,
    *,
    dry_run: bool = False,
    now: Optional[float] = None,
    benchmark_runner: Callable[..., GateResult] = run_benchmark_gate,
    dispatcher: Callable[[Sequence[str], dict, int], ProcessResult] = _subprocess_runner,
    module_available: Callable[[str], bool] = phase_module_available,
    branch_lister: Callable[[Optional[Path]], set] = _evolve_branches,
    env: Optional[dict] = None,
    out: Optional[Console] = None,
) -> CycleReport:
    """Run one full cycle and return what happened.

    Every side effect is behind an injectable callable - *benchmark_runner*,
    *dispatcher*, *module_available*, *branch_lister* - so the whole cycle can
    be exercised offline without spawning an optimizer or touching a model.
    """
    out = out or console
    started = time.time()
    cycle_now = store.now() if now is None else now
    # Allowlisted, not copied: a dispatched phase gets the variables a phase
    # needs and nothing else from this process's shell. See PHASE_ENV_PASSTHROUGH.
    environment = phase_environment(env)

    report = CycleReport(
        started_at=cycle_now,
        status=CycleStatus.NO_TARGETS,
        dry_run=dry_run,
        hermes_repo=str(config.hermes_repo) if config.hermes_repo else None,
        history_path=str(store.path),
    )

    out.print(
        "\n[bold cyan]Hermes Agent Self-Evolution[/bold cyan] - continuous loop"
        + (" [yellow](dry run)[/yellow]" if dry_run else "")
    )
    out.print(f"  History: {store.path}")
    out.print(f"  Hermes repo: {report.hermes_repo or 'unset'}")

    # ── 1. Scheduled checks ─────────────────────────────────────────────
    _step(out, "Scheduled checks")
    if dry_run:
        planned = ", ".join(config.benchmarks) or "none"
        report.notes.append(f"dry run: would run benchmarks [{planned}]")
        out.print(f"  [dim]would run: {planned}[/dim]")
    else:
        checks, notes = _run_scheduled_checks(
            config, store, cycle_now, benchmark_runner, out
        )
        report.checks = checks
        report.notes.extend(notes)

    # ── 2. Triage ───────────────────────────────────────────────────────
    _step(out, "Triage")
    ranked = AutoTriage(store, config.triage).rank(now=cycle_now)
    report.ranked = ranked

    if not ranked:
        out.print("  No candidates in the window. Nothing to optimize.")
        report.status = CycleStatus.DRY_RUN if dry_run else CycleStatus.NO_TARGETS
        report.duration_s = time.time() - started
        _print_footer(report, out)
        return report

    _render_ranking(ranked, out)

    actionable = [e for e in ranked if e.actionable]
    advisory = [e for e in ranked if not e.actionable]
    for entry in advisory:
        report.notes.append(
            f"advisory only: {entry.target} ({entry.target_type.value}) - "
            "no phase entry point optimizes this target type"
        )

    # ── 3. Dispatch ─────────────────────────────────────────────────────
    _step(out, "Dispatch")
    selected = actionable[: max(0, config.max_targets)]
    if not selected:
        if actionable:
            out.print(f"  --max-targets is {config.max_targets}, so nothing was dispatched.")
        else:
            out.print("  No actionable target. Every candidate is advisory only.")
        report.status = CycleStatus.DRY_RUN if dry_run else CycleStatus.NO_TARGETS
        report.duration_s = time.time() - started
        _print_footer(report, out)
        return report

    for entry in selected:
        report.dispatches.append(
            _dispatch_one(
                entry=entry,
                config=config,
                store=store,
                now=cycle_now,
                dry_run=dry_run,
                dispatcher=dispatcher,
                module_available=module_available,
                branch_lister=branch_lister,
                environment=environment,
                out=out,
            )
        )

    # ── 4. Status ───────────────────────────────────────────────────────
    if dry_run:
        report.status = CycleStatus.DRY_RUN
    elif report.proposed:
        report.status = CycleStatus.PROPOSED
    elif report.failed:
        report.status = CycleStatus.FAILED
    elif report.skipped:
        report.status = CycleStatus.SKIPPED
    elif report.dispatches:
        report.status = CycleStatus.NO_CHANGE
    else:
        report.status = CycleStatus.NO_TARGETS

    report.duration_s = time.time() - started
    _print_footer(report, out)
    return report


def _dispatch_one(
    *,
    entry: TriageEntry,
    config: LoopConfig,
    store: MetricStore,
    now: float,
    dry_run: bool,
    dispatcher: Callable[[Sequence[str], dict, int], ProcessResult],
    module_available: Callable[[str], bool],
    branch_lister: Callable[[Optional[Path]], set],
    environment: dict,
    out: Console,
) -> Dispatch:
    ok, reason, phase_entry = preflight(
        entry.target_type,
        config.hermes_repo,
        env=environment,
        module_available=module_available,
    )
    module = phase_entry.module if phase_entry else ""
    phase = phase_entry.phase if phase_entry else None
    command = (
        build_command(
            phase_entry,
            entry.target,
            python=config.python,
            iterations=config.iterations,
        )
        if phase_entry
        else []
    )

    cooling = _in_cooldown(store, entry.target, now, config.cooldown_days)
    if ok and cooling is not None:
        ok = False
        reason = (
            f"already proposed {cooling / 86400.0:.1f} days ago, "
            f"cooling down for {config.cooldown_days:.0f} days"
        )

    if not ok:
        dispatch = Dispatch(
            target=entry.target,
            target_type=entry.target_type,
            phase=phase,
            module=module,
            command=command,
            status=DispatchStatus.SKIPPED,
            reason=reason,
        )
        out.print(f"  [yellow]○ skipped[/yellow] {entry.target}: {reason}")
        if not dry_run:
            _record_outcome(store, entry, dispatch, now)
        return dispatch

    if dry_run:
        dispatch = Dispatch(
            target=entry.target,
            target_type=entry.target_type,
            phase=phase,
            module=module,
            command=command,
            status=DispatchStatus.DRY_RUN,
            reason="dry run: not executed",
        )
        out.print(f"  [dim]would run[/dim] {dispatch.command_line}")
        return dispatch

    child_env = dict(environment)
    child_env["HERMES_AGENT_REPO"] = str(config.hermes_repo)

    out.print(
        f"  [cyan]→ phase {phase}[/cyan] {phase_entry.label} on "
        f"[bold]{entry.target}[/bold]"
    )
    out.print(f"    {' '.join(shlex.quote(part) for part in command)}")

    branches_before = branch_lister(config.hermes_repo)

    started = time.time()
    result = dispatcher(command, child_env, config.dispatch_timeout)
    elapsed = time.time() - started

    # "Proposed" has to mean a branch a human can actually review. Exit code 0
    # only means the phase ran to completion, which it does just as happily when
    # the guard rejected the candidate or the rewrite came back identical.
    # Comparing the evolve/ refs before and after is the observable difference.
    new_branches = sorted(branch_lister(config.hermes_repo) - branches_before)
    if result.returncode != 0:
        status = DispatchStatus.FAILED
    elif new_branches:
        status = DispatchStatus.PROPOSED
    else:
        status = DispatchStatus.NO_CHANGE
    dispatch = Dispatch(
        target=entry.target,
        target_type=entry.target_type,
        phase=phase,
        module=module,
        command=command,
        status=status,
        reason=(
            f"branch {new_branches[0]}"
            if status is DispatchStatus.PROPOSED
            else "ran cleanly but produced no branch: nothing was deployable"
            if status is DispatchStatus.NO_CHANGE
            else f"exit code {result.returncode}"
        ),
        returncode=result.returncode,
        duration_s=elapsed,
        output_tail=result.output,
    )

    if status is DispatchStatus.PROPOSED:
        out.print(f"  [green]✓ proposed[/green] {entry.target} in {elapsed:.1f}s")
    elif status is DispatchStatus.NO_CHANGE:
        # The status has been distinct from FAILED since the branch comparison
        # went in, but this line was not: a run that did everything right and
        # found nothing to deploy printed "✗ failed ... exit 0", which is both
        # red and self-contradictory. Nothing deployable is an ordinary outcome
        # of a guard doing its job, not a failure.
        out.print(
            f"  [yellow]- no change[/yellow] {entry.target}: ran cleanly in "
            f"{elapsed:.1f}s and produced no branch"
        )
        if result.output:
            out.print(f"    [dim]{result.output.splitlines()[-1]}[/dim]")
    else:
        out.print(
            f"  [red]✗ failed[/red] {entry.target}: exit {result.returncode}"
        )
        if result.output:
            out.print(f"    [dim]{result.output.splitlines()[-1]}[/dim]")

    _record_outcome(store, entry, dispatch, now)
    return dispatch


def _record_outcome(
    store: MetricStore, entry: TriageEntry, dispatch: Dispatch, now: float
) -> None:
    """Write the dispatch back into the same history triage reads.

    Value is 1.0 only for a proposal that was actually produced. A skip and a
    failure both record 0.0, distinguished by ``metadata['status']``, so a later
    cycle can see that the loop has been unable to act rather than assuming the
    target was never picked.
    """
    store.record(
        OPTIMIZATION_RUN,
        entry.target,
        1.0 if dispatch.status is DispatchStatus.PROPOSED else 0.0,
        source="loop",
        timestamp=now,
        metadata={
            "status": dispatch.status.value,
            "phase": dispatch.phase,
            "module": dispatch.module,
            "target_type": entry.target_type.value,
            "triage_score": round(entry.score, 4),
            "reason": dispatch.reason,
        },
    )


def _print_footer(report: CycleReport, out: Console) -> None:
    _step(out, "Result")
    if report.status is CycleStatus.PROPOSED:
        names = ", ".join(d.target for d in report.proposed)
        out.print(f"  [bold green]✓ proposed optimizations for: {names}[/bold green]")
        out.print("  A human reviews and merges. This loop never merges anything.")
    elif report.status is CycleStatus.DRY_RUN:
        out.print("  [yellow]Dry run complete. Nothing was run and nothing was recorded.[/yellow]")
    elif report.status is CycleStatus.SKIPPED:
        for dispatch in report.skipped:
            out.print(f"  [yellow]○ {dispatch.target}: {dispatch.reason}[/yellow]")
        out.print("  [yellow]Cycle skipped. No optimization was attempted.[/yellow]")
    elif report.status is CycleStatus.FAILED:
        out.print("  [red]✗ every dispatched optimization failed.[/red]")
    elif report.status is CycleStatus.NO_CHANGE:
        out.print(
            "  [yellow]Every dispatched phase ran cleanly and found nothing "
            "deployable. No proposal this cycle.[/yellow]"
        )
    else:
        out.print("  Nothing to do this cycle.")


# ──────────────────────────────────────────────────────────────────────────
# Cron
# ──────────────────────────────────────────────────────────────────────────


def cron_line(
    *,
    schedule: str = DEFAULT_SCHEDULE,
    python: Optional[str] = None,
    history_path: Optional[Path] = None,
    hermes_repo: Optional[Path] = None,
    threshold: float = 0.30,
    max_targets: int = 1,
    log_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> str:
    """Build the crontab line that runs one cycle on a schedule.

    Printed for the user to install with ``crontab -e``. This function does not
    write to any crontab, and nothing in this module does either: editing a
    user's scheduler behind their back is not a decision a CLI gets to make.
    """
    fields = schedule.split()
    if len(fields) != 5:
        raise ValueError(
            f"cron schedule needs 5 fields (minute hour day month weekday), got {len(fields)}: {schedule!r}"
        )

    history = Path(history_path) if history_path else default_history_path()
    log = Path(log_path) if log_path else history.parent / "loop.log"
    working_dir = Path(cwd) if cwd else Path.cwd()
    interpreter = python or sys.executable

    command = [
        interpreter,
        "-m",
        "evolution.monitor.loop",
        "--once",
        "--history-path",
        str(history),
        "--threshold",
        f"{threshold:g}",
        "--max-targets",
        str(max_targets),
    ]
    if hermes_repo:
        command += ["--hermes-repo", str(hermes_repo)]

    quoted = " ".join(shlex.quote(part) for part in command)
    # mkdir -p before the redirect. The log defaults under the output directory,
    # which does not exist until a run creates it, and a cron job whose very
    # first action is a redirect into a missing directory fails before it starts
    # and leaves nothing behind to explain why.
    log_dir = shlex.quote(str(log.parent))
    return (
        f"{' '.join(fields)} cd {shlex.quote(str(working_dir))} && "
        f"mkdir -p {log_dir} && "
        f"{quoted} >> {shlex.quote(str(log))} 2>&1"
    )


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def _resolve_repo(hermes_repo: Optional[str]) -> Optional[Path]:
    """Best-effort repo resolution. Absent is a skip reason, not a crash."""
    try:
        return resolve_hermes_agent_path(hermes_repo)
    except FileNotFoundError:
        return None


def show_status(
    store: MetricStore,
    triage_config: TriageConfig,
    *,
    now: Optional[float] = None,
    out: Optional[Console] = None,
) -> list[TriageEntry]:
    """Report the current ranking without running or dispatching anything."""
    out = out or console
    out.print("\n[bold cyan]Hermes Agent Self-Evolution[/bold cyan] - monitor status")
    out.print(f"  History: {store.path}")

    points = store.load()
    out.print(f"  Points: {len(points)}" + (
        f" [yellow]({store.skipped_lines} unreadable lines skipped)[/yellow]"
        if store.skipped_lines
        else ""
    ))

    entries = AutoTriage(store, triage_config).rank(now=now)
    _step(out, "Triage")
    if not entries:
        out.print("  No candidates in the window.")
    else:
        _render_ranking(entries, out)

    out.print("\n  Run one cycle with --once, preview it with --once --dry-run,")
    out.print("  or print an installable schedule with --emit-cron.")
    return entries


def run_monitor(
    once: bool = False,
    dry_run: bool = False,
    hermes_repo: Optional[str] = None,
    threshold: float = 0.30,
    max_targets: int = 1,
    emit_cron: bool = False,
    history_path: Optional[str] = None,
    schedule: str = DEFAULT_SCHEDULE,
    window_days: float = 30.0,
    min_samples: int = 5,
    cooldown_days: float = 14.0,
    iterations: Optional[int] = None,
    out: Optional[Console] = None,
) -> int:
    """CLI body. Returns the process exit code."""
    out = out or console
    history = Path(history_path) if history_path else default_history_path()
    repo = _resolve_repo(hermes_repo)

    if emit_cron:
        try:
            line = cron_line(
                schedule=schedule,
                history_path=history,
                hermes_repo=repo,
                threshold=threshold,
                max_targets=max_targets,
            )
        except ValueError as exc:
            out.print(f"[red]✗ {exc}[/red]")
            return 2
        out.print("\n[bold]Install this line with 'crontab -e':[/bold]\n")
        # Printed unstyled so a copy-paste carries no markup.
        print(line)
        out.print(
            "\n  Nothing was installed. This command only prints the line.\n"
        )
        return 0

    store = MetricStore(history)
    triage_config = TriageConfig(
        failure_threshold=threshold,
        window_days=window_days,
        min_samples=min_samples,
    )

    if not once:
        show_status(store, triage_config, out=out)
        return 0

    config = LoopConfig(
        hermes_repo=repo,
        triage=triage_config,
        max_targets=max_targets,
        cooldown_days=cooldown_days,
        iterations=iterations,
    )
    report = run_cycle(config, store, dry_run=dry_run, out=out)

    # A skipped or empty cycle is a normal outcome for an unattended job and
    # exits clean. A dispatched optimization that crashed is worth an alert.
    return 1 if report.status is CycleStatus.FAILED else 0


@click.command()
@click.option("--once", is_flag=True, help="Run a single cycle end to end, then exit")
@click.option("--dry-run", is_flag=True, help="Show the plan without running or recording anything")
@click.option("--hermes-repo", default=None, help="Path to the hermes-agent repo")
@click.option("--threshold", default=0.30, show_default=True,
              help="Failure rate at or above which a target auto-triggers optimization")
@click.option("--max-targets", default=1, show_default=True,
              help="Maximum targets to dispatch in one cycle")
@click.option("--emit-cron", is_flag=True, help="Print an installable crontab line and exit")
@click.option("--history-path", default=None, help="Metric history JSONL (default: output/monitor/metrics.jsonl)")
@click.option("--schedule", default=DEFAULT_SCHEDULE, show_default=True,
              help="Cron schedule used by --emit-cron")
@click.option("--window-days", default=30.0, show_default=True, help="Trailing window triage looks at")
@click.option("--min-samples", default=5, show_default=True,
              help="Observations a target needs before it can trigger")
@click.option("--cooldown-days", default=14.0, show_default=True,
              help="Days to wait before re-proposing the same target")
@click.option("--iterations", default=None, type=int,
              help="Iterations to pass to the phase entry point (default: its own)")
def main(once, dry_run, hermes_repo, threshold, max_targets, emit_cron, history_path,
         schedule, window_days, min_samples, cooldown_days, iterations):
    """Monitor performance, triage the weakest target, and propose an optimization.

    With no flags this reports the current ranking and exits. Nothing is ever
    merged or deployed: every cycle stops at a proposal for a human to review.
    """
    code = run_monitor(
        once=once,
        dry_run=dry_run,
        hermes_repo=hermes_repo,
        threshold=threshold,
        max_targets=max_targets,
        emit_cron=emit_cron,
        history_path=history_path,
        schedule=schedule,
        window_days=window_days,
        min_samples=min_samples,
        cooldown_days=cooldown_days,
        iterations=iterations,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
