"""Evolve hermes-agent tool implementation code against a real bug.

Usage:
    python -m evolution.code.evolve_tool_code --tool file_tools --bug-issue 742 \\
        --repro-script repros/issue_742.py --iterations 10

This is Phase 4, the tier PLAN.md calls the highest risk: everything else
evolves text that an LLM reads, this evolves code that the interpreter runs.
The shape of the run reflects that.

    resolve target      one file, inside the hermes-agent checkout
    load reproduction   the specific bug this run is aimed at
    snapshot baseline   tests green, bug reproducing - or there is nothing to do
    ask the evolver     Darwinian Evolver, as an external CLI subprocess
    guardrails first    safety.py, before anything expensive runs
    then fitness        pytest as a hard gate, then benchmark, bug, quality
    rank honestly       the margin over the runner-up, and whether it means anything
    emit a branch       and a diff, and a PR body, and stop

Every number this command prints carries what is behind it. A score appears
with its evidence coverage, so a 0.85 from one heuristic never looks like a
0.85 from tests plus a benchmark plus a reproduction. A reproduction verdict
appears with its fix rate and interval, and ``--repro-runs`` is how a flaky
repro gets caught instead of believed. A winner appears with its margin over
the runner-up, and when that margin is inside what the score can resolve the
report says the pick is arbitrary rather than dressing a coin flip as a
result.

Three hard constraints shape the code below.

**Licensing.** Darwinian Evolver is AGPL v3. It is invoked as an external
process and nothing from it is ever imported, so no AGPL code is linked into
this MIT-licensed package. When it is not installed the run stops with a
non-zero exit and says so. It does not quietly substitute a weaker mutation
source and present the result as evolution.

**Containment.** The evolver is a binary this package does not control, and a
candidate is code that binary produced, so neither ever executes in the
operator's real checkout or with the operator's environment. Both run inside
:class:`evolution.code.sandbox.CodeSandbox`: a disposable clone of the repo
with no remotes and no credential helper, an environment built from an
explicit allowlist, and an OS-level enforcer (bubblewrap on Linux,
sandbox-exec on macOS) that denies network, hides the user's home directory
and makes everything outside the workspace read-only. When no enforcer is
available the run refuses to start - exit code 5 - rather than running
unprotected; ``--unsandboxed`` is the explicit, named waiver. Candidate
``path`` values are bounded to the run's own directories in every mode,
waiver included.

**No auto-merge.** PLAN.md requires human review of every line of evolved
code, so the deliverable is a git branch, a diff and a PULL_REQUEST.md that
carries the scores, the evidence, the cost and every candidate the guardrails
threw out. This command never merges, and always puts the operator back on the
branch they started on. It does not push and does not open a PR unless asked:
``--push`` and ``--open-pr`` are separate, and both default to off, because an
optimization run that phoned out to GitHub on its own would be a bad surprise.

The branch here is the one ``CodeOrganism`` already created, so there is no
second branch mechanism. Only the body is new.

Exit codes:
    0   the run completed (a winner, or nothing that survived the guardrails)
    1   setup problem: no repo, no target, dirty tree, red baseline
    2   Darwinian Evolver is not installed
    3   the evolver ran but produced no candidate to score
    4   the run completed, but an explicitly requested push or PR open failed
    5   no OS sandbox is available and --unsandboxed was not given
"""

from __future__ import annotations

import functools
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import resolve_hermes_agent_path
from evolution.core.cost import CostReport, UsageTracker
from evolution.core.gates import GateStatus, run_benchmark_gate
from evolution.core.pr_builder import (
    PullRequestPlan,
    RejectedCandidate,
    ScoreLine,
    render_body,
)
# organism.py has a GitError of its own for the branch work, so the deployment
# one is spelled out rather than shadowing it.
from evolution.core.pr_builder import GitError as DeploymentGitError
from evolution.code.fitness_code import (
    BaselineSnapshot,
    BugReproduction,
    CandidateRanking,
    CodeFitness,
    CodeFitnessEvaluator,
    PerTestPytestRunner,
    ReproStatus,
    rank_candidates,
)
from evolution.code.organism import (
    CodeOrganism,
    Mutation,
    OrganismError,
    git_available,
    is_git_repo,
)
from evolution.code.sandbox import (
    UNSANDBOXED,
    CodeSandbox,
    SandboxError,
    SandboxUnavailable,
    bounded_candidate_path,
    command_read_roots,
    executable_read_roots,
    require_enforcer,
)

console = Console()

__all__ = [
    "EvolverError",
    "EvolverNotInstalled",
    "SandboxError",
    "SandboxUnavailable",
    "UNSANDBOXED",
    "CodeSandbox",
    "Candidate",
    "EvolverJob",
    "ExternalEvolver",
    "BugFixBrief",
    "MUTATION_CONSTRAINTS",
    "PHASE_LABEL",
    "OPTIMIZER_LABEL",
    "EVOLVER_COST_NOTE",
    "REVIEW_NOTE",
    "find_evolver",
    "resolve_tool_file",
    "build_objective",
    "code_score_lines",
    "code_rejected_candidates",
    "code_statistics",
    "code_gate_lines",
    "build_code_pull_request",
    "evolve_tool_code",
    "main",
]


# Commands Darwinian Evolver is plausibly installed as. An explicit
# --evolver-cmd or DARWINIAN_EVOLVER_CMD beats all of them.
EVOLVER_CANDIDATE_COMMANDS = ("darwinian-evolver", "darwinian_evolver", "devolve")
EVOLVER_ENV_VAR = "DARWINIAN_EVOLVER_CMD"

# Handed to the evolver in the job spec and enforced afterwards by safety.py.
# Telling the mutation engine the rules is cheaper than rejecting everything
# it sends back, but the enforcement is what actually holds.
MUTATION_CONSTRAINTS = (
    "Do not change any function signature: names, parameters, defaults and "
    "star-args are frozen.",
    "Do not change, add or remove any registry.register(...) call.",
    "Do not reduce error handling: try, except, raise and finally coverage may "
    "not decrease, module-wide or in any individual function.",
    "Do not remove assertions, validation guards or early error returns.",
    "The full pytest suite must pass; a single failing test rejects the change.",
    "Change one file only, and change as little of it as possible.",
)


class EvolverError(RuntimeError):
    """Raised when the external evolver fails or returns nothing usable."""


class EvolverNotInstalled(EvolverError):
    """Raised when no Darwinian Evolver CLI can be found."""


class TargetNotFound(RuntimeError):
    """Raised when the requested tool file does not exist in the repo."""


# ──────────────────────────────────────────────────────────────────────────
# Target resolution
# ──────────────────────────────────────────────────────────────────────────


def resolve_tool_file(repo: Path, tool: str) -> Path:
    """Resolve ``--tool`` to a file inside *repo*.

    Accepts a bare module name (``file_tools``), a filename
    (``file_tools.py``), a repo-relative path (``tools/file_tools.py``) or an
    absolute path. Refuses anything outside the repo, since the whole point of
    the organism is that exactly one tracked file moves.
    """
    repo = Path(repo).expanduser().resolve()
    raw = tool.strip()

    if Path(raw).is_absolute():
        candidates = [Path(raw)]
    else:
        stem = raw[:-3] if raw.endswith(".py") else raw
        candidates = [
            repo / raw,
            repo / f"{stem}.py",
            repo / "tools" / f"{stem}.py",
            repo / "agent" / f"{stem}.py",
        ]

    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(repo)
            except ValueError as exc:
                raise TargetNotFound(
                    f"{resolved} is outside the hermes-agent repo {repo}"
                ) from exc
            return resolved

    raise TargetNotFound(
        f"could not find a source file for --tool {tool!r} under {repo} "
        "(tried the repo root, tools/ and agent/)"
    )


# ──────────────────────────────────────────────────────────────────────────
# External evolver
# ──────────────────────────────────────────────────────────────────────────


def find_evolver(explicit: Optional[str] = None, env: Optional[dict] = None) -> list[str]:
    """Locate the Darwinian Evolver CLI, or raise :class:`EvolverNotInstalled`.

    Returns the command as an argv list so an operator can point at a wrapper
    script with its own flags (``--evolver-cmd "uvx darwinian-evolver --quiet"``).
    """
    import os

    environ = env if env is not None else os.environ
    sources = [explicit, environ.get(EVOLVER_ENV_VAR)]

    for source in sources:
        if not source:
            continue
        argv = shlex.split(source)
        if not argv:
            continue
        if shutil.which(argv[0]) or Path(argv[0]).expanduser().is_file():
            return argv
        raise EvolverNotInstalled(
            f"evolver command not executable: {argv[0]} "
            f"(from {'--evolver-cmd' if source == explicit else EVOLVER_ENV_VAR})"
        )

    for name in EVOLVER_CANDIDATE_COMMANDS:
        found = shutil.which(name)
        if found:
            return [found]

    raise EvolverNotInstalled(
        "Darwinian Evolver is not installed. Tried: "
        + ", ".join(EVOLVER_CANDIDATE_COMMANDS)
        + f". Install it separately (it is AGPL v3 and is only ever run as an "
        f"external process), then point at it with --evolver-cmd or "
        f"{EVOLVER_ENV_VAR}."
    )


@dataclass
class Candidate:
    """One proposed rewrite of the target file."""

    index: int
    source: str
    notes: str = ""
    origin: str = ""

    @property
    def label(self) -> str:
        """Short stable label for this candidate, such as ``c03``."""
        return f"c{self.index:02d}"

    def to_dict(self) -> dict:
        """Serialise the candidate for the run artifacts."""
        return {
            "index": self.index,
            "label": self.label,
            "notes": self.notes,
            "origin": self.origin,
            "chars": len(self.source),
        }


@dataclass
class EvolverJob:
    """The job spec handed to the external evolver.

    This is our adapter contract, written to a JSON file and passed by path.
    Keeping it a file rather than a pipe means a failed run leaves the exact
    request behind for inspection.
    """

    target_path: str
    source: str
    objective: str
    iterations: int
    constraints: tuple[str, ...] = MUTATION_CONSTRAINTS
    bug_issue: Optional[str] = None
    reproduction: Optional[str] = None
    reproduction_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise the job so the external evolver can read it as JSON."""
        return {
            "target_path": self.target_path,
            "source": self.source,
            "objective": self.objective,
            "iterations": self.iterations,
            "constraints": list(self.constraints),
            "bug_issue": self.bug_issue,
            "reproduction": self.reproduction,
            "reproduction_path": self.reproduction_path,
        }


class ProposesCandidates(Protocol):
    """What :func:`evolve_tool_code` needs from a mutation source."""

    def propose(self, job: EvolverJob) -> list[Candidate]:  # pragma: no cover
        """Return candidate rewrites of the target described by *job*."""
        ...


class ExternalEvolver:
    """Drive Darwinian Evolver as a subprocess and collect its candidates.

    Never imported, only executed: the package is AGPL v3 and this one is MIT.
    And never executed bare: the evolver is a binary this package does not
    control, so it runs inside a :class:`CodeSandbox` - a disposable checkout,
    an allowlisted environment and an OS-level enforcer - unless the operator
    passed the ``UNSANDBOXED`` sentinel by name. With no *sandbox* argument a
    fresh sandbox is built for the call and torn down afterwards, and a
    machine with no enforcer refuses (:class:`SandboxUnavailable`) instead of
    quietly degrading.

    The adapter reads candidates from, in order of preference:

    1. ``<output>/candidates/*.py`` - one file per candidate
    2. ``<output>/candidates.jsonl`` - one JSON object per line, with a
       ``source`` string or a ``path`` to read
    3. stdout, as JSON lines of the same shape

    A ``path`` is honoured only inside the run's own directories - the
    workdir and the checkout the evolver actually saw. Absolute paths
    elsewhere, ``..`` components and symlinks that lead outside are refused,
    sandboxed or not: the evolver's output must not become a read primitive
    against the operator's filesystem.

    A non-zero exit with no candidates is an error. A non-zero exit that still
    produced candidates is reported and the candidates are scored anyway; the
    guardrails decide, not the evolver's opinion of its own run.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        repo: Path,
        workdir: Path,
        timeout: int = 3600,
        sandbox: object = None,
        env_passthrough: Sequence[str] = (),
        allow_network: bool = False,
    ) -> None:
        self.cmd = list(cmd)
        self.repo = Path(repo)
        # Absolute: the evolver runs with cwd set to a checkout, not the
        # directory the operator launched from, so a relative job or output
        # path would resolve against the wrong root.
        self.workdir = Path(workdir).expanduser().resolve()
        self.timeout = timeout
        self.sandbox = sandbox
        self.env_passthrough = tuple(env_passthrough)
        self.allow_network = allow_network
        self.last_stdout = ""
        self.last_stderr = ""
        self.last_returncode: Optional[int] = None

    def propose(self, job: EvolverJob) -> list[Candidate]:
        """Run the external evolver as a subprocess and collect its candidates.

        The job is handed over as JSON on disk and the results are read back from
        an output directory, so the evolver stays a separate process sharing no
        state with this one.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        if self.sandbox is UNSANDBOXED:
            return self._propose(None, job)
        if isinstance(self.sandbox, CodeSandbox):
            return self._propose(self.sandbox, job)
        # No sandbox was provided: build one for this call, fail closed when
        # the machine cannot enforce it, and leave nothing behind but the
        # workdir artifacts.
        roots = command_read_roots(self.cmd)
        ephemeral = CodeSandbox(
            self.repo,
            workdir=self.workdir,
            read_roots=roots,
            env_passthrough=self.env_passthrough,
            allow_network=self.allow_network,
        )
        try:
            return self._propose(ephemeral, job)
        finally:
            ephemeral.cleanup()

    def _propose(
        self, sandbox: Optional[CodeSandbox], job: EvolverJob
    ) -> list[Candidate]:
        job_path = self.workdir / "job.json"
        out_dir = self.workdir / "evolver_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        blob = job.to_dict()
        if sandbox is not None and blob.get("reproduction_path"):
            # The sandbox cannot read the operator's copy of the script, so
            # the job points at a copy inside the workspace - unless it
            # already does.
            script = Path(str(blob["reproduction_path"]))
            roots = self._candidate_roots(sandbox)
            inside = bounded_candidate_path(
                str(script), base=self.workdir, roots=roots
            )
            if inside is None:
                blob["reproduction_path"] = (
                    str(sandbox.import_file(script)) if script.is_file() else None
                )
        job_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")

        cmd = [*self.cmd, "--job", str(job_path), "--output", str(out_dir)]
        try:
            if sandbox is None:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(self.repo),
                )
            else:
                proc = sandbox.run(cmd, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise EvolverError(
                f"evolver timed out after {self.timeout}s"
            ) from exc
        except OSError as exc:
            raise EvolverNotInstalled(f"could not run {cmd[0]}: {exc}") from exc

        self.last_stdout = proc.stdout or ""
        self.last_stderr = proc.stderr or ""
        self.last_returncode = proc.returncode

        base = sandbox.checkout if sandbox is not None else self.repo
        candidates = self._collect(
            out_dir,
            self.last_stdout,
            base=base,
            roots=self._candidate_roots(sandbox),
        )
        if not candidates:
            tail = "\n".join(
                (self.last_stderr or self.last_stdout).strip().splitlines()[-15:]
            )
            raise EvolverError(
                f"evolver exited {proc.returncode} and produced no candidates"
                + (f":\n{tail}" if tail else "")
            )
        return candidates

    # ── candidate collection ────────────────────────────────────────────

    def _candidate_roots(
        self, sandbox: Optional[CodeSandbox]
    ) -> tuple[Path, ...]:
        """Where a candidate ``path`` may legitimately point.

        The run's own workdir and the tree the evolver was actually shown.
        Everything else on the machine is off the table, in both modes.
        """
        if sandbox is None:
            return (self.workdir, self.repo)
        return (self.workdir, sandbox.workspace)

    def _collect(
        self, out_dir: Path, stdout: str, *, base: Path, roots: Sequence[Path]
    ) -> list[Candidate]:
        candidates = self._from_directory(out_dir, roots=roots)
        if candidates:
            return candidates
        candidates = self._from_jsonl(
            out_dir / "candidates.jsonl", base=base, roots=roots
        )
        if candidates:
            return candidates
        return self._from_stdout(stdout, base=base, roots=roots)

    def _from_directory(
        self, out_dir: Path, *, roots: Sequence[Path]
    ) -> list[Candidate]:
        folder = out_dir / "candidates"
        if not folder.is_dir():
            return []
        out: list[Candidate] = []
        index = 0
        for path in sorted(folder.glob("*.py")):
            # Collection runs outside the sandbox, so a symlink the evolver
            # planted here would otherwise be followed with this process's
            # own privileges. Same bound as an explicit ``path`` entry.
            resolved = bounded_candidate_path(str(path), base=folder, roots=roots)
            if resolved is None:
                continue
            try:
                source = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            index += 1
            out.append(Candidate(index=index, source=source, origin=str(path)))
        return out

    def _from_jsonl(
        self, path: Path, *, base: Path, roots: Sequence[Path]
    ) -> list[Candidate]:
        # The jsonl file itself gets the same treatment as the paths inside
        # it: a symlink here is an arbitrary-read primitive too.
        if bounded_candidate_path(str(path), base=path.parent, roots=roots) is None:
            return []
        if not path.is_file():
            return []
        out: list[Candidate] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            candidate = self._candidate_from_line(
                line, len(out) + 1, str(path), base=base, roots=roots
            )
            if candidate:
                out.append(candidate)
        return out

    def _from_stdout(
        self, stdout: str, *, base: Path, roots: Sequence[Path]
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            candidate = self._candidate_from_line(
                line, len(out) + 1, "stdout", base=base, roots=roots
            )
            if candidate:
                out.append(candidate)
        return out

    def _candidate_from_line(
        self,
        line: str,
        index: int,
        origin: str,
        *,
        base: Path,
        roots: Sequence[Path],
    ) -> Optional[Candidate]:
        line = line.strip()
        if not line:
            return None
        try:
            blob = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(blob, dict):
            return None

        source = blob.get("source")
        if source is None and blob.get("path"):
            resolved = bounded_candidate_path(
                str(blob["path"]), base=base, roots=roots
            )
            if resolved is None:
                return None
            try:
                source = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return None
        if not isinstance(source, str) or not source.strip():
            return None

        return Candidate(
            index=index,
            source=source,
            notes=str(blob.get("notes") or blob.get("rationale") or ""),
            origin=origin,
        )


# ──────────────────────────────────────────────────────────────────────────
# Mutation brief
# ──────────────────────────────────────────────────────────────────────────


class BugFixBrief(dspy.Signature):
    """Turn a bug report and its reproduction into a precise mutation brief.

    The brief is the objective handed to the external evolver. It should name
    the observable defect, the input that triggers it and the behaviour that
    would be correct. It must not propose a design change, and must not ask
    for anything the constraints forbid.
    """

    tool_module: str = dspy.InputField(desc="The tool module being evolved")
    bug_report: str = dspy.InputField(desc="Issue number, title and description")
    reproduction: str = dspy.InputField(desc="Source of the reproduction script")
    constraints: str = dspy.InputField(desc="Rules the mutation must respect")
    objective: str = dspy.OutputField(
        desc="A short, concrete statement of what the mutation must achieve"
    )


def _lm_configured() -> bool:
    """True when DSPy has a language model to call.

    Checked rather than assumed: this command is useful offline, and a fitness
    run must never die because nobody exported an API key.
    """
    try:
        return getattr(dspy.settings, "lm", None) is not None
    except Exception:
        return False


def _template_objective(
    tool_module: str, bug_issue: Optional[str], reproduction: Optional[str]
) -> str:
    lines = [
        f"Fix the defect in {tool_module} without changing its public shape.",
    ]
    if bug_issue:
        lines.append(f"Target bug: {bug_issue}.")
    if reproduction:
        lines.append(
            "The reproduction script exits non-zero while the bug is present "
            "and zero once it is fixed."
        )
    lines.append("Make the smallest change that resolves it.")
    return " ".join(lines)


def build_objective(
    tool_module: str,
    bug_issue: Optional[str] = None,
    reproduction: Optional[str] = None,
    constraints: Iterable[str] = MUTATION_CONSTRAINTS,
    predictor=None,
) -> str:
    """Compose the objective handed to the evolver.

    Uses the LLM when one is configured, and a deterministic template when it
    is not. Either way the constraints are enforced afterwards by safety.py,
    so a weak brief costs candidates, never correctness.
    """
    if predictor is None:
        if not _lm_configured():
            return _template_objective(tool_module, bug_issue, reproduction)
        predictor = dspy.Predict(BugFixBrief)

    try:
        result = predictor(
            tool_module=tool_module,
            bug_report=bug_issue or "no issue supplied",
            reproduction=reproduction or "no reproduction supplied",
            constraints="\n".join(constraints),
        )
    except Exception as exc:  # a brief is a nicety, not a dependency
        console.print(f"[yellow]![/yellow] Could not draft an LLM brief ({exc})")
        return _template_objective(tool_module, bug_issue, reproduction)

    objective = str(getattr(result, "objective", "") or "").strip()
    return objective or _template_objective(tool_module, bug_issue, reproduction)


# ──────────────────────────────────────────────────────────────────────────
# Run record
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class CandidateOutcome:
    """A candidate, its commit and its verdict, kept together for the report."""

    candidate: Candidate
    fitness: CodeFitness
    mutation: Optional[Mutation] = None

    def to_dict(self) -> dict:
        """Serialise the candidate, its fitness and its commit together."""
        return {
            "candidate": self.candidate.to_dict(),
            "fitness": self.fitness.to_dict(),
            "mutation": self.mutation.to_dict() if self.mutation else None,
        }


# ──────────────────────────────────────────────────────────────────────────
# Pull request
# ──────────────────────────────────────────────────────────────────────────
#
# PLAN.md constraint 5 is "Deployment via PR (Never Direct Commit)". Phase 4
# already ends on a branch CodeOrganism created and a diff it produced, so
# nothing here creates a second branch: this section renders the body PLAN.md
# asks for and hands back a plan bound to the branch that already exists.
# Writing that body is local. Pushing it and opening the PR are not, and are
# never done unless the operator asked for them by name.


PHASE_LABEL = "Phase 4 (tool implementation code)"

OPTIMIZER_LABEL = "Darwinian Evolver, driven as an external CLI subprocess"

# The mutation engine is a separate process. Its model calls never enter dspy's
# history, so no tracker inside this process can see them, and a total that
# quietly omits the engine that did the actual work reads as the cost of the
# run when it is not. Naming the gap costs a sentence.
EVOLVER_COST_NOTE = (
    "Cost above covers this pipeline's own model calls only. Darwinian Evolver "
    "runs as a separate subprocess, so its model usage never reaches dspy's "
    "history and is not included. PLAN.md budgets roughly $2-9 per task for it, "
    "so read the true cost of this run as the figure above plus whatever the "
    "evolver spent."
)

REVIEW_NOTE = (
    "PLAN.md requires human review of every line of evolved code. Nothing was "
    "merged and nothing merges itself - this is a branch, a diff and a body, "
    "and it stays that way until a person decides otherwise."
)

# Long enough to show a real fix, short enough that a reviewer still scrolls.
MAX_REPORTED_TESTS = 5


def _one_line(text: object) -> str:
    """Collapse anything into a single line of prose."""
    return " ".join(str(text).split())


def _cell(text: object) -> str:
    """A single line that is safe inside a markdown table cell."""
    return _one_line(text).replace("|", "\\|")


def code_score_lines(
    baseline: BaselineSnapshot, winner: CandidateOutcome
) -> list[ScoreLine]:
    """Baseline versus winner on everything the run actually measured.

    The composite row carries its evidence coverage in the same cell as the
    score, for the reason the console table does: a 0.85 backed by tests, a
    benchmark and a reproduction and a 0.85 backed by one quality heuristic
    print as the same number, and only one of them is worth a reviewer's
    afternoon.
    """
    fitness = winner.fitness

    evidence = f"evidence {fitness.evidence_coverage:.0%}"
    if fitness.missing_evidence:
        evidence += f", no {' or '.join(fitness.missing_evidence)}"
    lines = [
        ScoreLine(
            "composite fitness",
            0.0,
            fitness.total,
            _cell(
                f"{evidence}. The baseline scores 0 by construction: a candidate "
                "identical to it is refused before it is ever scored"
            ),
        )
    ]

    trials = fitness.repro_trials
    if trials is not None and trials.measured_runs:
        baseline_trials = baseline.repro_trials
        before = (
            baseline_trials.fix_rate
            if baseline_trials is not None and baseline_trials.measured_runs
            else 0.0
        )
        lines.append(
            ScoreLine(
                "bug reproduction fix rate",
                before,
                trials.fix_rate,
                _cell(trials.describe()),
            )
        )

    suite = fitness.suite
    if suite is not None and suite.n:
        lines.append(
            ScoreLine(
                "test suite pass rate",
                suite.paired.baseline_rate,
                suite.paired.candidate_rate,
                _cell(
                    f"{suite.n} test(s) paired by node id, {suite.verdict}"
                ),
            )
        )

    baselines = baseline.benchmark_baselines()
    for result in fitness.benchmark_results:
        if result.score is None:
            continue
        before = baselines.get(result.name, result.baseline)
        if before is None:
            # No baseline measurement means no before/after to state. The gate
            # line still reports what the benchmark said.
            continue
        lines.append(ScoreLine(result.name, before, result.score, _cell(result.message)))

    return lines


def _rejection_detail(fitness: CodeFitness) -> str:
    """Why one candidate was refused, in the guardrail's own words.

    A reviewer of evolved *code* needs this more than a reviewer of evolved
    text does. "3 candidates rejected" says a filter ran; "the signature guard
    caught a new parameter on a public function" says what it caught.
    """
    if not fitness.safety.passed:
        spelled = [
            f"{failure.name}: {'; '.join(failure.violations) or failure.message}"
            for failure in fitness.safety.failures
        ]
        return _one_line(
            "refused by the safety guardrails - " + " / ".join(spelled)
        )

    if fitness.pytest_result.status is GateStatus.FAILED:
        detail = (
            "refused by the hard pytest gate - " + fitness.pytest_result.message
        )
        suite = fitness.suite
        if suite is not None and suite.newly_failing:
            shown = ", ".join(suite.newly_failing[:MAX_REPORTED_TESTS])
            extra = len(suite.newly_failing) - MAX_REPORTED_TESTS
            detail += f" (newly failing: {shown}"
            detail += f", and {extra} more)" if extra > 0 else ")"
        return _one_line(detail)

    return _one_line(fitness.rejection_reason or "refused, no reason recorded")


def code_rejected_candidates(
    outcomes: Sequence[CandidateOutcome],
) -> list[RejectedCandidate]:
    """Every candidate the run threw away, with the reason it was thrown away."""
    return [
        RejectedCandidate(o.candidate.label, _rejection_detail(o.fitness))
        for o in outcomes
        if not o.fitness.accepted
    ]


def code_statistics(
    baseline: BaselineSnapshot,
    winner: CandidateOutcome,
    ranking: Optional[CandidateRanking] = None,
) -> str:
    """The evidence section: the paired suite and the measured fix rate.

    Both of these the phase already computes. Leaving them in the metrics file
    and out of the PR body would mean the reviewer with the least context gets
    the least evidence.
    """
    fitness = winner.fitness
    lines: list[str] = []

    suite = fitness.suite
    if suite is not None:
        lines.append(
            f"- Paired test suite, baseline versus {winner.candidate.label}, "
            f"McNemar on outcomes paired by node id: {_one_line(suite.describe())}"
        )
        if suite.newly_failing:
            shown = ", ".join(suite.newly_failing[:MAX_REPORTED_TESTS])
            lines.append(f"  - Newly failing: {shown}")
    else:
        lines.append(
            "- Paired test suite: not available. No per-test outcomes were "
            "recorded on one side, so there was nothing legitimate to pair - "
            "which is not the same as no change."
        )

    trials = fitness.repro_trials
    if trials is not None and trials.measured_runs:
        lines.append(
            f"- Reproduction, {trials.measured_runs} measured run(s): "
            f"{_one_line(trials.describe())} (Wilson interval at "
            f"{trials.confidence:.0%})"
        )
        if trials.power_note:
            lines.append(f"  - {_one_line(trials.power_note)}")
        baseline_trials = baseline.repro_trials
        if baseline_trials is not None and baseline_trials.measured_runs:
            lines.append(
                f"  - At baseline: {_one_line(baseline_trials.describe())}"
            )
    else:
        lines.append(
            "- Reproduction: none measured, so nothing here demonstrates that a "
            "bug was fixed. The score rests on the suite and the heuristics."
        )

    if ranking is not None:
        lines.append(f"- Ranking: {_one_line(ranking.describe())}")

    return "\n".join(lines)


def code_gate_lines(winner: CandidateOutcome) -> list[str]:
    """What gated the winner, and what each gate said."""
    fitness = winner.fitness
    gates = [
        f"safety guardrails: {len(fitness.safety.results)} check(s), "
        f"{'all passed' if fitness.safety.passed else 'failed'}",
        f"pytest (hard gate, zero tolerance): "
        f"{fitness.pytest_result.status.value} - "
        f"{_one_line(fitness.pytest_result.message)}",
    ]
    for result in fitness.benchmark_results:
        gates.append(
            f"{result.name}: {result.status.value} - {_one_line(result.message)}"
        )
    trials = fitness.repro_trials
    if trials is not None:
        gates.append(f"bug reproduction: {_one_line(trials.message)}")
    return gates


def build_code_pull_request(
    *,
    repo: Path,
    branch: str,
    target: str,
    baseline: BaselineSnapshot,
    winner: CandidateOutcome,
    diff: str,
    outcomes: Sequence[CandidateOutcome] = (),
    ranking: Optional[CandidateRanking] = None,
    cost: Optional[CostReport] = None,
    iterations: Optional[int] = None,
    bug_issue: Optional[str] = None,
    repro_script: Optional[str] = None,
    repro_runs: int = 1,
    baseline_sha: str = "",
    winner_sha: str = "",
) -> PullRequestPlan:
    """Render the PR body for the branch the organism already built.

    Deliberately not :func:`evolution.core.pr_builder.build_pull_request`:
    that one creates ``evolve/<target>-<timestamp>`` and commits into it, and
    Phase 4 already has a branch with the winning commit on it. Two branch
    mechanisms racing over one checkout is exactly the failure this phase
    cannot afford, so the plan returned here is bound to the existing branch
    and never checks anything out. ``created_branch`` is False for the same
    reason: the organism owns the restore, and ``restore()`` here must not
    second-guess it.

    Nothing in this function touches the network.
    """
    dataset_parts = [
        f"hermes-agent pytest suite ({len(baseline.test_outcomes)} test(s) "
        "recorded at baseline)"
    ]
    if repro_script:
        dataset_parts.append(
            f"reproduction {Path(repro_script).name} x{repro_runs} per candidate"
        )
    else:
        dataset_parts.append("no reproduction supplied")

    notes = [EVOLVER_COST_NOTE, REVIEW_NOTE]
    if bug_issue:
        notes.append(f"Target bug: issue {bug_issue}")
    notes.append(f"Branch: `{branch}` in {repo}")
    if baseline_sha:
        review = f"git diff {baseline_sha[:8]} {branch} -- {target}"
        notes.append(f"Review the change with: `{review}`")
    if winner_sha:
        notes.append(f"Winning commit: `{winner_sha[:8]}` ({winner.candidate.label})")

    body = render_body(
        target=target,
        phase=PHASE_LABEL,
        scores=code_score_lines(baseline, winner),
        diff=diff,
        cost=cost,
        rejected=code_rejected_candidates(outcomes),
        gates=code_gate_lines(winner),
        dataset=", ".join(dataset_parts),
        optimizer=OPTIMIZER_LABEL,
        iterations=iterations,
        statistics=code_statistics(baseline, winner, ranking),
        notes=notes,
    )

    title = f"evolve: {target}"
    if bug_issue:
        title += f" (issue {bug_issue})"

    return PullRequestPlan(
        repo=Path(repo),
        branch=branch,
        title=title,
        # The organism made this commit; nothing here commits anything.
        commit_message=(
            winner.mutation.message
            if winner.mutation
            else f"evolve({Path(target).stem}): {winner.candidate.label}"
        ),
        body=body,
        files=(target,),
        created_branch=False,
        original_ref="",
    )


# ──────────────────────────────────────────────────────────────────────────
# Console helpers
# ──────────────────────────────────────────────────────────────────────────


def _step(title: str) -> None:
    console.print(f"\n[bold cyan]── {title} ─────────────────────────────[/bold cyan]")


def _gate_icon(status: GateStatus) -> str:
    return {
        GateStatus.PASSED: "[green]✓[/green]",
        GateStatus.FAILED: "[red]✗[/red]",
        GateStatus.UNAVAILABLE: "[yellow]○[/yellow]",
        GateStatus.SKIPPED: "[dim]-[/dim]",
    }[status]


# ──────────────────────────────────────────────────────────────────────────
# The run
# ──────────────────────────────────────────────────────────────────────────


def evolve_tool_code(
    tool: str,
    bug_issue: Optional[str] = None,
    repro_script: Optional[str] = None,
    iterations: int = 10,
    hermes_repo: Optional[str] = None,
    evolver_cmd: Optional[str] = None,
    strict_gates: bool = False,
    dry_run: bool = False,
    benchmarks: Sequence[str] = (),
    python: Optional[str] = None,
    pytest_subset: Optional[Sequence[str]] = None,
    allow_dirty: bool = False,
    output_root: Optional[Path] = None,
    evolver: Optional[ProposesCandidates] = None,
    repro_runs: int = 1,
    write_pr: bool = True,
    push: bool = False,
    open_pr: bool = False,
    remote: str = "origin",
    base: str = "main",
    sandbox: object = None,
    sandbox_env: Sequence[str] = (),
    allow_network: bool = False,
) -> int:
    """Run one code-evolution pass. Returns a process exit code.

    *evolver* is an injection point: pass an object with ``propose(job)`` to
    drive a different mutation source, or a fake one in tests. Left as None,
    the Darwinian Evolver CLI is discovered and used, and its absence stops
    the run.

    *sandbox* decides where untrusted execution happens - the evolver
    subprocess and every gate that runs candidate code (pytest, the
    reproduction, benchmarks). Left as None, a :class:`CodeSandbox` is built
    for the run: a disposable clone of the repo at the baseline commit, an
    allowlisted environment, and an OS enforcer, with exit code 5 when the
    machine has no enforcer. Pass ``UNSANDBOXED`` to waive all of it by name,
    or a prebuilt :class:`CodeSandbox` to control the boundary directly.
    *sandbox_env* names parent environment variables to pass inside;
    *allow_network* opens the network inside the sandbox, which is off by
    default because an evolver that can read its workspace and reach the
    network can exfiltrate whatever it was shown.

    *repro_runs* is how many times the reproduction script runs per candidate.
    One is a single Bernoulli trial and is enough for a deterministic repro;
    raise it for anything that touches time, the filesystem or a subprocess,
    where one clean pass and a fix are not the same thing.

    *write_pr* writes PULL_REQUEST.md beside the run's other artifacts when a
    winner exists. That is a local file next to a local branch, so it happens
    by default - but only when there is something to deploy. A dry run, a run
    with no survivor and a run whose winner is identical to the baseline all
    build nothing.

    *push* and *open_pr* are the two steps that leave this machine, and both
    default to off. Each has to be asked for by name.
    """
    console.print(
        "\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] - "
        f"Phase 4: evolving code in [bold]{tool}[/bold]\n"
    )

    # ── 1. Resolve the target ───────────────────────────────────────────
    try:
        repo = resolve_hermes_agent_path(hermes_repo)
    except FileNotFoundError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    if not Path(repo).is_dir():
        console.print(f"[red]✗ hermes-agent repo not found: {repo}[/red]")
        return 1

    try:
        target = resolve_tool_file(repo, tool)
    except TargetNotFound as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    relpath = target.relative_to(Path(repo).resolve()).as_posix()
    console.print(f"  Repo:   {repo}")
    console.print(f"  Target: {relpath} ({len(target.read_text(encoding='utf-8')):,} chars)")

    if not git_available():
        console.print("[red]✗ git is not installed - code evolution needs it[/red]")
        return 1
    if not is_git_repo(Path(repo)):
        console.print(f"[red]✗ {repo} is not a git repository[/red]")
        return 1

    # ── 2. Bug reproduction ─────────────────────────────────────────────
    repro: Optional[BugReproduction] = None
    repro_source: Optional[str] = None
    if repro_script:
        repro_path = Path(repro_script).expanduser()
        if not repro_path.is_file():
            console.print(f"[red]✗ reproduction script not found: {repro_path}[/red]")
            return 1
        repro = BugReproduction(script=repro_path.resolve(), issue=bug_issue)
        try:
            repro_source = repro.script.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            repro_source = None
        console.print(f"  Repro:  {repro.script} ({repro_runs} run(s) per candidate)")
        if repro_runs == 1:
            console.print(
                "[dim]          one run is one Bernoulli trial - use --repro-runs "
                "to measure a fix rate instead[/dim]"
            )
    else:
        console.print(
            "  Repro:  [yellow]none supplied - fitness cannot prove any bug was "
            "fixed[/yellow]"
        )

    # ── 3. Locate the evolver ───────────────────────────────────────────
    evolver_argv: Optional[list[str]] = None
    if evolver is None:
        try:
            evolver_argv = find_evolver(evolver_cmd)
        except EvolverNotInstalled as exc:
            console.print(f"\n[red]✗ {exc}[/red]")
            console.print(
                "[dim]  Nothing was mutated. This command does not substitute a "
                "different mutation engine when the requested one is absent.[/dim]"
            )
            return 2
        console.print(f"  Evolver: {' '.join(evolver_argv)}")
    else:
        console.print(f"  Evolver: injected ({type(evolver).__name__})")

    # ── 4. Resolve the containment boundary, before anything mutates ────
    enforcer = None
    if sandbox is UNSANDBOXED:
        console.print(
            "  Sandbox: [red]disabled (--unsandboxed)[/red] - the evolver and "
            "candidate code run with this process's environment and filesystem"
        )
    elif isinstance(sandbox, CodeSandbox):
        console.print(f"  Sandbox: {sandbox.describe()['enforcer']} (injected)")
    else:
        try:
            enforcer = require_enforcer()
        except SandboxUnavailable as exc:
            console.print(f"\n[red]✗ {exc}[/red]")
            console.print(
                "[dim]  Nothing was mutated. Code evolution does not run "
                "contributor-controlled code without containment unless told "
                "to by name.[/dim]"
            )
            return 5
        console.print(f"  Sandbox: {enforcer.name}")

    if dry_run:
        console.print("\n[bold green]DRY RUN - setup validated successfully.[/bold green]")
        console.print(f"  Would branch from HEAD and mutate {relpath}")
        console.print(f"  Would request {iterations} iteration(s) from the evolver")
        console.print(
            "  Would gate each candidate on: safety guardrails, pytest, "
            + (", ".join(benchmarks) if benchmarks else "no benchmarks")
            + (f", bug reproduction x{repro_runs}" if repro else "")
        )
        if sandbox is UNSANDBOXED:
            console.print(
                "  Would run the evolver and candidate code [red]unsandboxed[/red]"
            )
        else:
            console.print(
                "  Would run the evolver and every candidate-executing gate in "
                "a disposable checkout, behind the sandbox above"
            )
        console.print("  Would emit a branch and a diff. Never a merge.")
        console.print(
            "  Would write PULL_REQUEST.md beside the artifacts"
            if write_pr
            else "  Would not write a PR body (--no-write-pr)"
        )
        console.print(
            f"  Would {'push to ' + remote if push else 'not push'} and "
            f"would {'open a PR against ' + base if open_pr else 'not open a PR'}"
        )
        console.print("[dim]  A dry run builds nothing: no branch, no body.[/dim]")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root or Path("output")) / "code" / Path(relpath).stem / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    outcomes: list[CandidateOutcome] = []
    winner: Optional[CandidateOutcome] = None
    exit_code = 0
    run_sandbox: Optional[CodeSandbox] = None
    owns_sandbox = False

    try:
        organism = CodeOrganism(repo, target, allow_dirty=allow_dirty)
    except OrganismError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    try:
        organism.start()
    except OrganismError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    try:
        baseline_source = organism.baseline_source
        console.print(f"  Branch: [bold]{organism.branch}[/bold] (from {organism.original_ref})")

        if isinstance(sandbox, CodeSandbox):
            run_sandbox = sandbox
        elif sandbox is not UNSANDBOXED:
            # The clone is cut at the baseline commit and the target file is
            # overwritten with the working-tree source, so an allow_dirty run
            # evaluates what the operator actually has, not what HEAD says.
            read_roots: list[Path] = list(command_read_roots(evolver_argv or []))
            read_roots.extend(executable_read_roots(python or "python"))
            try:
                run_sandbox = CodeSandbox(
                    Path(repo),
                    target_relpath=relpath,
                    baseline_sha=organism.baseline_sha,
                    baseline_source=baseline_source,
                    workdir=out_dir,
                    read_roots=read_roots,
                    env_passthrough=sandbox_env,
                    allow_network=allow_network,
                    enforcer=enforcer,
                )
            except SandboxError as exc:
                console.print(f"[red]✗ could not build the sandbox: {exc}[/red]")
                return 1
            owns_sandbox = True

        if run_sandbox is not None:
            eval_repo = run_sandbox.checkout
            eval_target = run_sandbox.checkout / relpath
            exec_fn = run_sandbox.run
            repro_eval = (
                BugReproduction(
                    script=run_sandbox.import_file(repro.script),
                    issue=bug_issue,
                    exec_fn=exec_fn,
                )
                if repro
                else None
            )
        else:
            eval_repo = Path(repo)
            eval_target = target
            exec_fn = subprocess.run
            repro_eval = repro

        evaluator = CodeFitnessEvaluator(
            repo=eval_repo,
            target=eval_target,
            repro=repro_eval,
            benchmarks=benchmarks,
            python=python,
            pytest_subset=pytest_subset,
            strict=strict_gates,
            repro_runs=repro_runs,
            # The per-test runner gates identically to the shared one - exit
            # status decides - and additionally keeps each test's outcome, so
            # the candidate suite can be paired against the baseline suite.
            # Every gate that executes candidate code runs through exec_fn,
            # which is the sandbox when there is one.
            pytest_runner=PerTestPytestRunner(exec_fn=exec_fn),
            benchmark_runner=functools.partial(run_benchmark_gate, exec_fn=exec_fn),
        )

        # ── 5. Baseline snapshot ────────────────────────────────────────
        _step("Baseline")
        baseline = evaluator.snapshot_baseline(baseline_source)
        console.print(
            f"  {_gate_icon(baseline.pytest_result.status)} pytest: "
            f"{baseline.pytest_result.message}"
        )
        if baseline.test_outcomes:
            console.print(
                f"  [green]✓[/green] per-test outcomes: "
                f"{len(baseline.test_outcomes)} test(s) recorded for pairing"
            )
        for bench in baseline.benchmark_results:
            console.print(f"  {_gate_icon(bench.status)} {bench.name}: {bench.message}")
        if baseline.repro:
            icon = "[green]✓[/green]" if baseline.bug_reproduces else "[yellow]![/yellow]"
            console.print(f"  {icon} repro: {baseline.repro.message}")
            if baseline.repro_trials and baseline.repro_trials.n > 1:
                console.print(f"    [dim]{baseline.repro_trials.describe()}[/dim]")

        if baseline.pytest_result.status is GateStatus.FAILED:
            console.print(
                "\n[red]✗ The baseline test suite is already failing. "
                "A red baseline cannot gate anything - fix it first.[/red]"
            )
            return 1
        if baseline.pytest_result.status is GateStatus.UNAVAILABLE:
            message = (
                "pytest could not run against this repo, so the hard gate is "
                "not actually gating."
            )
            if strict_gates:
                console.print(f"\n[red]✗ {message} Refusing under --strict-gates.[/red]")
                return 1
            console.print(f"[yellow]⚠ {message}[/yellow]")

        if repro is not None and baseline.repro is not None:
            if baseline.repro.status is ReproStatus.FIXED:
                message = (
                    "the reproduction script already passes at baseline, so it "
                    "does not reproduce the bug"
                )
                if strict_gates:
                    console.print(f"\n[red]✗ {message}. Refusing under --strict-gates.[/red]")
                    return 1
                console.print(f"[yellow]⚠ {message} - bug fitness will be meaningless[/yellow]")
            elif baseline.repro.status is ReproStatus.UNAVAILABLE:
                console.print(f"[yellow]⚠ {baseline.repro.message}[/yellow]")

        # ── 6. Ask the evolver ──────────────────────────────────────────
        _step("Mutation")
        # Everything this pipeline spends on model calls happens between here
        # and the end of the candidate loop, so that is what gets measured.
        # The evolver's own spend is not in it and cannot be: it is a separate
        # process, and EVOLVER_COST_NOTE says so wherever the total is shown.
        with UsageTracker() as usage:
            objective = build_objective(
                tool_module=relpath,
                bug_issue=f"issue {bug_issue}" if bug_issue else None,
                reproduction=repro_source,
            )
            console.print(f"  Objective: {objective}")

            job = EvolverJob(
                target_path=relpath,
                source=baseline_source,
                objective=objective,
                iterations=iterations,
                bug_issue=str(bug_issue) if bug_issue else None,
                reproduction=repro_source,
                # The copy the sandbox can actually read, when there is one.
                reproduction_path=str(repro_eval.script) if repro_eval else None,
            )

            engine: ProposesCandidates = evolver or ExternalEvolver(
                cmd=evolver_argv or [],
                repo=Path(repo),
                workdir=out_dir,
                sandbox=run_sandbox if run_sandbox is not None else UNSANDBOXED,
            )

            try:
                candidates = engine.propose(job)
            except EvolverNotInstalled as exc:
                console.print(f"[red]✗ {exc}[/red]")
                return 2
            except EvolverError as exc:
                console.print(f"[red]✗ {exc}[/red]")
                return 3

            console.print(f"  Received {len(candidates)} candidate(s)")

            # ── 7. Guardrails, then fitness, one candidate at a time ────
            _step("Evaluation")
            for candidate in candidates:
                console.print(f"\n  [bold]{candidate.label}[/bold] {candidate.notes}".rstrip())
                mutation = organism.mutate(
                    candidate.source,
                    label=candidate.label,
                    message=(
                        f"evolve({Path(relpath).stem}): candidate {candidate.label}"
                        + (f" for issue {bug_issue}" if bug_issue else "")
                    ),
                )
                if mutation.is_empty:
                    console.print("    [dim]no textual change[/dim]")

                # The organism's commit is the lineage record; the sandbox
                # checkout is where the candidate actually executes.
                if run_sandbox is not None:
                    run_sandbox.write_target(relpath, candidate.source)

                fitness = evaluator.evaluate(
                    baseline_source, candidate.source, label=candidate.label
                )
                for line in fitness.safety.summary().splitlines():
                    console.print(f"    {line}")
                if fitness.pytest_result.status is not GateStatus.SKIPPED:
                    console.print(
                        f"    {_gate_icon(fitness.pytest_result.status)} pytest: "
                        f"{fitness.pytest_result.message}"
                    )
                shown = ""
                if fitness.suite:
                    shown = f"suite vs baseline: {fitness.suite.describe()}"
                    console.print(f"    suite: {fitness.suite.describe()}")
                if fitness.repro:
                    console.print(f"    repro: {fitness.repro.message}")
                # Everything else the evaluator wanted on the record, including
                # the power notes. A caveat kept out of the console is a caveat
                # nobody reads.
                for note in fitness.notes:
                    if note != shown:
                        console.print(f"    [dim]{note}[/dim]")
                if fitness.accepted:
                    console.print(f"    [green]accepted[/green] score {fitness.score_line()}")
                else:
                    console.print(f"    [red]rejected[/red] {fitness.rejection_reason}")

                outcomes.append(CandidateOutcome(candidate, fitness, mutation))
                # Candidates are alternatives generated from the same baseline,
                # not a sequence. Rewind so the next one is scored against the
                # baseline too - in the checkout as well as on the branch.
                organism.revert_last()
                if run_sandbox is not None:
                    run_sandbox.write_target(relpath, baseline_source)

        cost = usage.report

        # ── 8. Pick a winner and re-apply it ────────────────────────────
        accepted = [o for o in outcomes if o.fitness.accepted]
        # Same winner max() picks: rank_candidates sorts stably, so an exact
        # tie keeps the order the candidates arrived in. What it adds is the
        # margin and whether that margin is worth anything.
        ranking: Optional[CandidateRanking] = rank_candidates(
            [o.fitness for o in accepted]
        )
        if accepted:
            winner = max(accepted, key=lambda o: o.fitness.total)
            final = organism.reapply(
                winner.mutation,
                label=winner.candidate.label,
            ) if winner.mutation else None
            winner_diff = organism.diff_from_baseline()
        else:
            final = None
            winner_diff = ""

        elapsed = time.time() - started

        # ── 9. Report ───────────────────────────────────────────────────
        _step("Results")
        table = Table(title=f"Code evolution: {relpath}")
        table.add_column("Candidate", style="bold")
        table.add_column("Safety")
        table.add_column("pytest")
        table.add_column("Bug")
        table.add_column("Quality", justify="right")
        table.add_column("Score", justify="right")
        # The score alone is not a result. This column is what was actually
        # measured to produce it, and it sits beside the score on purpose.
        table.add_column("Evidence", justify="right")
        table.add_column("Verdict")

        for outcome in outcomes:
            fitness = outcome.fitness
            safety_cell = (
                "[green]✓[/green]"
                if fitness.safety.passed
                else f"[red]✗ {len(fitness.safety.violations)}[/red]"
            )
            bug_cell = "-"
            if fitness.repro:
                bug_cell = (
                    "[green]fixed[/green]"
                    if fitness.repro.fixed
                    else f"[red]{fitness.repro.status.value}[/red]"
                )
            trials = fitness.repro_trials
            if trials is not None and trials.n > 1 and trials.measured_runs:
                colour = "green" if trials.fixed else "red"
                bug_cell = f"[{colour}]{trials.fixes}/{trials.measured_runs}[/{colour}]"
            table.add_row(
                outcome.candidate.label,
                safety_cell,
                _gate_icon(fitness.pytest_result.status),
                bug_cell,
                f"{fitness.quality.score:.2f}",
                f"{fitness.total:.3f}",
                f"{fitness.evidence_coverage:.0%}" if fitness.accepted else "-",
                "[green]accepted[/green]" if fitness.accepted else "[red]rejected[/red]",
            )

        console.print()
        console.print(table)

        if ranking is not None:
            icon = "[green]✓[/green]" if ranking.separated else "[yellow]![/yellow]"
            console.print(f"\n  {icon} {ranking.describe()}")
            if ranking.winner_coverage < 1.0:
                console.print(
                    f"    [dim]the winning score was measured on "
                    f"{ranking.winner_coverage:.0%} of the intended weight[/dim]"
                )

        console.print(f"\n  Cost: {cost.describe()}")
        console.print(f"  [dim]{EVOLVER_COST_NOTE}[/dim]")

        # ── 10. Emit the branch, the diff and the PR body ────────────────
        (out_dir / "baseline.py").write_text(baseline_source, encoding="utf-8")
        for outcome in outcomes:
            (out_dir / f"{outcome.candidate.label}.py").write_text(
                outcome.candidate.source, encoding="utf-8"
            )

        # No winner, or a winner that changed nothing, means there is nothing
        # to deploy - and nothing to deploy means no PR body. A run that
        # produced no diff should not leave a document implying it did.
        deployable = bool(winner and winner_diff.strip())
        pr_plan: Optional[PullRequestPlan] = None
        pr_body_path: Optional[Path] = None

        if deployable:
            (out_dir / "winner.py").write_text(winner.candidate.source, encoding="utf-8")
            (out_dir / "winner.diff").write_text(winner_diff, encoding="utf-8")
            if write_pr or push or open_pr:
                pr_plan = build_code_pull_request(
                    repo=Path(repo),
                    branch=organism.branch or "",
                    target=relpath,
                    baseline=baseline,
                    winner=winner,
                    diff=winner_diff,
                    outcomes=outcomes,
                    ranking=ranking,
                    cost=cost,
                    iterations=iterations,
                    bug_issue=str(bug_issue) if bug_issue else None,
                    repro_script=str(repro.script) if repro else None,
                    repro_runs=repro_runs,
                    baseline_sha=organism.baseline_sha or "",
                    winner_sha=final.sha if final else "",
                )
            if pr_plan is not None and write_pr:
                pr_body_path = pr_plan.write_body(out_dir)

        metrics = {
            "tool": tool,
            "target": relpath,
            "repo": str(repo),
            "branch": organism.branch,
            "baseline_sha": organism.baseline_sha,
            "bug_issue": bug_issue,
            "repro_script": str(repro.script) if repro else None,
            "repro_runs": repro_runs,
            "iterations": iterations,
            "strict_gates": strict_gates,
            "benchmarks": list(benchmarks),
            "objective": objective,
            "elapsed_seconds": round(elapsed, 2),
            "baseline": baseline.to_dict(),
            "candidates": [o.to_dict() for o in outcomes],
            "ranking": ranking.to_dict() if ranking else None,
            "winner": winner.candidate.label if winner else None,
            "winner_sha": final.sha if final else None,
            "cost": cost.to_dict(),
            "cost_excludes": EVOLVER_COST_NOTE,
            "pull_request": pr_plan.to_dict() if pr_plan else None,
            # What confined the untrusted execution, or the explicit record
            # that nothing did - a run artifact should say which it was.
            "sandbox": (
                run_sandbox.describe()
                if run_sandbox is not None
                else {"enforcer": None, "unsandboxed": True}
            ),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if deployable:
            extra_lines = ""
            if ranking is not None and ranking.margin is not None:
                extra_lines += f"Margin:  {ranking.margin:+.3f} over {ranking.runner_up}"
                if ranking.within_noise:
                    extra_lines += " - within noise, this pick is arbitrary"
                extra_lines += "\n"
            trials = winner.fitness.repro_trials
            if trials is not None and trials.measured_runs:
                extra_lines += f"Repro:   {trials.describe()}\n"
            if winner.fitness.suite is not None:
                extra_lines += f"Suite:   {winner.fitness.suite.describe()}\n"
            if pr_body_path is not None:
                extra_lines += f"PR body: {pr_body_path}\n"
            else:
                extra_lines += "PR body: not written (--no-write-pr)\n"
            console.print(
                Panel(
                    f"Branch:  [bold]{organism.branch}[/bold]\n"
                    f"Commit:  {final.short_sha if final else '-'}\n"
                    f"Diff:    {out_dir / 'winner.diff'}\n"
                    f"Score:   {winner.fitness.score_line()}\n"
                    f"{extra_lines}\n"
                    "Nothing was merged. PLAN.md requires human review of every "
                    "line of evolved code:\n"
                    f"  git diff {organism.baseline_sha[:8] if organism.baseline_sha else 'HEAD'} "
                    f"{organism.branch} -- {relpath}",
                    title="✓ Candidate ready for review",
                    border_style="green",
                )
            )

            # ── 11. Only now, and only if asked, leave this machine ─────
            if pr_plan is not None and push:
                console.print(f"\n  Pushing {pr_plan.branch} to {remote} (--push)")
                try:
                    pr_plan.push(remote)
                    console.print(f"  [green]✓[/green] pushed to {remote}")
                except DeploymentGitError as exc:
                    console.print(f"[red]✗ push failed: {exc}[/red]")
                    exit_code = 4
            if pr_plan is not None and open_pr:
                if not push:
                    console.print(
                        "  [dim]--open-pr without --push: gh needs the branch on "
                        "the remote, so this works only if it is already there.[/dim]"
                    )
                console.print(f"  Opening a PR against {base} (--open-pr)")
                try:
                    output = pr_plan.open(base).strip()
                    console.print(f"  [green]✓[/green] {output or 'PR opened'}")
                except DeploymentGitError as exc:
                    console.print(f"[red]✗ could not open the PR: {exc}[/red]")
                    exit_code = 4
            if not push and not open_pr:
                console.print(
                    "\n[dim]  Nothing was pushed and no PR was opened. Pass --push "
                    "and --open-pr to do either; neither happens on its own.[/dim]"
                )
        elif winner:
            console.print(
                "\n[yellow]⚠ The winning candidate is identical to the baseline - "
                "nothing to review.[/yellow]"
            )
        else:
            console.print(
                "\n[yellow]⚠ No candidate survived the guardrails. "
                "Nothing to review, nothing changed.[/yellow]"
            )
            for outcome in outcomes:
                console.print(
                    f"    {outcome.candidate.label}: {outcome.fitness.rejection_reason}"
                )

        console.print(f"\n  Run artifacts: {out_dir}/")
        console.print(f"  Elapsed: {elapsed:.1f}s")
    finally:
        # The workspace holds a clone plus whatever the sandboxed processes
        # wrote; a crashed or timed-out run cleans up the same way a good one
        # does. An injected sandbox is the caller's to clean.
        if owns_sandbox and run_sandbox is not None:
            run_sandbox.cleanup()
        # Restoring the operator's branch matters more than anything above it,
        # so a failure here is reported rather than raised over the real result.
        original = organism.original_ref
        try:
            organism.close()
            console.print(f"  Restored branch: {original}")
        except OrganismError as exc:
            console.print(
                f"[red]✗ Could not restore branch {original}: {exc}[/red]\n"
                f"[red]  You are still on {organism.branch}.[/red]"
            )

    return exit_code


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--tool", required=True, help="Tool module to evolve (e.g. file_tools)")
@click.option("--bug-issue", default=None, help="GitHub issue number this run targets")
@click.option("--repro-script", default=None, help="Script that reproduces the bug")
@click.option("--repro-runs", default=1, type=click.IntRange(min=1),
              help="Times to run the reproduction per candidate (1 is one Bernoulli "
                   "trial; more measures a fix rate with an interval)")
@click.option("--iterations", default=10, help="Iterations to request from the evolver")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--evolver-cmd", default=None, help="Path to the Darwinian Evolver CLI")
@click.option("--benchmark", "benchmarks", multiple=True,
              help="Benchmark to gate on (repeatable, e.g. --benchmark tblite)")
@click.option("--python", "python_bin", default=None,
              help="Interpreter used for the hermes-agent test suite and repro")
@click.option("--pytest-subset", multiple=True,
              help="Narrow the pytest gate (repeatable path or -k expression)")
@click.option("--allow-dirty", is_flag=True,
              help="Evolve on top of uncommitted changes in the hermes-agent repo")
@click.option("--strict-gates", is_flag=True,
              help="Treat an unavailable gate as a failure instead of a warning")
@click.option("--dry-run", is_flag=True, help="Validate setup without mutating anything")
@click.option("--write-pr/--no-write-pr", default=True,
              help="Write PULL_REQUEST.md beside the run artifacts when there is a "
                   "winner to deploy (local file, nothing is sent)")
@click.option("--push/--no-push", default=False,
              help="Push the evolution branch to the remote. Off unless asked for")
@click.option("--open-pr/--no-open-pr", default=False,
              help="Open the pull request with gh. Off unless asked for; never merges")
@click.option("--remote", default="origin", show_default=True,
              help="Remote to push the evolution branch to")
@click.option("--pr-base", "base", default="main", show_default=True,
              help="Base branch for the pull request")
@click.option("--unsandboxed", is_flag=True,
              help="Run the evolver and candidate code with no sandbox: the "
                   "operator's environment, filesystem and network. The old "
                   "behaviour, available only by asking for it")
@click.option("--sandbox-env", "sandbox_env", multiple=True, metavar="NAME",
              help="Environment variable to pass into the sandbox by name "
                   "(repeatable). Nothing else from the parent environment "
                   "crosses in")
@click.option("--sandbox-allow-network", "allow_network", is_flag=True,
              help="Allow network access inside the sandbox. Off by default: "
                   "an evolver that can read its workspace and reach the "
                   "network can exfiltrate what it was shown")
def main(
    tool,
    bug_issue,
    repro_script,
    repro_runs,
    iterations,
    hermes_repo,
    evolver_cmd,
    benchmarks,
    python_bin,
    pytest_subset,
    allow_dirty,
    strict_gates,
    dry_run,
    write_pr,
    push,
    open_pr,
    remote,
    base,
    unsandboxed,
    sandbox_env,
    allow_network,
):
    """Evolve hermes-agent tool code with Darwinian Evolver, under guardrails."""
    code = evolve_tool_code(
        tool=tool,
        bug_issue=bug_issue,
        repro_script=repro_script,
        iterations=iterations,
        hermes_repo=hermes_repo,
        evolver_cmd=evolver_cmd,
        strict_gates=strict_gates,
        dry_run=dry_run,
        benchmarks=tuple(benchmarks),
        python=python_bin,
        pytest_subset=tuple(pytest_subset) or None,
        allow_dirty=allow_dirty,
        repro_runs=repro_runs,
        write_pr=write_pr,
        push=push,
        open_pr=open_pr,
        remote=remote,
        base=base,
        sandbox=UNSANDBOXED if unsandboxed else None,
        sandbox_env=tuple(sandbox_env),
        allow_network=allow_network,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
