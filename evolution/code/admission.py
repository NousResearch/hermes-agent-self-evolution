"""The admission gate for evolved code.

Phase 4 mutates executable code, which makes it categorically riskier than the
earlier phases. A bad skill edit produces a worse answer; a bad code edit ships
a defect into every agent that loads the tool. So a candidate is not scored on
quality until it has earned the right to be scored at all.

The gate has three properties that a plain "mutate and run pytest" loop lacks.

**It runs nowhere near the real repository.** Checks execute against a
materialised copy of the candidate. The working tree the operator is sitting in
is never the thing under test.

This is a *safety* boundary, not a *security* one, and the distinction matters
because the code being run was written by a model. Credentials and proxy
variables are stripped from the environment, but the process still has the
operator's filesystem access and working network. It protects the checkout from
an accidental bad edit; it does not contain deliberately hostile code. Run
Phase 4 somewhere you would be willing to run an untrusted pull request.

**Ground truth outranks opinion.** Alongside the test suite, the gate replays
commands Hermes actually ran and recorded in ``verification_evidence.db``, with
their real exit codes. A candidate that breaks something a human verified for
real fails, regardless of what the suite says.

**Some checks are sealed.** Visible check failures are returned to the mutator
as feedback, because that is how it improves. Hidden checks are held out: they
gate admission but their details never reach the mutator. Without that split an
optimizer with enough iterations learns to satisfy the specific checks it can
see, and the gate stops measuring anything. This mirrors the train/holdout
split the earlier phases already use for prompts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

# A single check that hangs must not hang the sweep.
DEFAULT_CHECK_TIMEOUT_S = 600

# Environment that must not leak into a sandboxed check run. Credentials in
# particular: a mutated tool executing with live API keys is the failure mode
# this whole gate exists to prevent.
_STRIPPED_ENV_PREFIXES = (
    "ANTHROPIC_", "OPENAI_", "OPENROUTER_", "AWS_", "GITHUB_", "GH_",
    "SLACK_", "NOTION_", "HERMES_TOKEN", "DATABASE_URL",
)


@dataclass
class CheckResult:
    """The outcome of one check against one candidate."""

    name: str
    passed: bool
    detail: str = ""
    duration_s: float = 0.0
    exit_code: Optional[int] = None

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        suffix = f" — {self.detail}" if self.detail else ""
        return f"{mark} {self.name} ({self.duration_s:.1f}s){suffix}"


@dataclass
class AdmissionVerdict:
    """Whether a candidate may be scored, and what the mutator is told."""

    admitted: bool
    visible: list[CheckResult] = field(default_factory=list)
    hidden: list[CheckResult] = field(default_factory=list)
    reason: str = ""

    @property
    def visible_failures(self) -> list[CheckResult]:
        return [c for c in self.visible if not c.passed]

    @property
    def hidden_failures(self) -> list[CheckResult]:
        return [c for c in self.hidden if not c.passed]

    def feedback_for_mutator(self) -> str:
        """What the mutator is allowed to learn from this attempt.

        Visible failures come back in full. Hidden failures are reported only
        as a count, with no name, command or output — enough for the mutator to
        know it was rejected, not enough for it to target the held-out checks.
        """
        lines: list[str] = []
        for check in self.visible_failures:
            lines.append(f"FAILED {check.name}: {check.detail[:500]}")

        hidden_failed = len(self.hidden_failures)
        if hidden_failed:
            lines.append(
                f"Rejected by {hidden_failed} held-out check(s). Their contents "
                "are deliberately withheld: fix the change properly rather than "
                "targeting specific checks."
            )
        return "\n".join(lines) or "All checks passed."

    def summary(self) -> str:
        v = f"{sum(c.passed for c in self.visible)}/{len(self.visible)} visible"
        h = f"{sum(c.passed for c in self.hidden)}/{len(self.hidden)} hidden"
        return f"{'ADMITTED' if self.admitted else 'REJECTED'} · {v} · {h}"


class CodeCheck(Protocol):
    """Something that can be run against a candidate repository."""

    name: str

    def run(self, repo: Path) -> CheckResult: ...


def _sandbox_env() -> dict:
    """A stripped environment for running untrusted candidate code."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(k.startswith(p) for p in _STRIPPED_ENV_PREFIXES)
    }
    # Keep runs hermetic and offline-ish; a mutated tool should not be able to
    # quietly reach the network through a proxy the operator happens to have set.
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(var, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["EVOLUTION_SANDBOX"] = "1"
    return env


def _run_command(
    name: str,
    argv: Sequence[str],
    repo: Path,
    timeout_s: int,
    expect_exit: int = 0,
) -> CheckResult:
    started = time.time()
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_sandbox_env(),
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"timed out after {timeout_s}s",
            duration_s=time.time() - started,
        )
    except (OSError, ValueError) as exc:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"could not run: {exc}",
            duration_s=time.time() - started,
        )

    tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-6:]
    return CheckResult(
        name=name,
        passed=proc.returncode == expect_exit,
        detail="" if proc.returncode == expect_exit else "\n".join(tail),
        duration_s=time.time() - started,
        exit_code=proc.returncode,
    )


@dataclass
class CommandCheck:
    """Any command that must exit zero — build, lint, typecheck."""

    name: str
    argv: Sequence[str]
    timeout_s: int = DEFAULT_CHECK_TIMEOUT_S
    expect_exit: int = 0

    def run(self, repo: Path) -> CheckResult:
        return _run_command(self.name, self.argv, repo, self.timeout_s, self.expect_exit)


@dataclass
class PytestCheck:
    """The test suite, optionally narrowed to specific paths.

    The full-suite variant is mandatory before any deploy: a candidate that
    passes its own targeted tests while breaking something three modules away
    is the exact failure this catches.
    """

    name: str = "pytest"
    paths: Sequence[str] = ("tests/",)
    timeout_s: int = DEFAULT_CHECK_TIMEOUT_S
    extra_args: Sequence[str] = ("-q", "--tb=no")

    def run(self, repo: Path) -> CheckResult:
        argv = ["python", "-m", "pytest", *self.paths, *self.extra_args]
        result = _run_command(self.name, argv, repo, self.timeout_s)
        # pytest exit 5 means "no tests collected", which is not a pass — a
        # candidate that deletes the tests it was failing must not be admitted.
        if result.exit_code == 5:
            return CheckResult(
                name=self.name,
                passed=False,
                detail="no tests were collected",
                duration_s=result.duration_s,
                exit_code=5,
            )
        return result


@dataclass
class RecordedCommandCheck:
    """Replay a command Hermes actually ran, and require its recorded outcome.

    These come from ``verification_evidence.db`` — real test, build and lint
    invocations with real exit codes. They are the strongest signal available
    because nobody wrote them to be a benchmark; they are just what happened.
    """

    name: str
    command: str
    expected_exit: int = 0
    timeout_s: int = DEFAULT_CHECK_TIMEOUT_S

    def run(self, repo: Path) -> CheckResult:
        started = time.time()
        try:
            proc = subprocess.run(
                self.command,
                shell=True,
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=_sandbox_env(),
            )
        except subprocess.TimeoutExpired:
            return CheckResult(self.name, False, f"timed out after {self.timeout_s}s",
                               time.time() - started)
        except (OSError, ValueError) as exc:
            return CheckResult(self.name, False, f"could not run: {exc}",
                               time.time() - started)

        passed = proc.returncode == self.expected_exit
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-4:]
        return CheckResult(
            name=self.name,
            passed=passed,
            detail="" if passed else
                   f"exit {proc.returncode}, recorded {self.expected_exit}: " + "\n".join(tail),
            duration_s=time.time() - started,
            exit_code=proc.returncode,
        )


class AdmissionGate:
    """Decides whether a candidate is allowed to be scored.

    Visible checks run first and cheapest-first; hidden checks run only if the
    visible ones pass, so the expensive held-out suite is not spent on
    candidates that were never going to make it.
    """

    def __init__(
        self,
        visible: Sequence[CodeCheck],
        hidden: Sequence[CodeCheck] = (),
        require_all_visible: bool = True,
    ):
        self.visible = list(visible)
        self.hidden = list(hidden)
        self.require_all_visible = require_all_visible

    def admit(self, repo: Path) -> AdmissionVerdict:
        repo = Path(repo)
        if not repo.is_dir():
            return AdmissionVerdict(
                admitted=False, reason=f"candidate repo does not exist: {repo}"
            )

        visible_results: list[CheckResult] = []
        for check in self.visible:
            result = check.run(repo)
            visible_results.append(result)
            if self.require_all_visible and not result.passed:
                # Short-circuit: no point paying for the rest, and the mutator
                # gets a focused failure rather than a wall of them.
                return AdmissionVerdict(
                    admitted=False,
                    visible=visible_results,
                    reason=f"visible check failed: {result.name}",
                )

        hidden_results = [check.run(repo) for check in self.hidden]
        hidden_ok = all(r.passed for r in hidden_results)

        return AdmissionVerdict(
            admitted=hidden_ok,
            visible=visible_results,
            hidden=hidden_results,
            reason="" if hidden_ok else
                   f"{sum(not r.passed for r in hidden_results)} held-out check(s) failed",
        )


def materialize_candidate(
    baseline_repo: Path,
    file_contents: dict[str, str],
    dest: Path,
) -> Path:
    """Copy the baseline repo and overlay the candidate's files.

    Returns the sandbox path. The baseline is copied rather than mutated in
    place, so an aborted or crashing check can never leave the operator's real
    checkout in a modified state.
    """
    baseline_repo = Path(baseline_repo)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    shutil.copytree(
        baseline_repo,
        dest,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".venv", "node_modules",
            ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ),
    )

    _write_candidate_files(dest, file_contents)
    return dest


def _write_candidate_files(dest: Path, file_contents: dict[str, str]) -> None:
    """Write a candidate's files into an existing sandbox, refusing escapes."""
    for rel, content in file_contents.items():
        target = dest / rel
        # Refuse to write outside the sandbox — a candidate's file map is
        # attacker-adjacent input once a model is generating it.
        try:
            target.resolve().relative_to(dest.resolve())
        except ValueError as exc:
            raise ValueError(f"candidate path escapes the sandbox: {rel}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


class CandidateSandbox:
    """One materialised repo, reused across candidates in a run.

    Copying the baseline per candidate is correct but expensive: hermes-agent
    is 164 MB across ~9,000 files, so a default run spent gigabytes of I/O
    re-copying an identical tree twenty times.

    Hardlinking the copy would be the obvious speed-up and is *unsafe here*:
    the checks execute the candidate's code, and anything that writes through
    a hardlink modifies the operator's real file. So the tree is copied once,
    for real, and only the candidate's own files are swapped between runs.

    Only the target files are restored between candidates. A check that leaves
    other artifacts behind (a log, a cache) will leak into the next candidate;
    that is the trade being made for not re-copying the tree, and it is why
    ``reset()`` exists for callers who need a pristine start.
    """

    def __init__(self, baseline_repo: Path, root: Path):
        self.baseline_repo = Path(baseline_repo)
        self.root = Path(root)
        self._applied: set[str] = set()

    def __enter__(self) -> "CandidateSandbox":
        materialize_candidate(self.baseline_repo, {}, self.root)
        return self

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def apply(self, file_contents: dict[str, str]) -> Path:
        """Put one candidate in the sandbox and return the repo path."""
        # Restore anything a previous candidate overwrote, so a file that this
        # candidate does not touch is the baseline's version and not the last
        # candidate's.
        for rel in self._applied - set(file_contents):
            source = self.baseline_repo / rel
            target = self.root / rel
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            elif target.exists():
                target.unlink()

        _write_candidate_files(self.root, file_contents)
        self._applied = set(file_contents)
        return self.root

    def reset(self) -> None:
        """Rebuild the tree from the baseline, discarding any check artifacts."""
        shutil.rmtree(self.root, ignore_errors=True)
        materialize_candidate(self.baseline_repo, {}, self.root)
        self._applied = set()


def build_default_gate(
    hermes_repo: Path,
    targeted_tests: Sequence[str] = (),
    recorded: Sequence[RecordedCommandCheck] = (),
) -> AdmissionGate:
    """The gate Phase 4 uses unless a caller supplies its own.

    Visible: a fast targeted test run the mutator can iterate against.
    Hidden: the full suite plus replayed real commands — the parts that must
    not become optimization targets.
    """
    visible: list[CodeCheck] = []
    if targeted_tests:
        visible.append(
            PytestCheck(name="targeted-tests", paths=tuple(targeted_tests), timeout_s=300)
        )

    hidden: list[CodeCheck] = [
        PytestCheck(name="full-suite", paths=("tests/",), timeout_s=900),
    ]
    hidden.extend(recorded)

    return AdmissionGate(visible=visible, hidden=hidden)
