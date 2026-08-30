"""Turn a finished run into a reviewable pull request.

PLAN.md constraint 5 is "Deployment via PR (Never Direct Commit)": every
evolved change reaches hermes-agent as a branch and a PR whose body carries the
before/after scores on train, validation and holdout, the full diff, the cost of
the run, and any constraint violations caught and rejected along the way. Until
now the pipeline stopped one step short of that - it wrote evolved text and a
metrics file into an output directory and left the reviewer to assemble the rest
by hand. ``EvolutionConfig.create_pr`` has defaulted to ``True`` the whole time
with nothing reading it.

**Nothing here reaches the network on its own.** Building a branch and writing a
PR body are local operations and happen by default; pushing that branch and
opening the PR are separate, explicitly requested steps. An optimization run
that phoned out to GitHub because a config field defaulted to True would be a
bad surprise, and "never direct commit" is a rule about review, not a licence to
publish automatically. :meth:`PullRequestPlan.push` and
:meth:`PullRequestPlan.open` exist for a caller that has been told to, and both
refuse rather than guess when the remote or the CLI is missing.

The branch name follows PLAN.md's ``evolve/<target>-<timestamp>``. The timestamp
is passed in rather than read from the clock so a run is reproducible and a test
can assert on the name.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from evolution.core.cost import CostReport

__all__ = [
    "ScoreLine",
    "dirty_paths",
    "require_clean_worktree",
    "RejectedCandidate",
    "PullRequestPlan",
    "build_pull_request",
    "GitError",
]


class GitError(RuntimeError):
    """A git or gh invocation failed, with its stderr attached."""


def _run(args: Sequence[str], cwd: Path) -> str:
    try:
        proc = subprocess.run(
            list(args), cwd=str(cwd), capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError as exc:
        raise GitError(f"{args[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"{' '.join(args)} timed out") from exc
    if proc.returncode != 0:
        raise GitError(f"{' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


@dataclass(frozen=True)
class ScoreLine:
    """One split's before and after, for the PR body's headline table."""

    split: str
    baseline: float
    evolved: float
    detail: str = ""

    @property
    def delta(self) -> float:
        """Change from baseline to evolved."""
        return self.evolved - self.baseline

    def row(self) -> str:
        """This split rendered as one Markdown table row."""
        # Always five cells, matching the five-column header: an empty detail
        # is an empty cell, not a row that stops one pipe short of the table.
        return (
            f"| {self.split} | {self.baseline:.3f} | {self.evolved:.3f} "
            f"| {self.delta:+.3f} | {self.detail} |"
        )


@dataclass(frozen=True)
class RejectedCandidate:
    """A variant the run threw away, and why.

    PLAN.md asks for these in the body on purpose. A PR that shows only the
    winner hides how hard the gates were working, and a reviewer who cannot see
    what was rejected cannot tell a careful run from a lucky one.
    """

    label: str
    reason: str


@dataclass
class PullRequestPlan:
    """A branch, a commit message, and a PR body, all on disk and nothing sent."""

    repo: Path
    branch: str
    title: str
    body: str
    commit_message: str
    files: tuple[str, ...] = ()
    created_branch: bool = False
    original_ref: str = ""
    body_path: Optional[Path] = None

    def write_body(self, output_dir: Path) -> Path:
        """Save the PR body next to the run's other artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "PULL_REQUEST.md"
        path.write_text(self.body, encoding="utf-8")
        self.body_path = path
        return path

    def push(self, remote: str = "origin") -> str:
        """Push the branch. Only call this when the operator asked for it."""
        return _run(["git", "push", "-u", remote, self.branch], self.repo)

    def open(self, base: str = "main") -> str:
        """Open the PR with ``gh``. Only call this when the operator asked for it."""
        if shutil.which("gh") is None:
            raise GitError(
                "gh is not installed, so the PR cannot be opened from here. "
                "The branch and PULL_REQUEST.md are ready to use by hand."
            )
        # The body embeds the run's diff, and a diff-sized string does not
        # belong in argv: past the OS argument limit the failure is an opaque
        # OSError, not a message. gh reads it from a file instead - the one
        # write_body() already saved, or a temporary one cleaned up after.
        body_path = self.body_path if self.body_path and self.body_path.is_file() else None
        scratch: Optional[Path] = None
        if body_path is None:
            fd, raw = tempfile.mkstemp(prefix="hase-pr-body-", suffix=".md")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(self.body)
            scratch = body_path = Path(raw)
        try:
            return _run(
                [
                    "gh", "pr", "create",
                    "--base", base,
                    "--head", self.branch,
                    "--title", self.title,
                    "--body-file", str(body_path),
                ],
                self.repo,
            )
        finally:
            if scratch is not None:
                scratch.unlink(missing_ok=True)

    def restore(self) -> None:
        """Return the checkout to the ref it was on before the branch was made."""
        if self.created_branch and self.original_ref:
            _run(["git", "checkout", self.original_ref], self.repo)

    def discard(self) -> None:
        """Restore the original ref and delete the branch entirely.

        For a caller that built a branch, failed to push it, and does not want
        to leave a dead ``evolve/`` ref behind in the operator's checkout.
        """
        if not self.created_branch:
            return
        _abandon_branch(self.repo, self.branch, self.original_ref)
        self.created_branch = False

    def to_dict(self) -> dict:
        """Serialise the plan, without the body text itself."""
        return {
            "branch": self.branch,
            "title": self.title,
            "files": list(self.files),
            "created_branch": self.created_branch,
            "original_ref": self.original_ref,
            "body_path": str(self.body_path) if self.body_path else None,
        }


def dirty_paths(repo: Path, files: Sequence[str]) -> list[str]:
    """Which of *files* have uncommitted modifications right now.

    Only the paths a run is about to touch matter. Unrelated work elsewhere in
    the checkout survives a branch switch untouched, so refusing over it would
    be noise; work in a file the run is about to overwrite and commit would not.
    """
    if not files:
        return []
    try:
        out = _run(["git", "status", "--porcelain", "--", *files], repo)
    except GitError:
        # Callers that only want to describe the tree can live with "nothing
        # known". require_clean_worktree cannot, and asks the question itself
        # rather than reading this empty list as an all-clear.
        return []
    dirty: list[str] = []
    for line in out.splitlines():
        if len(line) > 3 and not line.startswith("??"):
            dirty.append(line[3:].strip())
    return dirty


def require_clean_worktree(
    repo: Path, files: Sequence[str], allow_dirty: bool = False
) -> None:
    """Refuse to start when the operator has uncommitted work in *files*.

    Call this **before** a phase writes anything, which is the only moment the
    distinction is still visible. Afterwards the run's own edits make the same
    files dirty and there is no way to tell whose changes they are.

    It matters because the deployment step is destructive without it:
    ``git checkout -b`` carries uncommitted edits onto the new branch, the
    commit absorbs them, and restoring the original ref leaves the operator's
    work stranded on a branch they did not create - from the working tree it
    simply looks deleted. Phase 4 has always refused this through
    :class:`~evolution.code.organism.CodeOrganism`; this is the same rule for
    the phases that rewrite text.
    """
    if allow_dirty:
        return
    if not files:
        return
    if not (repo / ".git").exists():
        # Not a checkout at all, so there is no uncommitted work to strand and
        # no branch to strand it on. build_pull_request refuses on its own
        # terms later; refusing here would fail a run whose only crime is that
        # the operator does not keep hermes-agent under git.
        return
    try:
        status = _run(["git", "status", "--porcelain", "--", *files], repo)
    except GitError as exc:
        # Failing open here is how the guard stops guarding. A missing git
        # binary, a timeout or a lock held by another process all land in this
        # branch, and every one of them leaves the worktree state unknown -
        # which is the one state where the destructive sequence below must not
        # run. Refuse instead, and say what to do about it.
        raise GitError(
            f"could not read the worktree state of {repo}: {exc}. This run "
            "would checkout a new branch, commit onto it and switch back, "
            "which strands any uncommitted work it cannot see. Fix the "
            "checkout, or pass --allow-dirty to proceed anyway."
        ) from exc

    dirty = [
        line[3:].strip()
        for line in status.splitlines()
        if len(line) > 3 and not line.startswith("??")
    ]
    if not dirty:
        return
    raise GitError(
        "uncommitted changes in "
        + ", ".join(dirty[:5])
        + (" and others" if len(dirty) > 5 else "")
        + ". This run would overwrite them and commit the result onto an evolve "
        "branch, leaving your work off your current branch. Commit or stash "
        "first, or pass --allow-dirty to evolve on top of it."
    )


def _abandon_branch(repo: Path, branch: str, original: str) -> None:
    """Put the checkout back and delete the half-built branch. Never raises.

    The checkout is deliberately not forced. ``git checkout --force`` discards
    local modifications across the whole tree, which would destroy exactly the
    unrelated work :func:`require_clean_worktree` promises will survive - and
    this runs on the failure path, when the operator is already having a bad
    time. A plain checkout cannot conflict here anyway: the branch is abandoned
    before any commit lands on it, so it still points at *original*.

    If the checkout does refuse, that refusal is git protecting something. Leave
    the operator on the branch with their tree intact rather than clearing the
    way; a stray ``evolve/`` ref is recoverable and their afternoon is not.
    """
    try:
        _run(["git", "checkout", original], repo)
    except GitError:
        return
    try:
        _run(["git", "branch", "-D", branch], repo)
    except GitError:
        pass


def _current_ref(repo: Path) -> str:
    """The branch name, or the commit sha when the checkout is detached."""
    ref = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo).strip()
    if ref and ref != "HEAD":
        return ref
    return _run(["git", "rev-parse", "HEAD"], repo).strip()


def render_body(
    *,
    target: str,
    phase: str,
    scores: Sequence[ScoreLine],
    diff: str,
    cost: Optional[CostReport] = None,
    rejected: Sequence[RejectedCandidate] = (),
    gates: Sequence[str] = (),
    dataset: str = "",
    optimizer: str = "",
    iterations: Optional[int] = None,
    statistics: str = "",
    notes: Sequence[str] = (),
    max_diff_lines: int = 400,
) -> str:
    """Render the PR body PLAN.md specifies, in that order."""
    lines: list[str] = []

    lines.append(f"Evolved `{target}` with {phase}.")
    lines.append("")

    if scores:
        lines.append("## Scores")
        lines.append("")
        lines.append("| Split | Before | After | Change | Notes |")
        lines.append("|---|---:|---:|---:|---|")
        lines.extend(s.row() for s in scores)
        lines.append("")

    if statistics:
        lines.append("## Evidence")
        lines.append("")
        lines.append(statistics)
        lines.append("")

    if gates:
        lines.append("## Gates")
        lines.append("")
        lines.extend(f"- {g}" for g in gates)
        lines.append("")

    if rejected:
        lines.append("## Rejected along the way")
        lines.append("")
        lines.append(
            f"{len(rejected)} candidate(s) were produced and refused before this one:"
        )
        lines.append("")
        lines.extend(f"- `{r.label}`: {r.reason}" for r in rejected)
        lines.append("")

    lines.append("## Run")
    lines.append("")
    if optimizer:
        detail = f"{optimizer}"
        if iterations is not None:
            detail += f", {iterations} iteration(s)"
        lines.append(f"- Optimizer: {detail}")
    if dataset:
        lines.append(f"- Eval dataset: {dataset}")
    lines.append(f"- Cost: {cost.describe() if cost else 'not measured'}")
    lines.extend(f"- {n}" for n in notes)
    lines.append("")

    if diff:
        diff_lines = diff.splitlines()
        clipped = len(diff_lines) > max_diff_lines
        shown = diff_lines[:max_diff_lines]
        lines.append("## Diff")
        lines.append("")
        lines.append("```diff")
        lines.extend(shown)
        lines.append("```")
        if clipped:
            lines.append("")
            lines.append(
                f"_Diff clipped at {max_diff_lines} lines of "
                f"{len(diff_lines)}; the branch has all of it._"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_pull_request(
    *,
    repo: Path,
    target: str,
    phase: str,
    timestamp: str,
    files: Sequence[str],
    scores: Sequence[ScoreLine] = (),
    cost: Optional[CostReport] = None,
    rejected: Sequence[RejectedCandidate] = (),
    gates: Sequence[str] = (),
    dataset: str = "",
    optimizer: str = "",
    iterations: Optional[int] = None,
    statistics: str = "",
    notes: Sequence[str] = (),
    commit: bool = True,
) -> PullRequestPlan:
    """Create ``evolve/<target>-<timestamp>``, commit *files*, and render the body.

    The working tree is expected to already hold the evolved content: this
    stages what it is given and does not decide what changed. Nothing is pushed
    and no PR is opened.

    Raises :class:`GitError` when *repo* is not a git repository, so a caller
    never believes a branch exists that does not.
    """
    repo = Path(repo)
    if not (repo / ".git").exists():
        raise GitError(f"{repo} is not a git repository")

    original = _current_ref(repo)
    branch = f"evolve/{target}-{timestamp}"
    _run(["git", "checkout", "-b", branch], repo)

    # From here on the checkout is on a branch the caller did not ask to be
    # left on. Anything that fails has to put them back before it propagates,
    # or a caller that only handles GitError is stranded with no plan object to
    # call restore() on.
    try:
        diff = ""
        if commit and files:
            _run(["git", "add", "--", *files], repo)
            # Scoped to the run's own paths. A bare `git diff --cached` reports
            # the whole index, so anything the operator had already staged
            # elsewhere would be quoted into the PR body as if this run had
            # produced it.
            diff = _run(["git", "diff", "--cached", "--", *files], repo)
    except Exception:
        _abandon_branch(repo, branch, original)
        raise

    headline = ""
    if scores:
        best = scores[-1]
        headline = f" - {best.split} {best.baseline:.3f} to {best.evolved:.3f}"

    title = f"evolve: {target}"
    body = render_body(
        target=target,
        phase=phase,
        scores=scores,
        diff=diff,
        cost=cost,
        rejected=rejected,
        gates=gates,
        dataset=dataset,
        optimizer=optimizer,
        iterations=iterations,
        statistics=statistics,
        notes=notes,
    )

    message_lines = [f"evolve: {target}{headline}", ""]
    if optimizer:
        detail = optimizer
        if iterations is not None:
            detail += f" ({iterations} iterations)"
        message_lines.append(f"Optimizer: {detail}")
    if dataset:
        message_lines.append(f"Eval dataset: {dataset}")
    for score in scores:
        message_lines.append(
            f"{score.split}: {score.baseline:.3f} -> {score.evolved:.3f} "
            f"({score.delta:+.3f})"
        )
    if cost is not None:
        message_lines.append(f"Cost: {cost.describe()}")
    commit_message = "\n".join(message_lines).rstrip() + "\n"

    if commit and files:
        try:
            # --no-verify, matching CodeOrganism.mutate. The two used to
            # disagree, which meant a checkout whose hooks Phase 4 ignored could
            # still fail Phases 2 and 3. Skipping them is the deliberate choice:
            # this commit lands on a scratch evolve/ branch that nothing merges
            # on its own, the repo's hooks run for real when a human commits the
            # reviewed result, and a rejecting hook here throws away a paid
            # optimization run to enforce a rule about a branch nobody keeps.
            #
            # Scoped to *files* for the same reason the diff above is. `git
            # commit` writes the whole index, so work the operator had staged
            # anywhere else in the repo would be committed onto this evolve/
            # branch, and restore() would then leave it off the branch they
            # were on - gone, from the working tree's point of view.
            # require_clean_worktree cannot catch that, because it only looks
            # at the paths this run owns.
            _run(
                ["git", "commit", "--no-verify", "-m", commit_message, "--", *files],
                repo,
            )
        except Exception:
            _abandon_branch(repo, branch, original)
            raise

    return PullRequestPlan(
        repo=repo,
        branch=branch,
        title=title,
        body=body,
        commit_message=commit_message,
        files=tuple(files),
        created_branch=True,
        original_ref=original,
    )
