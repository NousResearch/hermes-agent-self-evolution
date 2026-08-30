"""Wrap a hermes-agent source file as a git-branch-backed organism.

Phase 4 is the only tier that rewrites executable code, so it is the only tier
where a bad candidate can break the agent outright. The mitigation PLAN.md
specifies is lineage: every candidate mutation becomes a commit on a branch we
created, so "what changed" is a diff, "who proposed it" is a commit message,
and "undo" is a git reset rather than a hand-edited file.

Three properties this module guarantees, because the guardrails are the
feature here:

1. **The operator's checkout is not collateral damage.** A dirty worktree
   aborts the run unless ``allow_dirty=True`` is passed explicitly. Nothing is
   ever force-checked-out over uncommitted work.
2. **The original branch is always restored.** ``close()`` runs from
   ``__exit__``, so an exception mid-evolution still leaves the operator on the
   branch they started on, with the target file as they left it.
3. **Only the target file moves.** Commits stage exactly one path. A candidate
   that writes anywhere else is not something this class can express.

The evolution branch survives ``close()`` on purpose. PLAN.md forbids
auto-merge for code changes, so the deliverable of a run is a branch plus a
diff for a human to review.

Nothing here imports Darwinian Evolver. It is AGPL v3 and is only ever driven
as an external CLI subprocess, from evolve_tool_code.py.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

__all__ = [
    "OrganismError",
    "GitError",
    "DirtyWorktreeError",
    "Mutation",
    "CodeOrganism",
    "git_available",
    "is_git_repo",
]


class OrganismError(RuntimeError):
    """Raised when an organism cannot be created or driven safely."""


class GitError(OrganismError):
    """Raised when a git command fails."""


class DirtyWorktreeError(OrganismError):
    """Raised when the target repo has uncommitted work and opt-in was not given."""


# Identity used only when the target repo has no committer configured. Commits
# still need an author, and borrowing one from the operator's global config is
# not something to do silently.
FALLBACK_AUTHOR_NAME = "hermes-self-evolution"
FALLBACK_AUTHOR_EMAIL = "self-evolution@localhost"


def git_available() -> bool:
    """True when a git binary is on PATH."""
    return shutil.which("git") is not None


def is_git_repo(path: Path, git_binary: str = "git") -> bool:
    """True when *path* sits inside a git working tree."""
    path = Path(path)
    if not path.is_dir() or shutil.which(git_binary) is None:
        return False
    try:
        proc = subprocess.run(
            [git_binary, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=str(path),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


@dataclass(frozen=True)
class Mutation:
    """One candidate applied to the target file and committed to the branch."""

    index: int
    label: str
    message: str
    sha: str
    parent_sha: str
    diff: str
    source: str

    @property
    def short_sha(self) -> str:
        """The commit sha abbreviated for display."""
        return self.sha[:8]

    @property
    def is_empty(self) -> bool:
        """True when the candidate produced no textual change."""
        return not self.diff.strip()

    def to_dict(self) -> dict:
        """Serialise the mutation, including its diff, for the lineage record."""
        return {
            "index": self.index,
            "label": self.label,
            "message": self.message,
            "sha": self.sha,
            "short_sha": self.short_sha,
            "parent_sha": self.parent_sha,
            "diff_lines": len(self.diff.splitlines()),
            "empty": self.is_empty,
        }


class CodeOrganism:
    """One hermes-agent source file, evolved on its own git branch.

    Typical use, which is also the only use that restores state reliably::

        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            mutation = organism.mutate(candidate_source, label="cand-1")
            if not accepted:
                organism.revert_last()
            print(organism.diff_from_baseline())

    The branch is left behind for review; the operator is put back on the
    branch they started on.
    """

    def __init__(
        self,
        repo: Path,
        target: Path | str,
        *,
        branch: Optional[str] = None,
        branch_prefix: str = "evolve/code",
        allow_dirty: bool = False,
        git_binary: str = "git",
        command_timeout: int = 120,
    ) -> None:
        self.repo = Path(repo).expanduser()
        if not self.repo.is_dir():
            raise OrganismError(f"repo not found: {self.repo}")
        self.repo = self.repo.resolve()

        self.git_binary = git_binary
        self.command_timeout = command_timeout
        self.allow_dirty = allow_dirty
        self.branch_prefix = branch_prefix.strip("/")
        self._requested_branch = branch

        self.target = self._resolve_target(target)
        self.relpath = self.target.relative_to(self.repo).as_posix()

        self._branch: Optional[str] = None
        self._original_ref: Optional[str] = None
        self._original_was_detached = False
        self._baseline_sha: Optional[str] = None
        self._baseline_source: Optional[str] = None
        self._dirty_snapshot: Optional[str] = None
        self._mutations: list[Mutation] = []
        self._open = False

    # ── introspection ───────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """True between start() and close()."""
        return self._open

    @property
    def branch(self) -> Optional[str]:
        """The evolve branch this organism created, or None before start()."""
        return self._branch

    @property
    def original_ref(self) -> Optional[str]:
        """The ref the checkout was on before start(), restored by close()."""
        return self._original_ref

    @property
    def baseline_sha(self) -> Optional[str]:
        """HEAD when the branch was created, which every candidate is scored against."""
        return self._baseline_sha

    @property
    def baseline_source(self) -> str:
        """The target file's contents when the branch was created."""
        if self._baseline_source is None:
            raise OrganismError("organism is not open - call start() first")
        return self._baseline_source

    @property
    def lineage(self) -> tuple[Mutation, ...]:
        """Every mutation still on the branch, oldest first."""
        return tuple(self._mutations)

    def current_source(self) -> str:
        """Read the target file as it stands right now."""
        return self.target.read_text(encoding="utf-8")

    def describe(self) -> dict:
        """The repo, target, branch and full mutation lineage as a dict."""
        return {
            "repo": str(self.repo),
            "target": self.relpath,
            "branch": self._branch,
            "original_ref": self._original_ref,
            "baseline_sha": self._baseline_sha,
            "mutations": [m.to_dict() for m in self._mutations],
        }

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> "CodeOrganism":
        """Create the working branch and record the baseline.

        Refuses to run against a repo with uncommitted tracked changes unless
        ``allow_dirty=True`` was passed. Untracked files are ignored: they
        survive branch switches untouched, so they are not at risk.
        """
        if self._open:
            return self

        if not git_available():
            raise GitError("git is not installed or not on PATH")
        if not is_git_repo(self.repo, self.git_binary):
            raise GitError(f"not a git repository: {self.repo}")

        dirty = self._dirty_paths()
        if dirty and not self.allow_dirty:
            listed = ", ".join(dirty[:5]) + ("..." if len(dirty) > 5 else "")
            raise DirtyWorktreeError(
                f"{self.repo} has uncommitted changes ({listed}). "
                "Commit or stash them, or pass allow_dirty=True to evolve on top "
                "of them anyway."
            )

        self._original_ref, self._original_was_detached = self._current_ref()
        self._branch = self._requested_branch or self._generate_branch_name()

        # Everything worth recording is read before the checkout: `checkout -b`
        # changes neither HEAD's commit nor the worktree, so the values are
        # identical, and reading them first leaves nothing between creating the
        # branch and flipping _open. The moment the operator is on the new
        # branch, close() owns the way back - a probe that raised in that gap
        # used to strand them there with close() refusing to run.
        self._baseline_sha = self._git(["rev-parse", "HEAD"]).stdout.strip()
        self._baseline_source = self.current_source()
        self._dirty_snapshot = self._baseline_source if dirty else None
        self._mutations = []

        self._git(["checkout", "-b", self._branch])
        self._open = True
        return self

    def close(self, restore: bool = True) -> None:
        """Drop uncommitted changes and return to the original branch.

        Idempotent, and safe to call from an exception path - that is exactly
        when it matters most. Commits already made on the evolution branch are
        left alone; they are the review artifact.
        """
        if not self._open:
            return
        try:
            if restore:
                # Discard only the target file's uncommitted state. A repo-wide
                # reset would take work this class never touched.
                self._git(["checkout", "--", self.relpath], check=False)
                if self._original_ref:
                    self._git(["checkout", self._original_ref])
                if self._dirty_snapshot is not None:
                    if self.current_source() != self._dirty_snapshot:
                        self.target.write_text(self._dirty_snapshot, encoding="utf-8")
        finally:
            self._open = False

    def __enter__(self) -> "CodeOrganism":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    # ── mutation ────────────────────────────────────────────────────────

    def mutate(
        self,
        new_source: str,
        label: str,
        message: Optional[str] = None,
    ) -> Mutation:
        """Write *new_source* to the target file and commit it.

        Empty candidates are committed too (``--allow-empty``) so the lineage
        has one commit per candidate considered, not per candidate that
        happened to differ.
        """
        self._require_open()

        parent = self._git(["rev-parse", "HEAD"]).stdout.strip()
        self.target.write_text(new_source, encoding="utf-8")
        self._git(["add", "--", self.relpath])

        subject = message or f"evolve({self.target.stem}): candidate {label}"
        self._git(["commit", "--allow-empty", "--no-verify", "-m", subject])

        sha = self._git(["rev-parse", "HEAD"]).stdout.strip()
        diff = self._diff(parent, sha)

        mutation = Mutation(
            index=len(self._mutations) + 1,
            label=label,
            message=subject,
            sha=sha,
            parent_sha=parent,
            diff=diff,
            source=new_source,
        )
        self._mutations.append(mutation)
        return mutation

    def revert_last(self) -> Optional[Mutation]:
        """Undo the most recent mutation, commit and all.

        Returns the mutation that was dropped, or None when there is nothing
        to undo. The reset is confined to the branch this organism created.
        """
        self._require_open()
        if not self._mutations:
            return None
        dropped = self._mutations.pop()
        self._rewind_to(dropped.parent_sha)
        return dropped

    def revert_to_baseline(self) -> None:
        """Undo every mutation and return the file to its baseline content."""
        self._require_open()
        assert self._baseline_sha is not None  # guaranteed by _require_open
        self._rewind_to(self._baseline_sha)
        self._mutations = []

    def reapply(self, mutation: Mutation, label: Optional[str] = None) -> Mutation:
        """Re-commit a previously reverted mutation on top of the current HEAD.

        Candidates are all generated against the baseline, so they are
        alternatives rather than a sequence. The winner is chosen after the
        fact and put back with this.
        """
        return self.mutate(
            mutation.source,
            label=label or mutation.label,
            message=f"evolve({self.target.stem}): accepted candidate {mutation.label}",
        )

    # ── diffs ───────────────────────────────────────────────────────────

    def diff_from_baseline(self) -> str:
        """Unified diff of the target file, baseline to current HEAD."""
        self._require_open()
        assert self._baseline_sha is not None
        head = self._git(["rev-parse", "HEAD"]).stdout.strip()
        return self._diff(self._baseline_sha, head)

    def diff_working_tree(self) -> str:
        """Unified diff of uncommitted changes to the target file."""
        self._require_open()
        return self._git(
            ["--no-pager", "diff", "--no-ext-diff", "--no-color", "--", self.relpath]
        ).stdout

    # ── internals ───────────────────────────────────────────────────────

    def _resolve_target(self, target: Path | str) -> Path:
        candidate = Path(target).expanduser()
        if not candidate.is_absolute():
            candidate = self.repo / candidate
        candidate = candidate.resolve()

        if not candidate.is_file():
            raise OrganismError(f"target file not found: {candidate}")
        try:
            candidate.relative_to(self.repo)
        except ValueError as exc:
            raise OrganismError(
                f"target {candidate} is outside the repo {self.repo}"
            ) from exc
        return candidate

    def _generate_branch_name(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.branch_prefix}/{self.target.stem}-{stamp}"

    def _current_ref(self) -> tuple[str, bool]:
        """Return the checked-out branch, or the HEAD sha when detached."""
        name = self._git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        if name and name != "HEAD":
            return name, False
        return self._git(["rev-parse", "HEAD"]).stdout.strip(), True

    def _dirty_paths(self) -> list[str]:
        out = self._git(
            ["status", "--porcelain", "--untracked-files=no"]
        ).stdout.strip()
        return [line[3:].strip() for line in out.splitlines() if line.strip()]

    def _diff(self, left: str, right: str) -> str:
        return self._git(
            [
                "--no-pager",
                "diff",
                "--no-ext-diff",
                "--no-color",
                left,
                right,
                "--",
                self.relpath,
            ]
        ).stdout

    def _require_open(self) -> None:
        if not self._open:
            raise OrganismError(
                "organism is not open - use it as a context manager or call start()"
            )

    def _rewind_to(self, sha: str) -> None:
        """Move the branch back to *sha*, touching only the target file.

        ``git reset --hard`` would do this in one call, and that is what this
        used to be. But ``--hard`` rewrites the entire working tree, so an
        operator running with ``allow_dirty=True`` lost uncommitted work in
        files this class never staged - and ``revert_last`` runs once per
        candidate, so the first one took it. Rewind the ref with ``--soft``,
        which leaves the working tree alone, then restore the one path this
        organism is allowed to move.
        """
        self._git(["reset", "--soft", sha])
        restored = self._git(["checkout", sha, "--", self.relpath], check=False)
        if restored.returncode != 0:
            # The target does not exist at *sha*: it was untracked when the
            # organism started and only entered history via mutate(). Leave
            # what a hard reset left - no index entry and no file.
            self._git(["rm", "--cached", "--force", "--quiet", "--", self.relpath],
                      check=False)
            self.target.unlink(missing_ok=True)

    def _identity_args(self) -> list[str]:
        """Supply an author only when the repo has none configured."""
        # Through the bounded wrapper like every other git call, so a git that
        # blocks on an index lock or a credential helper becomes a GitError
        # with a message instead of a raw subprocess exception the callers do
        # not handle. No recursion: _git prepends identity args only for
        # `commit`, and this probe runs `config`.
        probe = self._git(["config", "--get", "user.email"], check=False)
        if probe.returncode == 0 and probe.stdout.strip():
            return []
        return [
            "-c",
            f"user.name={FALLBACK_AUTHOR_NAME}",
            "-c",
            f"user.email={FALLBACK_AUTHOR_EMAIL}",
        ]

    def _git(self, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.git_binary, "-c", "commit.gpgsign=false"]
        if args and args[0] == "commit":
            cmd.extend(self._identity_args())
        cmd.extend(args)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.repo),
                timeout=self.command_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError(
                f"git {' '.join(args)} timed out after {self.command_timeout}s"
            ) from exc
        except OSError as exc:
            raise GitError(f"could not run git {' '.join(args)}: {exc}") from exc

        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise GitError(f"git {' '.join(args)} failed: {detail}")
        return proc
