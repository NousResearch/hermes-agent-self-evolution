"""GitBasedOrganism wrapper for Phase 4 code evolution.

Each baseline tool file is treated as the canonical organism. Candidates live in
disposable ``git worktree`` directories under a session root, each tagged with
the organism name + iteration index. Mutations are written into the worktree,
committed (so the diff is auditable), then evaluated by sandboxed pytest/ruff.

This module deliberately does not import any AGPL-licensed code. The
``InternalMutator`` is fed baseline source and a bug brief; the resulting
candidate file is what we drop into the worktree.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import tempfile


class OrganismError(RuntimeError):
    """Raised for worktree/git failures."""


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OrganismError(f"git {' '.join(args)} failed: {exc}") from exc


@dataclass(frozen=True)
class CandidateOrganism:
    """A mutated tool file living in its own git worktree."""

    organism_name: str
    iteration: int
    worktree_path: Path
    relative_tool_path: Path
    branch_name: str

    @property
    def tool_path(self) -> Path:
        return self.worktree_path / self.relative_tool_path

    @property
    def candidate_sha256(self) -> str:
        return sha256(self.tool_path.read_bytes()).hexdigest()


class CodeOrganism:
    """The baseline tool file plus a factory for mutated candidates."""

    def __init__(
        self,
        *,
        repo_root: Path,
        relative_tool_path: Path,
        session_root: Path | None = None,
    ):
        self.repo_root = repo_root.resolve()
        self.relative_tool_path = Path(relative_tool_path)
        absolute = (self.repo_root / self.relative_tool_path).resolve()
        try:
            absolute.relative_to(self.repo_root)
        except ValueError as exc:
            raise OrganismError(
                f"tool path {absolute} is outside repo root {self.repo_root}",
            ) from exc
        if not absolute.is_file():
            raise OrganismError(f"tool file does not exist: {absolute}")
        self.baseline_tool_path = absolute
        self.organism_name = self.relative_tool_path.stem
        self.session_root = (session_root or Path(tempfile.mkdtemp(prefix="hermes-evo-"))).resolve()
        self.session_root.mkdir(parents=True, exist_ok=True)
        self._worktrees: list[Path] = []

        if not (self.repo_root / ".git").exists():
            raise OrganismError(
                f"repo root {self.repo_root} is not a git repository; worktrees require git",
            )

    @property
    def baseline_sha256(self) -> str:
        return sha256(self.baseline_tool_path.read_bytes()).hexdigest()

    @property
    def baseline_source(self) -> str:
        return self.baseline_tool_path.read_text(encoding="utf-8")

    def create_candidate(
        self,
        *,
        iteration: int,
        mutated_source: str,
        commit_message: str | None = None,
    ) -> CandidateOrganism:
        """Materialize ``mutated_source`` in a fresh worktree."""
        branch_name = f"hermes-evo/{self.organism_name}/iter-{iteration:03d}"
        worktree_path = (self.session_root / branch_name).resolve()
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        head = _run_git(["rev-parse", "HEAD"], cwd=self.repo_root)
        if head.returncode != 0:
            raise OrganismError(
                f"repo {self.repo_root} has no HEAD commit: {head.stderr.strip()}",
            )
        base_ref = head.stdout.strip()

        # Clean up any stale branch from a previous aborted run.
        _run_git(["branch", "-D", branch_name], cwd=self.repo_root)

        result = _run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), base_ref],
            cwd=self.repo_root,
        )
        if result.returncode != 0:
            raise OrganismError(
                f"git worktree add failed: {result.stderr.strip() or result.stdout.strip()}",
            )

        target = (worktree_path / self.relative_tool_path).resolve()
        try:
            target.relative_to(worktree_path)
        except ValueError as exc:
            raise OrganismError("relative tool path escaped worktree root") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(mutated_source, encoding="utf-8")

        add_res = _run_git(["add", str(self.relative_tool_path)], cwd=worktree_path)
        if add_res.returncode != 0:
            raise OrganismError(f"git add failed: {add_res.stderr.strip()}")

        msg = commit_message or f"hermes-evo: {self.organism_name} iter {iteration}"
        commit_res = _run_git(
            [
                "-c", "user.email=hermes-evo@local",
                "-c", "user.name=hermes-evo",
                "commit", "-m", msg, "--allow-empty",
            ],
            cwd=worktree_path,
        )
        if commit_res.returncode != 0:
            raise OrganismError(f"git commit failed: {commit_res.stderr.strip()}")

        self._worktrees.append(worktree_path)
        return CandidateOrganism(
            organism_name=self.organism_name,
            iteration=iteration,
            worktree_path=worktree_path,
            relative_tool_path=self.relative_tool_path,
            branch_name=branch_name,
        )

    def cleanup(self) -> None:
        """Remove every worktree this organism produced. Safe to call repeatedly."""
        for worktree in list(self._worktrees):
            _run_git(["worktree", "remove", "--force", str(worktree)], cwd=self.repo_root)
            if worktree.exists():
                shutil.rmtree(worktree, ignore_errors=True)
        self._worktrees.clear()
