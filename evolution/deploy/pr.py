"""Open a pull request for an evolved artifact.

The README promised that "all changes go through PR review, never direct
commit". ``create_pr`` existed as a config field and nothing implemented it —
there was no branch, no commit, no ``gh`` call anywhere in the tree. Evolution
wrote a file into ``output/`` and stopped, so the guardrail was documentation
rather than behaviour.

This closes that. The PR body carries the constraint report and the A/B
summary, because a reviewer looking at a machine-authored prompt diff needs to
see what was measured, on how many observations, and against what noise band —
the diff alone does not tell them whether to merge.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Branch names are derived from the skill name, which comes from a filesystem
# path — restrict to what git accepts without quoting.
_UNSAFE_BRANCH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class PRResult:
    """Outcome of a deployment attempt."""

    created: bool
    branch: str = ""
    url: str = ""
    detail: str = ""
    commands: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.created:
            return f"PR opened on {self.branch}: {self.url or '(no url reported)'}"
        return f"PR not opened: {self.detail}"


def _run(
    args: Sequence[str],
    cwd: Path,
    log: list[str],
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    log.append(" ".join(args))
    result = subprocess.run(
        list(args), cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{' '.join(args)} failed: {stderr[:400]}")
    return result


def safe_branch_name(skill_name: str, timestamp: str) -> str:
    slug = _UNSAFE_BRANCH_CHARS.sub("-", skill_name).strip("-").lower() or "skill"
    return f"evolve/{slug}-{timestamp}"


def build_pr_body(
    skill_name: str,
    report_markdown: str,
    constraint_lines: Sequence[str],
    run_metadata: Optional[dict] = None,
) -> str:
    """Assemble a review-ready PR body.

    Everything a reviewer needs to accept or reject without re-running the
    evaluation themselves: the measured result with its noise band, every
    constraint that was checked, and how the run was configured.
    """
    parts = [
        f"Automated skill evolution for `{skill_name}`.",
        "",
        "This branch was produced by an optimizer, not a human. Read the "
        "measurement below before the diff: a prompt change that reads well "
        "can still be a regression, and a delta inside the noise band is not "
        "evidence of anything.",
        "",
        report_markdown.strip(),
        "",
        "## Constraint gates",
        "",
    ]
    parts.extend(f"- {line}" for line in constraint_lines)

    if run_metadata:
        parts += ["", "## Run configuration", ""]
        for key, value in run_metadata.items():
            parts.append(f"- **{key}**: {value}")

    parts += [
        "",
        "---",
        "",
        "Merging deploys this skill to every agent that loads it. If the "
        "verdict above is HOLD, close this rather than merging.",
    ]
    return "\n".join(parts)


class PRPublisher:
    """Commits an evolved artifact to a branch and opens a PR for it."""

    def __init__(
        self,
        repo: Path,
        base_branch: str = "main",
        draft: bool = True,
        remote: str = "origin",
    ):
        self.repo = Path(repo)
        self.base_branch = base_branch
        self.draft = draft
        self.remote = remote

    # ── preflight ────────────────────────────────────────────────────────

    def preflight(self) -> tuple[bool, str]:
        """Check everything needed before touching the repository."""
        if not (self.repo / ".git").exists():
            return False, f"{self.repo} is not a git repository"
        if shutil.which("git") is None:
            return False, "git not found on PATH"
        try:
            status = _run(["git", "status", "--porcelain"], self.repo, [], check=True)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            return False, f"git status failed: {exc}"
        if status.stdout.strip():
            return False, (
                "working tree is dirty — refusing to branch off uncommitted "
                "changes that are not ours"
            )
        return True, "ready"

    def has_gh(self) -> bool:
        return shutil.which("gh") is not None

    # ── publish ──────────────────────────────────────────────────────────

    def publish(
        self,
        skill_name: str,
        target_path: Path,
        content: str,
        title: str,
        body: str,
        timestamp: str,
        push: bool = True,
        dry_run: bool = False,
    ) -> PRResult:
        """Write one artifact on a new branch, commit, push, and open a PR.

        Args:
            target_path: Absolute path of the file to write, inside the repo.
            push: When False, stop after committing locally. Useful for
                inspecting the branch before anything leaves the machine.
            dry_run: Report what would happen and change nothing.
        """
        return self.publish_many(
            skill_name=skill_name,
            files={target_path: content},
            title=title,
            body=body,
            timestamp=timestamp,
            push=push,
            dry_run=dry_run,
        )

    def publish_many(
        self,
        skill_name: str,
        files: dict[Path, str],
        title: str,
        body: str,
        timestamp: str,
        push: bool = True,
        dry_run: bool = False,
    ) -> PRResult:
        """Write several files as ONE branch and ONE commit.

        A change that spans files is atomic, and publishing it file-by-file is
        not a smaller version of the same thing — it produces one branch per
        file, each holding a fraction of the change, all labelled as if they
        held the whole. The first is pushed, the rest are stranded locally, and
        the branch reported back is the one that never left the machine.
        """
        log: list[str] = []
        branch = safe_branch_name(skill_name, timestamp)

        if not files:
            return PRResult(created=False, branch=branch, detail="no files to publish")

        rels: list[str] = []
        for target_path in files:
            try:
                rels.append(str(Path(target_path).resolve().relative_to(self.repo.resolve())))
            except ValueError:
                return PRResult(
                    created=False,
                    branch=branch,
                    detail=f"{target_path} is outside the repo {self.repo}",
                )
        rel = ", ".join(rels)

        if dry_run:
            return PRResult(
                created=False,
                branch=branch,
                detail=(
                    f"dry run — would write {rel}, commit on {branch}, "
                    f"and open a PR against {self.base_branch}"
                ),
                commands=[
                    f"git checkout -b {branch}",
                    *(
                        f"write {r} ({len(c):,} chars)"
                        for r, c in zip(rels, files.values(), strict=True)
                    ),
                    f"git commit -m {title!r}",
                    f"git push -u {self.remote} {branch}",
                    "gh pr create …",
                ],
            )

        ok, reason = self.preflight()
        if not ok:
            return PRResult(created=False, branch=branch, detail=reason)

        original_branch = ""
        try:
            original_branch = _run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], self.repo, log
            ).stdout.strip()

            _run(["git", "checkout", "-b", branch], self.repo, log)
            for target_path, content in files.items():
                path = Path(target_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            _run(["git", "add", *rels], self.repo, log)

            # Nothing staged means the optimizer produced a byte-identical
            # file; opening an empty PR would waste a reviewer's time.
            staged = _run(["git", "diff", "--cached", "--stat"], self.repo, log)
            if not staged.stdout.strip():
                _run(["git", "checkout", original_branch], self.repo, log, check=False)
                _run(["git", "branch", "-D", branch], self.repo, log, check=False)
                return PRResult(
                    created=False,
                    branch=branch,
                    detail="evolved artifact is identical to the current file(s)",
                    commands=log,
                )

            _run(["git", "commit", "-m", title, "-m", body], self.repo, log)

            if not push:
                return PRResult(
                    created=False,
                    branch=branch,
                    detail=f"committed locally on {branch}; push skipped",
                    commands=log,
                )

            _run(["git", "push", "-u", self.remote, branch], self.repo, log)

            if not self.has_gh():
                return PRResult(
                    created=False,
                    branch=branch,
                    detail=(
                        f"branch pushed to {self.remote}/{branch}, but gh is not "
                        "installed so no PR was opened — open it manually"
                    ),
                    commands=log,
                )

            args = [
                "gh", "pr", "create",
                "--base", self.base_branch,
                "--head", branch,
                "--title", title,
                "--body", body,
            ]
            if self.draft:
                args.append("--draft")
            created = _run(args, self.repo, log)
            url = _extract_url(created.stdout)
            return PRResult(created=True, branch=branch, url=url, commands=log)

        except Exception as exc:  # noqa: BLE001 — report, never crash the run
            return PRResult(
                created=False,
                branch=branch,
                detail=f"{type(exc).__name__}: {exc}",
                commands=log,
            )
        finally:
            # Leave the repo on the branch it started on so a failed publish
            # does not strand the working copy somewhere unexpected.
            if original_branch:
                try:
                    current = _run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        self.repo, [], check=False,
                    ).stdout.strip()
                    if current != original_branch:
                        _run(
                            ["git", "checkout", original_branch],
                            self.repo, [], check=False,
                        )
                except Exception:  # noqa: BLE001
                    pass


def _extract_url(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("http"):
            return line
    return ""
