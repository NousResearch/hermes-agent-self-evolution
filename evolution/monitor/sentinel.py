"""Self-hosting triage sentinel for the code-evolution loop (item 11 / Phase 5).

Propose-only: scans a target repo's recent git stream for repair candidates the
validated repair loop could attempt, classifies and ranks them, and (via
:mod:`evolution.monitor.queue`) emits a triage queue with ready-to-run commands. It
NEVER auto-evolves or opens PRs — a human pulls the trigger. This is the
continuous-improvement front-end the loop was missing: the loop is the consumer, the
sentinel is the supply.

Candidates come from the recent fix-stream (the harvester, time-windowed). Each is
classified by kind: a ``dependency_regression`` (the fix commit also changed a
dependency manifest — the genuinely-novel case whose pre-bump behavior is a correct
reference, ranked first) or a general ``bug_fix``. Verification, when a candidate is
attempted, reuses the existing oracle gate (no new verifier).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evolution.code.harvest import Candidate, _git, harvest_candidates

# Dependency-manifest filenames whose change in a fix commit marks a
# dependency-regression repair (a bump broke a test; the fix adapted the tool).
_DEP_FILES = (
    "pyproject.toml", "uv.lock", "poetry.lock", "Pipfile.lock", "setup.py",
    "setup.cfg", "requirements.txt", "requirements.in",
)
_KIND_WEIGHT = {"dependency_regression": 2.0, "bug_fix": 1.0}


@dataclass(frozen=True)
class RepairCandidate:
    """A ranked, classified repair target for the triage queue."""

    tool_path: str
    test_path: str
    fix_sha: str
    parent_sha: str
    kind: str            # "dependency_regression" | "bug_fix"
    committed_at: str    # ISO 8601 committer date (also the recency sort key)

    @property
    def score(self) -> float:
        return _KIND_WEIGHT.get(self.kind, 1.0)


def _is_dep_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in _DEP_FILES


def classify(repo: Path, c: Candidate) -> RepairCandidate:
    """Tag a candidate by kind + commit date from a single ``git show``."""
    res = _git(repo, "show", "--name-only", "--format=%cI", c.fix_sha)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    committed_at = lines[0] if lines else ""
    files = lines[1:]
    kind = "dependency_regression" if any(_is_dep_file(f) for f in files) else "bug_fix"
    return RepairCandidate(c.tool_path, c.test_path, c.fix_sha, c.parent_sha,
                           kind, committed_at)


def scan(
    repo: Path,
    *,
    since_days: int = 90,
    max_per_tool: int = 5,
    targets: list[tuple[str, str]] | None = None,
) -> list[RepairCandidate]:
    """Scan ``repo``'s recent stream for repair candidates, classified and ranked.

    Ranking is git-only (no worktree/LM): dependency-regressions first (their
    pre-bump behavior is a correct reference), then most-recent. Difficulty/value
    isn't estimated here — that needs a worktree — and the queue is propose-only,
    so a human (or a separate --attempt pass) decides what to actually repair.
    """
    candidates = harvest_candidates(
        repo, targets, max_commits_per_tool=max_per_tool, since_days=since_days)
    ranked = [classify(repo, c) for c in candidates]
    ranked.sort(key=lambda r: (r.score, r.committed_at), reverse=True)
    return ranked
