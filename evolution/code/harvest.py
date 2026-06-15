"""Real-bug harvester for the code-evolution measurement campaign.

Mines a target repo's git history for authentic tool bugs. A commit that fixed a
tool — touching both ``tools/X.py`` and its ``tests/tools/test_X.py`` — pairs the
PARENT commit's (buggy) tool source with the fix commit's test. That parent state
is a real bug iff the pre-fix source fails ≥1 test the upstream fix passes. The
fix commit is kept as a ground-truth **oracle**: the campaign verifies a repair by
behavioral match to the upstream fix, so the harvest never needs the held-out
split (whose independence is limited).

Records **refs, not blobs** — sources are re-materialized from git at run time.
Validity is established in the real :class:`WorktreeEnv` (full env + conftest), not
a bare tmpdir, so the harvester is not limited to self-contained tools.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from evolution.code.gate import _parse_pytest_failures
from evolution.code.worktree import WorktreeEnv, WorktreeError

_GIT_TIMEOUT = 120


@dataclass(frozen=True)
class Candidate:
    """A commit that fixed a tool, before validity is established."""

    tool_path: str  # repo-relative, e.g. tools/fuzzy_match.py
    test_path: str  # repo-relative, e.g. tests/tools/test_fuzzy_match.py
    fix_sha: str
    parent_sha: str


@dataclass(frozen=True)
class HarvestedBug:
    """A validated organism: a real bug with an upstream-fix oracle.

    ``bug_tests`` are the test node-ids the parent fails but the upstream fix
    passes — the bug-catching subset (robust to env-flaky tests elsewhere in the
    file). The campaign repairs against these and verifies the repair behaves
    like the oracle.
    """

    tool_path: str
    test_path: str
    fix_sha: str
    parent_sha: str
    bug_tests: tuple[str, ...]  # node-id suffixes the parent fails & the fix passes
    fail_excerpt: str = ""


def _git(repo: Path, *args: str, timeout: int = _GIT_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout
    )


def tool_for_test(test_relpath: str) -> str | None:
    """Map ``tests/tools/test_X.py`` → ``tools/X.py`` (the repo convention)."""
    name = test_relpath.rsplit("/", 1)[-1]
    if not (name.startswith("test_") and name.endswith(".py")):
        return None
    return "tools/" + name[len("test_"):]


def discover_targets(repo: Path) -> list[tuple[str, str]]:
    """All ``(tool_path, test_path)`` pairs where both files exist at HEAD.

    No import/self-containment filter: validity runs in the real worktree (full
    env), so fixture-using tests are in scope — which is what widens the supply
    beyond the handful of self-contained tools.
    """
    res = _git(repo, "ls-files", "tests/tools/test_*.py")
    pairs: list[tuple[str, str]] = []
    for test_path in res.stdout.splitlines():
        test_path = test_path.strip()
        tool_path = tool_for_test(test_path)
        if not tool_path:
            continue
        if (repo / tool_path).exists():
            pairs.append((tool_path, test_path))
    return pairs


def harvest_candidates(
    repo: Path,
    targets: list[tuple[str, str]] | None = None,
    *,
    max_commits_per_tool: int = 40,
    since_days: int | None = None,
) -> list[Candidate]:
    """Bug-fix-shaped commits: those touching both the tool and its test.

    Cheap (git only). Each candidate's validity is established later in a worktree.
    ``since_days`` restricts to commits within that recency window (the monitor
    sentinel uses it to scan the recent stream rather than all history).
    """
    targets = targets if targets is not None else discover_targets(repo)
    since_args = ["--since", f"{since_days} days ago"] if since_days else []
    candidates: list[Candidate] = []
    for tool_path, test_path in targets:
        log = _git(repo, "log", "--format=%H %P", "-n", str(max_commits_per_tool),
                   *since_args, "--no-merges", "--", test_path)
        for line in log.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue  # root commit (no parent) — can't form a parent/fix pair
            fix_sha, parent_sha = parts[0], parts[1]
            # Keep only commits that also touched the tool itself (a fix, not a
            # test-only edit) — the bug-fix shape.
            touched = _git(repo, "show", "--name-only", "--format=", fix_sha)
            files = {ln.strip() for ln in touched.stdout.splitlines() if ln.strip()}
            if tool_path in files and test_path in files:
                candidates.append(Candidate(tool_path, test_path, fix_sha, parent_sha))
    return candidates


def stratify(candidates: list[Candidate], *, max_per_tool: int | None = 3) -> list[Candidate]:
    """Round-robin interleave candidates across tools, capping per tool.

    Commits from one tool are not independent organisms (they share a code
    surface and failure modes), and `harvest_candidates` returns them grouped by
    tool. Interleaving + a per-tool cap keeps the campaign's organism-level sample
    diverse so its cluster statistics aren't dominated by one prolific tool.
    """
    by_tool: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_tool.setdefault(c.tool_path, []).append(c)
    if max_per_tool is not None:
        by_tool = {t: cs[:max_per_tool] for t, cs in by_tool.items()}
    out: list[Candidate] = []
    queues = list(by_tool.values())
    i = 0
    while any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
        if i % len(queues) == 0:
            queues = [q for q in queues if q]
            i = 0
            if not queues:
                break
    return out


def _failures(env: WorktreeEnv, test_relpath: str) -> set[str]:
    """Node-ids that fail when running the whole test file in the worktree."""
    run = env.run_test(test_relpath, extra_args=["--tb=no"], full_output=True)
    return _parse_pytest_failures(run.output)


def validate_candidate(
    repo: Path, c: Candidate, *, base_python: str | None = None
) -> HarvestedBug | None:
    """Establish oracle-validity for one candidate, in the real worktree.

    A candidate is valid iff there is a non-empty ``bug_tests`` set — tests the
    buggy parent fails that the upstream fix passes. Computing the fix's own
    failures first makes this robust to env-flaky tests elsewhere in the file
    (network/service tests that fail regardless of the bug): those appear in
    *both* runs and cancel out of the difference.
    """
    try:
        env = WorktreeEnv.create(repo, base_ref=c.fix_sha, base_python=base_python)
    except WorktreeError:
        return None
    try:
        env.assert_authoritative(c.tool_path.split("/")[0])
        # worktree is at fix_sha → the tool is the upstream fix (oracle).
        fix_failures = _failures(env, c.test_path)
        parent_src = _git(repo, "show", f"{c.parent_sha}:{c.tool_path}")
        if parent_src.returncode != 0:
            return None
        env.write_tool(c.tool_path, parent_src.stdout)
        parent_failures = _failures(env, c.test_path)
        bug_tests = tuple(sorted(parent_failures - fix_failures))
        if not bug_tests:
            return None  # parent doesn't fail anything the fix passes → not a clean bug
        return HarvestedBug(
            tool_path=c.tool_path, test_path=c.test_path, fix_sha=c.fix_sha,
            parent_sha=c.parent_sha, bug_tests=bug_tests,
            fail_excerpt="; ".join(bug_tests[:5]),
        )
    except WorktreeError:
        return None
    finally:
        env.destroy()


def recon(
    repo: Path,
    candidates: list[Candidate],
    *,
    target_valid: int = 20,
    cap: int | None = None,
    base_python: str | None = None,
    on_result=None,
) -> list[HarvestedBug]:
    """Validate candidates until ``target_valid`` oracle-valid bugs are found (or
    ``cap`` candidates are checked). Pure git + worktree + pytest — no LM spend.
    ``on_result(candidate, bug_or_None)`` is called after each check for progress.
    """
    valid: list[HarvestedBug] = []
    for i, c in enumerate(candidates):
        if cap is not None and i >= cap:
            break
        bug = validate_candidate(repo, c, base_python=base_python)
        if on_result is not None:
            on_result(c, bug)
        if bug is not None:
            valid.append(bug)
            if len(valid) >= target_valid:
                break
    return valid
