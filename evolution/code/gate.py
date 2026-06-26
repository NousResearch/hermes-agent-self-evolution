"""Deploy gate for code evolution — the project's distinctive asset.

The repair loop is commodity (an LLM fixing code under a test). What makes a
deploy trustworthy is the gate, and a passing test is a *weak* verifier: under a
whole-file rewrite the proposer can pass the one failing test while gaming an
incomplete suite — the "suite states the win" failure, transplanted to code and
worse. This gate is the set of checks that make a green test mean what it says.

Checks, cheap-and-decisive first; the first failure rejects:

  1. repair landed     — the visible split actually passes (free).
  2. surface freeze     — no public function/class/signature/schema drift, and
                          the rewrite is within blast-radius bounds (free, AST).
  3. file scope         — the worktree's git diff touches *only* the target tool;
                          no test file, no other module (cheap git).
  4. held-out split     — a frozen test the proposer never saw and was never fed
                          back must also pass. This is the anti-gaming core: a
                          fix that teaches to the visible test (hard-codes its
                          expected value) fails here. Deploy requires both splits.
  5. regression floor   — the `tests/tools` subset stays green, so a repair that
                          fixes one tool but breaks a sibling is caught.

A sixth tier — the full Hermes suite — is too expensive to run per repair (tens
of thousands of tests); the CLI runs it once at PR time via the benchmark hook.
Every decision records *which* floor actually ran, so a subset is never able to
masquerade as the full suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from evolution.code.freeze_check import DEFAULT_MIN_RETAIN_RATIO, freeze_violations
from evolution.code.repair import RepairResult
from evolution.code.worktree import WorktreeEnv

GATE_SCHEMA_VERSION = "1"
DEFAULT_FLOOR_PATHS = ("tests/tools",)
_PYTEST_NO_TESTS_COLLECTED = 5  # pytest exit code: zero tests ran


class CodeGateError(RuntimeError):
    """The gate could not establish a precondition it needs to judge honestly
    (e.g. it could not derive the pre-repair baseline). Caught and turned into a
    hard reject — never an implicit pass."""


@dataclass
class CodeGateResult:
    """The gate's verdict plus the structured decision to persist."""

    deploy: bool
    reason: str
    decision: dict = field(default_factory=dict)


def _test_relnorm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _parse_pytest_failures(output: str) -> set[str]:
    """The set of failing/erroring test node ids from pytest's short summary.

    pytest prints ``FAILED <nodeid> - <reason>`` / ``ERROR <nodeid> ...`` lines;
    we take the node id so two runs can be compared by identity. Used by the
    baseline-diff floor to isolate failures the repair introduced from
    environment-level ones present on both runs."""
    failures: set[str] = set()
    for line in output.splitlines():
        s = line.strip()
        for prefix in ("FAILED ", "ERROR "):
            if s.startswith(prefix):
                nodeid = s[len(prefix):].split(" - ", 1)[0].strip()
                if nodeid:
                    failures.add(nodeid)
    return failures


def _is_test_path(path: str) -> bool:
    """Whether ``path`` is a test artifact a repair must never modify: any file
    under a ``tests/`` directory, a ``test_*.py``/``*_test.py`` module, or a
    ``conftest.py`` (which silently shapes test behavior)."""
    name = path.rsplit("/", 1)[-1]
    return (
        path.startswith("tests/")
        or "/tests/" in path
        or name == "conftest.py"
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
    )


def run_code_gate(
    env: WorktreeEnv,
    *,
    tool_relpath: str,
    visible_test_relpath: str,
    holdout_test_relpath: str,
    repair_result: RepairResult,
    floor_paths: tuple[str, ...] = DEFAULT_FLOOR_PATHS,
    min_retain_ratio: float = DEFAULT_MIN_RETAIN_RATIO,
    run_inputs: Optional[dict] = None,
) -> CodeGateResult:
    """Evaluate the repaired worktree and return a deploy/reject decision.

    The worktree must hold the repaired source (the repair engine wrote it) and
    be authoritative. ``run_inputs`` is recorded verbatim for later calibration.
    """
    guards: dict = {}
    decision: dict = {
        "schema_version": GATE_SCHEMA_VERSION,
        "artifact_type": "code",
        "decision_signal": "deterministic_test",
        "target_tool": tool_relpath,
        "visible_test": visible_test_relpath,
        "holdout_test": holdout_test_relpath,
        "repair": {
            "fixed": repair_result.fixed,
            "fixed_round": repair_result.fixed_round,
            "rounds_used": len(repair_result.rounds),
        },
        "guards": guards,
        "run_inputs": run_inputs or {},
    }

    def _reject(reason: str) -> CodeGateResult:
        decision["decision"] = "reject"
        decision["reason"] = reason
        return CodeGateResult(deploy=False, reason=reason, decision=decision)

    # 0. config sanity: a held-out split equal to the visible split provides
    # zero anti-gaming signal — the gate's central check would be a tautology.
    if _test_relnorm(holdout_test_relpath) == _test_relnorm(visible_test_relpath):
        return _reject("held-out test path equals the visible test path — "
                       "the held-out split would provide no anti-gaming signal")

    # 1. repair landed
    guards["repair_passed_visible"] = repair_result.fixed
    if not repair_result.fixed or repair_result.final_source is None:
        return _reject("repair did not produce a fix that passes the visible test")
    repaired = repair_result.final_source
    # The repaired file is on disk; derive the pre-repair base from the
    # worktree's HEAD so the freeze compares pre- vs post-repair, not post vs post.
    try:
        base_src = _base_source(env, tool_relpath)
    except CodeGateError as exc:
        return _reject(str(exc))

    # 2. surface freeze + blast radius
    violations = freeze_violations(base_src, repaired, min_retain_ratio=min_retain_ratio)
    guards["freeze_ok"] = not violations
    guards["freeze_violations"] = violations
    if violations:
        return _reject("freeze/diff-shape violation: " + "; ".join(violations))

    # 3. file scope — only the target tool may change; no test file touched
    changed = [_test_relnorm(p) for p in env.changed_files()]
    guards["changed_files"] = changed
    target = _test_relnorm(tool_relpath)
    offenders = [p for p in changed if p != target]
    test_touched = [p for p in changed if _is_test_path(p)]
    guards["file_scope_ok"] = not offenders and not test_touched
    if test_touched:
        return _reject(f"a test file was modified: {test_touched}")
    if offenders:
        return _reject(f"files other than the target tool changed: {offenders}")

    # 4. held-out split (the anti-gaming core)
    holdout = env.run_test(holdout_test_relpath)
    guards["holdout"] = {"passed": holdout.passed, "exit_code": holdout.exit_code,
                         "duration_seconds": round(holdout.duration_seconds, 2)}
    if holdout.exit_code == _PYTEST_NO_TESTS_COLLECTED:
        decision["holdout_output_tail"] = holdout.output[-2000:]
        return _reject("held-out test split collected no tests — no anti-gaming "
                       "signal (check the --holdout-test path)")
    if not holdout.passed:
        decision["holdout_output_tail"] = holdout.output[-2000:]
        return _reject("held-out test split failed — the fix does not generalize "
                       "beyond the visible test (teaching-to-the-test)")

    # 5. regression floor — no NEW failures vs the pre-repair baseline. A large
    # suite (or an isolated venv missing optional deps) routinely has unrelated
    # pre-existing failures, so demanding absolute green would reject every
    # repair. Diff the failing-test sets instead: run the floor on the repaired
    # source, then on the base source, and reject only failures the repair
    # *introduced*. (`--tb=no` + full_output keeps the FAILED summary complete
    # and parseable.)
    repaired_floor = env.run_test(*floor_paths, extra_args=["--tb=no"], full_output=True)
    if repaired_floor.exit_code == _PYTEST_NO_TESTS_COLLECTED:
        decision["floor_output_tail"] = repaired_floor.output[-2000:]
        return _reject(f"regression floor {list(floor_paths)} collected no tests "
                       f"(check the --floor-path)")
    repaired_failures = _parse_pytest_failures(repaired_floor.output)

    if base_src == "":
        # Brand-new file: no baseline to diff against, so require absolute green.
        base_failures: set[str] = set()
        new_failures = sorted(repaired_failures)
    else:
        env.write_tool(tool_relpath, base_src)
        try:
            base_floor = env.run_test(*floor_paths, extra_args=["--tb=no"], full_output=True)
        finally:
            env.write_tool(tool_relpath, repaired)  # always restore the repair
        base_failures = _parse_pytest_failures(base_floor.output)
        new_failures = sorted(repaired_failures - base_failures)

    guards["floor"] = {
        "ran": list(floor_paths),
        "new_failures": new_failures,
        "base_failure_count": len(base_failures),
        "repaired_failure_count": len(repaired_failures),
        "duration_seconds": round(repaired_floor.duration_seconds, 2),
        "is_full_suite": False,
    }
    if new_failures:
        decision["floor_output_tail"] = repaired_floor.output[-2000:]
        return _reject(f"regression floor introduced {len(new_failures)} new "
                       f"failure(s): {new_failures[:10]}")

    decision["decision"] = "deploy"
    decision["reason"] = "visible+held-out pass, surface frozen, regression floor green"
    return CodeGateResult(deploy=True, reason=decision["reason"], decision=decision)


def _base_source(env: WorktreeEnv, tool_relpath: str) -> str:
    """The pre-repair source of ``tool_relpath`` from the worktree's HEAD.

    The repaired source is on disk, so the freeze must compare against the
    committed base, not the working tree. A git-show failure is *not* treated as
    "empty baseline": that would silently zero out both the surface-freeze and
    the diff-shape guard (an empty base reads every public name as a benign
    addition and skips the retain floor), deploying a surface-drifting repair on
    a git hiccup. So we only return "" when the path provably does not exist at
    HEAD (a genuinely new file, additions-only); any other failure is a hard
    error the caller turns into a reject.
    """
    import subprocess

    try:
        show = subprocess.run(
            ["git", "show", f"HEAD:{tool_relpath}"],
            cwd=str(env.worktree), capture_output=True, text=True, timeout=60,
        )
        if show.returncode == 0:
            return show.stdout
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{tool_relpath}"],
            cwd=str(env.worktree), capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise CodeGateError(f"git failed deriving base source of {tool_relpath}: {exc}")
    if exists.returncode != 0:
        return ""  # path absent at HEAD — genuinely new, additions-only is correct
    raise CodeGateError(
        f"could not read base source of {tool_relpath} from HEAD "
        f"(it exists at HEAD but git show failed: {show.stderr.strip()})"
    )


def _baseline_diff_floor(
    env: WorktreeEnv, tool_relpath: str, repaired: str, base_src: str,
    floor_paths: tuple[str, ...],
) -> tuple[list[str], dict, str]:
    """Run the regression floor on repaired vs base source; return the failures
    the repair INTRODUCED (not pre-existing), a guard dict, and the repaired
    output tail. Shared by the campaign's oracle gate; the held-out gate keeps
    its own inline copy so it stays byte-identical."""
    repaired_floor = env.run_test(*floor_paths, extra_args=["--tb=no"], full_output=True)
    if repaired_floor.exit_code == _PYTEST_NO_TESTS_COLLECTED:
        return (["<no tests collected>"],
                {"ran": list(floor_paths), "collected": 0}, repaired_floor.output[-2000:])
    repaired_failures = _parse_pytest_failures(repaired_floor.output)
    if base_src == "":
        base_failures: set[str] = set()
    else:
        env.write_tool(tool_relpath, base_src)
        try:
            base_floor = env.run_test(*floor_paths, extra_args=["--tb=no"], full_output=True)
        finally:
            env.write_tool(tool_relpath, repaired)
        base_failures = _parse_pytest_failures(base_floor.output)
    new_failures = sorted(repaired_failures - base_failures)
    guard = {
        "ran": list(floor_paths),
        "new_failures": new_failures,
        "base_failure_count": len(base_failures),
        "repaired_failure_count": len(repaired_failures),
        "duration_seconds": round(repaired_floor.duration_seconds, 2),
        "is_full_suite": False,
    }
    return new_failures, guard, repaired_floor.output[-2000:]


def run_code_oracle_gate(
    env: WorktreeEnv,
    *,
    tool_relpath: str,
    test_relpath: str,
    bug_tests: tuple[str, ...],
    oracle_failures: frozenset[str],
    base_src: str,
    repair_result: RepairResult,
    pass_to_pass: tuple[str, ...] = (),
    floor_paths: Optional[tuple[str, ...]] = None,
    min_retain_ratio: float = DEFAULT_MIN_RETAIN_RATIO,
    run_inputs: Optional[dict] = None,
) -> CodeGateResult:
    """Oracle-based correctness verdict for the measurement campaign.

    Unlike :func:`run_code_gate` (held-out split — for a future novel-bug
    product), this verifies a repair against the upstream-fix ORACLE that every
    harvested historical bug carries. The worktree is at ``fix_sha`` (so the
    fix-commit test file with all bug-catching tests is present) with the buggy
    parent tool written in; ``base_src`` is that buggy parent source (the repair's
    starting point) and ``oracle_failures`` is the set of node-ids the upstream
    fix itself fails on the full test file (env-flaky tests that cancel out).

    A repair is CORRECT iff: surface frozen + file-scope clean + it passes the
    bug-catching tests + it introduces no failure the upstream fix does not also
    have (matches the oracle across the full fix-commit test file). The oracle
    match over the tool's whole test file IS the regression check for a
    measurement run, so the broad ``tests/tools`` cross-tool floor is OFF by
    default (it is ~minutes/run and a product-deploy concern, not a
    correctness-of-re-derivation one); pass ``floor_paths`` to enable it.

    Known limitation (honest): the oracle test-match catches a repair that breaks
    behavior the upstream fix preserves, but NOT pure input-hardcoding of the bug
    tests — defending against that needs a fuzzed differential vs the oracle (the
    deferred L3). An honest repair proposer (not adversarial) makes hardcoding
    unlikely, which is why test-match is an adequate correctness proxy here.
    """
    guards: dict = {}
    decision: dict = {
        "schema_version": GATE_SCHEMA_VERSION,
        "artifact_type": "code",
        "decision_signal": "oracle_match",
        "target_tool": tool_relpath,
        "test": test_relpath,
        "bug_tests": list(bug_tests),
        "oracle_failure_count": len(oracle_failures),
        "pass_to_pass_count": len(pass_to_pass),
        "repair": {
            "fixed": repair_result.fixed,
            "fixed_round": repair_result.fixed_round,
            "rounds_used": len(repair_result.rounds),
        },
        "guards": guards,
        "run_inputs": run_inputs or {},
    }

    def _reject(reason: str) -> CodeGateResult:
        decision["decision"] = "incorrect"
        decision["reason"] = reason
        return CodeGateResult(deploy=False, reason=reason, decision=decision)

    if not repair_result.fixed or repair_result.final_source is None:
        return _reject("repair did not produce a fix that passes the bug tests")
    repaired = repair_result.final_source

    # surface freeze + blast radius (against the buggy parent the repair started from)
    violations = freeze_violations(base_src, repaired, min_retain_ratio=min_retain_ratio)
    guards["freeze_ok"] = not violations
    guards["freeze_violations"] = violations
    if violations:
        return _reject("freeze/diff-shape violation: " + "; ".join(violations))

    # file scope — only the target tool may change; no test file touched
    changed = [_test_relnorm(p) for p in env.changed_files()]
    guards["changed_files"] = changed
    target = _test_relnorm(tool_relpath)
    offenders = [p for p in changed if p != target]
    test_touched = [p for p in changed if _is_test_path(p)]
    guards["file_scope_ok"] = not offenders and not test_touched
    if test_touched:
        return _reject(f"a test file was modified: {test_touched}")
    if offenders:
        return _reject(f"files other than the target tool changed: {offenders}")

    # the bug is actually fixed
    bug_fail = env.failing_tests(*bug_tests)
    guards["bug_tests_passed"] = not bug_fail
    if bug_fail:
        return _reject(f"repair does not pass the bug tests: {sorted(bug_fail)[:6]}")

    # oracle match — the repair introduces no failure the upstream fix doesn't
    # also have, across the FULL fix-commit test file (catches a fix that passes
    # the bug tests but breaks other behavior the upstream fix preserves).
    # pass_to_pass narrows the scope to specific node-ids (SWE-bench path); when
    # empty the full test file is the scope (Hermes path, byte-identical to before).
    oracle_scope = pass_to_pass or (test_relpath,)
    new_vs_oracle = sorted(env.failing_tests(*oracle_scope) - set(oracle_failures))
    guards["oracle_match"] = {"new_vs_oracle": new_vs_oracle,
                              "oracle_failure_count": len(oracle_failures),
                              "oracle_scope_size": len(oracle_scope)}
    if new_vs_oracle:
        return _reject(f"repair fails {len(new_vs_oracle)} test(s) the upstream fix "
                       f"passes: {new_vs_oracle[:6]}")

    # optional broad cross-tool regression floor (off by default — oracle-match
    # above is the regression check for a measurement run)
    if floor_paths:
        new_failures, floor_guard, tail = _baseline_diff_floor(
            env, tool_relpath, repaired, base_src, floor_paths)
        guards["floor"] = floor_guard
        if new_failures:
            decision["floor_output_tail"] = tail
            return _reject(f"regression floor introduced {len(new_failures)} new "
                           f"failure(s): {new_failures[:10]}")
    else:
        guards["floor"] = None

    decision["decision"] = "correct"
    decision["reason"] = "bug fixed, matches upstream-fix oracle, surface frozen, floor green"
    return CodeGateResult(deploy=True, reason=decision["reason"], decision=decision)
