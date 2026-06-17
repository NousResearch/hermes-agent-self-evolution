"""Tests for the Phase 4 verifier/fitness harness."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from evolution.code.verifier_harness import (
    CheckResult,
    CodeEvalTask,
    CodeFitnessHarness,
    ToolActionSession,
)

BUGGY_MATH_UTILS = """def add(a, b):
    return a - b

def add_all(values):
    total = 0
    for value in values:
        total = add(total, value)
    return total
"""


TRAIN_ONLY_MATH_UTILS = """def add(a, b):
    if a == 2 and b == 3:
        return 5
    return a - b

def add_all(values):
    total = 0
    for value in values:
        total = add(total, value)
    return total
"""


def _load_math_utils(workspace: Path):
    path = workspace / "repo" / "math_utils.py"
    shutil.rmtree(path.parent / "__pycache__", ignore_errors=True)
    importlib.invalidate_caches()
    module_name = f"math_utils_{abs(hash(path))}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _visible_check(workspace: Path) -> CheckResult:
    try:
        mod = _load_math_utils(workspace)
        if mod.add(2, 3) == 5:
            return CheckResult.pass_("visible add case passed")
    except Exception as exc:
        return CheckResult.fail(str(exc), ("visible_exception",))
    return CheckResult.fail("visible add case failed", ("visible_add_failed",))


def _hidden_check(workspace: Path) -> CheckResult:
    try:
        mod = _load_math_utils(workspace)
        probes = [
            mod.add(-2, 5) == 3,
            mod.add_all([1, 2, 3]) == 6,
        ]
    except Exception as exc:
        return CheckResult.fail(str(exc), ("hidden_exception",))
    score = sum(1 for ok in probes if ok) / len(probes)
    if score == 1.0:
        return CheckResult.pass_("hidden repo repair passed")
    residues = tuple(
        f"hidden_probe_{i}_failed" for i, ok in enumerate(probes) if not ok
    )
    return CheckResult.fail("hidden repo repair failed", residues, score=score)


def _full_suite_passes(_: Path) -> CheckResult:
    return CheckResult.pass_("pytest gate passed")


def _full_suite_fails(_: Path) -> CheckResult:
    return CheckResult.fail("pytest gate failed", ("pytest_failed",))


def _repo_repair_task(*, full_suite=_full_suite_passes) -> CodeEvalTask:
    return CodeEvalTask(
        name="repo_repair",
        workspace_files={
            "repo/math_utils.py": BUGGY_MATH_UTILS,
            "visible_cases.json": json.dumps({"add": [[2, 3, 5]]}, sort_keys=True),
        },
        visible_check=_visible_check,
        hidden_check=_hidden_check,
        full_suite_check=full_suite,
    )


def _frozen_noop(session: ToolActionSession) -> None:
    session.read_file("repo/math_utils.py")


def _train_only_candidate(session: ToolActionSession) -> None:
    session.write_file("repo/math_utils.py", TRAIN_ONLY_MATH_UTILS)
    session.run_visible_check()


def _adaptive_candidate_with_rollback(session: ToolActionSession) -> None:
    original = session.read_file("repo/math_utils.py")
    assert session.apply_patch("repo/math_utils.py", "return a - b", "return a * b")
    if not session.run_visible_check().passed:
        session.revert_file("repo/math_utils.py", original)
    assert session.apply_patch("repo/math_utils.py", "return a - b", "return a + b")
    session.run_visible_check()


def test_hidden_gate_rejects_train_only_candidate(tmp_path):
    harness = CodeFitnessHarness()

    decision = harness.evaluate(
        _repo_repair_task(),
        adaptive_runner=_train_only_candidate,
        frozen_runner=_frozen_noop,
        candidate_id="train-only-visible-fit",
        root=tmp_path,
    )

    assert not decision.accepted
    assert decision.reason == "hidden_gate_failed"
    assert decision.adaptive.visible.passed
    assert not decision.adaptive.hidden.passed


def test_adaptive_patch_beats_frozen_and_records_rollback_trace(tmp_path):
    harness = CodeFitnessHarness()

    decision = harness.evaluate(
        _repo_repair_task(),
        adaptive_runner=_adaptive_candidate_with_rollback,
        frozen_runner=_frozen_noop,
        candidate_id="single-line-add-fix",
        root=tmp_path,
    )

    assert decision.accepted
    assert decision.reason == "accepted"
    assert decision.frozen.hidden.score < decision.adaptive.hidden.score
    assert decision.adaptive.full_suite.passed
    statuses = [(trace.action, trace.status) for trace in decision.adaptive.traces]
    assert ("revert_file", "reverted") in statuses
    assert statuses.count(("apply_patch", "committed")) == 2
    assert decision.adaptive.trace_digest


def test_rejection_cache_blocks_repeat_candidate(tmp_path):
    harness = CodeFitnessHarness()
    task = _repo_repair_task()

    first = harness.evaluate(
        task,
        adaptive_runner=_train_only_candidate,
        frozen_runner=_frozen_noop,
        candidate_id="same-bad-candidate",
        root=tmp_path / "first",
    )
    second = harness.evaluate(
        task,
        adaptive_runner=_train_only_candidate,
        frozen_runner=_frozen_noop,
        candidate_id="same-bad-candidate",
        root=tmp_path / "second",
    )

    assert not first.accepted
    assert second.cached
    assert second.reason == "rejected_cached"
    assert second.adaptive.skipped
    assert not list((tmp_path / "second").glob("**/*"))


def test_hidden_expectations_stay_out_of_workspace(tmp_path):
    hidden_canary = "SECRET_HIDDEN_CASE_ADD_NEGATIVE_AND_ADD_ALL"

    def hidden_check_with_canary(workspace: Path) -> CheckResult:
        assert hidden_canary
        return _hidden_check(workspace)

    task = _repo_repair_task()
    task = CodeEvalTask(
        name=task.name,
        workspace_files=task.workspace_files,
        visible_check=task.visible_check,
        hidden_check=hidden_check_with_canary,
        full_suite_check=task.full_suite_check,
        allowed_tools=task.allowed_tools,
    )

    workspace = task.provision(tmp_path)
    blob = ""
    for path in workspace.rglob("*"):
        if path.is_file():
            assert "hidden" not in path.name.lower()
            blob += path.read_text(encoding="utf-8")

    workspace_filenames = {path.name for path in workspace.rglob("*")}
    assert hidden_canary not in blob
    assert "visible_cases.json" in workspace_filenames


def test_full_suite_gate_blocks_hidden_success(tmp_path):
    harness = CodeFitnessHarness()

    decision = harness.evaluate(
        _repo_repair_task(full_suite=_full_suite_fails),
        adaptive_runner=_adaptive_candidate_with_rollback,
        frozen_runner=_frozen_noop,
        candidate_id="hidden-pass-regression-fail",
        root=tmp_path,
    )

    assert not decision.accepted
    assert decision.reason == "full_suite_failed"
    assert decision.adaptive.hidden.passed
    assert not decision.adaptive.full_suite.passed
