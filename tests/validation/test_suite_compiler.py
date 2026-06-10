"""Suite-constraint compiler: render a zero-LM 'floor' prompt from a suite's
generic constraints, never per-instance eval specifics."""
import pytest

from evolution.validation.suite_compiler import (
    HoldoutLeakageError,
    assert_no_holdout_leakage,
    compile_suite_floor,
)
from evolution.validation.task import Task


def _patch_task(task_id, skill, stale):
    return Task(
        task_id=task_id, user_message="m", expected_tools=("skill_manage",),
        expected_action="patch", target_skill=skill, stale_token=stale,
    )


def _conv_task(task_id, required, forbidden):
    return Task(
        task_id=task_id, user_message="m", expected_action="convention",
        required_cmd_substr=required, forbidden_cmd_substr=forbidden,
    )


def _control_task(task_id, forbidden_tools=("skill_manage",)):
    return Task(task_id=task_id, user_message="m", forbidden_tools=forbidden_tools)


def test_patch_clause_renders_tool_and_action():
    floor = compile_suite_floor([
        _patch_task("a", "line-counter", "wc --lines"),
        _patch_task("b", "top-lines", "head --count"),
    ])
    assert "skill_manage" in floor
    assert "action='patch'" in floor or "patch" in floor
    # generic rule only — never the per-task secrets
    assert "line-counter" not in floor and "wc --lines" not in floor
    assert "top-lines" not in floor and "head --count" not in floor


def test_convention_clause_renders_wrappers_not_secrets():
    floor = compile_suite_floor([
        _conv_task("c", required=("bin/check",), forbidden=("pytest",)),
    ])
    assert "bin/check" in floor  # the repo convention is generic, not a secret
    assert "pytest" in floor


def test_over_eagerness_clause_when_tool_both_expected_and_forbidden():
    # skill_manage is expected in patch tasks but forbidden in controls →
    # the floor should warn against over-calling it.
    floor = compile_suite_floor([
        _patch_task("a", "line-counter", "wc --lines"),
        _control_task("ctl"),
    ])
    low = floor.lower()
    assert "only call" in low or "do not call" in low or "not needed" in low


def test_no_clause_for_forbidden_tool_never_expected():
    # A tool forbidden in a control but never expected anywhere is not a
    # discipline the suite is teaching → no over-eagerness clause for it.
    floor = compile_suite_floor([_control_task("ctl", forbidden_tools=("rm",))])
    assert "rm" not in floor


def test_empty_suite_renders_empty():
    assert compile_suite_floor([]) == ""


def test_deterministic_and_dedup():
    tasks = [
        _patch_task("a", "s1", "t1"), _patch_task("b", "s2", "t2"),
        _conv_task("c", ("bin/check",), ("pytest",)),
        _conv_task("d", ("bin/check",), ("pytest",)),  # dup convention
    ]
    out1 = compile_suite_floor(tasks)
    out2 = compile_suite_floor(list(reversed(tasks)))
    assert out1 == out2  # order-independent
    assert out1.count("skill_manage") == 1  # patch clause once
    assert out1.count("bin/check") == 1  # convention deduped


def test_leakage_guard_passes_on_generic_floor():
    holdout = [_patch_task("h", "secret-skill", "secret.token")]
    floor = compile_suite_floor([_patch_task("a", "line-counter", "wc --lines")])
    assert_no_holdout_leakage(floor, holdout)  # no raise


def test_leakage_guard_raises_when_secret_present():
    holdout = [_patch_task("h", "secret-skill", "secret.token")]
    with pytest.raises(HoldoutLeakageError, match="secret"):
        assert_no_holdout_leakage("... call skill_manage on secret-skill ...", holdout)
