"""Sanity tests for the systematic-debugging planted-bug suite.

The suite at ``evolution/validation/suites/systematic_debugging.jsonl``
is the substrate for skill-side closed-loop validation. These tests
prove the suite itself is well-formed and that every task's planted
bug + test pair behaves as designed (test fails on baseline, passes
when the documented fix is applied) — without these guards a typo in
the test file or a too-permissive assertion would silently invalidate
every closed-loop run.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from evolution.validation.task import TaskSuite


SUITE_PATH = (
    Path(__file__).resolve().parents[2]
    / "evolution"
    / "validation"
    / "suites"
    / "systematic_debugging.jsonl"
)


# Known-good fix per task_id. Each value replaces solution.py wholesale
# after the buggy fixture is materialized. The test then re-runs
# test_solution.py to confirm the fix actually passes the asserts.
_KNOWN_FIXES: dict[str, str] = {
    "debug_off_by_one_range": (
        "def get_range_inclusive(n):\n"
        "    return list(range(1, n + 1))\n"
    ),
    "debug_inverted_condition": (
        "def first_positive(values):\n"
        "    for x in values:\n"
        "        if x > 0:\n"
        "            return x\n"
        "    return None\n"
    ),
    "debug_missing_base_case": (
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
    ),
    "debug_wrong_operator": (
        "def square(x):\n"
        "    return x * x\n"
    ),
    "debug_mutable_default_arg": (
        "def append_to_new(item, items=None):\n"
        "    if items is None:\n"
        "        items = []\n"
        "    items.append(item)\n"
        "    return items\n"
    ),
}


@pytest.fixture(scope="module")
def suite() -> TaskSuite:
    return TaskSuite.from_jsonl(SUITE_PATH)


class TestSuiteShape:
    def test_loads_five_tasks(self, suite):
        assert len(suite.tasks) == 5

    def test_every_task_has_test_command(self, suite):
        for task in suite.tasks:
            assert task.test_command, f"{task.task_id}: missing test_command"

    def test_every_task_has_solution_and_test_files(self, suite):
        for task in suite.tasks:
            files = set(task.fixture_setup.keys())
            assert "solution.py" in files, f"{task.task_id}: no solution.py"
            assert "test_solution.py" in files, f"{task.task_id}: no test_solution.py"

    def test_every_task_id_has_a_known_fix(self, suite):
        # Future-proofing: when adding tasks, the contributor must also
        # add an entry to _KNOWN_FIXES so the round-trip tests below cover it.
        suite_ids = {t.task_id for t in suite.tasks}
        fix_ids = set(_KNOWN_FIXES.keys())
        assert suite_ids == fix_ids, (
            f"task ids and known-fix ids out of sync: "
            f"missing fixes={suite_ids - fix_ids}, "
            f"extra fixes={fix_ids - suite_ids}"
        )

    def test_test_command_is_runnable_form(self, suite):
        # We use shlex.split (no shell) on the test_command. Each command
        # should split cleanly without quoting tricks.
        for task in suite.tasks:
            parts = shlex.split(task.test_command)
            assert len(parts) >= 2, f"{task.task_id}: test_command too short"


class TestPlantedBugsAreReal:
    """For every task: materializing the baseline buggy fixture and running
    its test must exit non-zero. If this fails, the planted bug is no bug
    at all — the suite isn't measuring what it claims to measure."""

    @pytest.mark.parametrize(
        "task_id",
        list(_KNOWN_FIXES.keys()),
        ids=list(_KNOWN_FIXES.keys()),
    )
    def test_baseline_buggy_code_fails(self, task_id, tmp_path, suite):
        task = next(t for t in suite.tasks if t.task_id == task_id)
        _materialize(task.fixture_setup, tmp_path)
        result = subprocess.run(
            shlex.split(task.test_command),
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode != 0, (
            f"{task_id}: baseline buggy code unexpectedly passed the test.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestKnownFixesPass:
    """For every task: applying the known fix and re-running the test must
    exit zero. If this fails, the test is pathologically broken (asserts
    the wrong thing, has a syntax error, etc.) and would never accept
    any candidate fix — including the right one."""

    @pytest.mark.parametrize(
        "task_id,fixed_solution",
        list(_KNOWN_FIXES.items()),
        ids=list(_KNOWN_FIXES.keys()),
    )
    def test_known_fix_passes(self, task_id, fixed_solution, tmp_path, suite):
        task = next(t for t in suite.tasks if t.task_id == task_id)
        _materialize(task.fixture_setup, tmp_path)
        # Apply the known fix.
        (tmp_path / "solution.py").write_text(fixed_solution)
        result = subprocess.run(
            shlex.split(task.test_command),
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"{task_id}: known fix did not make the test pass.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def _materialize(setup: dict[str, str], target: Path) -> None:
    for rel, content in setup.items():
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
