"""Sanity tests for the advanced systematic-debugging planted-bug suite.

Same shape as test_systematic_debugging_suite.py — proves every planted
bug is real (baseline test exits non-zero) and every known fix passes
(test exits zero after applying it). Without these guards, a too-permissive
assertion would silently invalidate every closed-loop run that uses this
suite.

The advanced suite exists for closed-loop runs where the textbook suite
saturates against a capable agent model — these bugs require structured
debugging (the spec edge case the obvious patch misses) and a "just try
edits" methodology fails on them.
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
    / "systematic_debugging_advanced.jsonl"
)


# Known-good fix per task_id. Each value replaces solution.py wholesale.
_KNOWN_FIXES: dict[str, str] = {
    "debug_generator_exhaustion": (
        "def count_partitions(items, predicate):\n"
        "    items = list(items)\n"
        "    n_true = sum(1 for x in items if predicate(x))\n"
        "    n_false = sum(1 for x in items if not predicate(x))\n"
        "    return (n_true, n_false)\n"
    ),
    "debug_shared_mutable_return": (
        "_history_cache = {}\n"
        "\n"
        "def get_history(user_id):\n"
        "    if user_id not in _history_cache:\n"
        "        _history_cache[user_id] = []\n"
        "    return list(_history_cache[user_id])\n"
    ),
    "debug_float_precision_equality": (
        "import math\n"
        "\n"
        "def is_target_amount(payments, target):\n"
        "    return math.isclose(sum(payments), target, abs_tol=1e-6)\n"
    ),
    "debug_binary_search_boundary": (
        "def leftmost_insert(sorted_arr, target):\n"
        "    lo, hi = 0, len(sorted_arr)\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if sorted_arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid\n"
        "    return lo\n"
    ),
    "debug_class_vs_instance_attribute": (
        "class TokenBucket:\n"
        "    def __init__(self, capacity):\n"
        "        self.capacity = capacity\n"
        "        self.consumed = []\n"
        "\n"
        "    def take(self, n):\n"
        "        if sum(self.consumed) + n > self.capacity:\n"
        "            raise ValueError('over capacity')\n"
        "        self.consumed.append(n)\n"
        "\n"
        "    def used(self):\n"
        "        return sum(self.consumed)\n"
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
        suite_ids = {t.task_id for t in suite.tasks}
        fix_ids = set(_KNOWN_FIXES.keys())
        assert suite_ids == fix_ids, (
            f"task ids and known-fix ids out of sync: "
            f"missing fixes={suite_ids - fix_ids}, "
            f"extra fixes={fix_ids - suite_ids}"
        )

    def test_no_task_id_collision_with_basic_suite(self, suite):
        # Sanity guard: basic + advanced suites are separate files but
        # behavioral examples key by task_id. Collisions would silently
        # overwrite verdicts in the cache. Confirm zero overlap.
        basic_path = SUITE_PATH.parent / "systematic_debugging.jsonl"
        basic = TaskSuite.from_jsonl(basic_path)
        basic_ids = {t.task_id for t in basic.tasks}
        advanced_ids = {t.task_id for t in suite.tasks}
        assert not (basic_ids & advanced_ids), (
            f"task_id collision between suites: {basic_ids & advanced_ids}"
        )


class TestPlantedBugsAreReal:
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
    @pytest.mark.parametrize(
        "task_id,fixed_solution",
        list(_KNOWN_FIXES.items()),
        ids=list(_KNOWN_FIXES.keys()),
    )
    def test_known_fix_passes(self, task_id, fixed_solution, tmp_path, suite):
        task = next(t for t in suite.tasks if t.task_id == task_id)
        _materialize(task.fixture_setup, tmp_path)
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
