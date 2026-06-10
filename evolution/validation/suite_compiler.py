"""Suite-constraint compiler: render a zero-LM "floor" prompt from a suite.

Stage 0 found that on the project's best headroom case (SKILLS_GUIDANCE), a
flat imperative restating the suite's *generic* constraints recovers ~79-85%
of what hand-tuned/evolved prompting achieves — the residual is within the
measured noise floor. So a mechanically-compiled floor is a useful artifact in
its own right, and a strong pre-flight signal: if baseline + floor already
nears the ceiling, GEPA's search spend is unlikely to be justified.

The compiler renders only constraint fields that are GENERIC across the suite
(tool names, the patch action, repo command conventions) — never per-instance
eval specifics (``target_skill`` / ``stale_token`` / fixture content), which
would be eval leakage. ``assert_no_holdout_leakage`` enforces that on any seed.
"""
from __future__ import annotations

from typing import Iterable

from evolution.validation.task import Task


class HoldoutLeakageError(RuntimeError):
    """A compiled floor contains a holdout task's per-instance eval specifics."""


def compile_suite_floor(tasks: Iterable[Task]) -> str:
    """Render generic constraint clauses from ``tasks`` as an imperative section.

    Deterministic and order-independent (clauses are built from sorted sets).
    Returns "" for a suite with no compilable constraints.
    """
    tasks = list(tasks)
    clauses: list[str] = []

    # Patch discipline: tasks whose verdict is "proactively patch the stale
    # skill you're using." Render the tool + action, never which skill/token.
    patch_tools = sorted(
        {t for task in tasks if task.expected_action == "patch" for t in task.expected_tools}
    )
    if patch_tools:
        tool = " or ".join(patch_tools)
        clauses.append(
            f"When you use a skill and its instructions are stale or incorrect, "
            f"call {tool} with action='patch' to fix it. Do not patch skills "
            f"that are already correct."
        )

    # Convention: required wrappers vs forbidden defaults. These substrings are
    # the repo's generic conventions (the rule under test), not per-row secrets.
    conv_pairs = sorted(
        {
            (tuple(task.required_cmd_substr), tuple(task.forbidden_cmd_substr))
            for task in tasks
            if task.expected_action == "convention" and task.required_cmd_substr
        }
    )
    for required, forbidden in conv_pairs:
        req = ", ".join(f"`{r}`" for r in required)
        clause = f"Use {req} for the corresponding action"
        if forbidden:
            forb = ", ".join(f"`{f}`" for f in forbidden)
            clause += f", not {forb}"
        clauses.append(clause + ".")

    # Over-eagerness guard: a tool that is expected somewhere but forbidden in a
    # control task is a discipline the suite teaches ("use it, but only when
    # needed"). A tool only ever forbidden (never expected) isn't being taught.
    expected_anywhere = {t for task in tasks for t in task.expected_tools}
    forbidden_anywhere = {t for task in tasks for t in task.forbidden_tools}
    guarded = sorted(expected_anywhere & forbidden_anywhere)
    if guarded:
        tool = " or ".join(guarded)
        clauses.append(
            f"Only call {tool} when it is actually needed; do not call it for "
            f"tasks that do not require it."
        )

    return "\n".join(clauses)


def assert_no_holdout_leakage(floor_text: str, holdout_tasks: Iterable[Task]) -> None:
    """Raise if ``floor_text`` contains any holdout task's per-instance specifics.

    The compiler never emits these by construction; this guards seeds that are
    hand-edited or compiled from a split that wasn't strictly train-only.
    """
    leaks: list[str] = []
    for task in holdout_tasks:
        for secret in (task.target_skill, task.stale_token):
            if secret and secret in floor_text:
                leaks.append(secret)
    if leaks:
        raise HoldoutLeakageError(
            f"compiled floor leaks holdout eval specifics: {sorted(set(leaks))}"
        )


def main(argv: list[str] | None = None) -> int:
    """Render the compiled floor for a suite to stdout (inspect/deploy)."""
    import argparse
    from pathlib import Path

    from evolution.validation.task import TaskSuite

    parser = argparse.ArgumentParser(
        description="Render a zero-LM constraint floor from a closed-loop suite."
    )
    parser.add_argument("--tasks", type=Path, required=True, help="Suite JSONL path.")
    args = parser.parse_args(argv)

    suite = TaskSuite.from_jsonl(args.tasks)
    floor = compile_suite_floor(suite.tasks)
    assert_no_holdout_leakage(floor, suite.tasks)
    if not floor:
        print("(no compilable constraints in this suite)")
        return 0
    print(floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
