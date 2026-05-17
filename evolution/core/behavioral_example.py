"""Build behavioral ``dspy.Example``s from a closed-loop ``TaskSuite``.

The metric, when called on a behavioral example, scores the current candidate
text via the closed-loop validator (instead of the LM judge). Those scores
contribute to GEPA's ``sum(minibatch_scores)`` acceptance rule, so a candidate
that passes a behavioral task its predecessor failed can break a judge tie
and get accepted.

The example carries a ``closed_loop_task_id`` marker the metric routes on,
plus a placeholder task-input value (the suite's ``user_message``) that the
module's ``forward()`` skips past on the behavioral branch. Both fields are
marked as input keys so DSPy passes them via ``program(**example.inputs())``.

``task_field`` parameterizes the input field name to match the host module's
forward signature: ``ToolModule.forward(task=...)`` uses ``"task"`` (the
default); ``SkillModule.forward(task_input=...)`` passes ``"task_input"``.
"""

from __future__ import annotations

import dspy

from evolution.validation.task import TaskSuite


def build_behavioral_examples(
    suite: TaskSuite, *, task_field: str = "task"
) -> list[dspy.Example]:
    """One example per task in ``suite``, stable order by ``task_id``."""
    examples = [
        dspy.Example(
            **{task_field: task.user_message},
            closed_loop_task_id=task.task_id,
        ).with_inputs(task_field, "closed_loop_task_id")
        for task in sorted(suite.tasks, key=lambda t: t.task_id)
    ]
    return examples
