"""Build behavioral ``dspy.Example``s from a closed-loop ``TaskSuite``.

The metric, when called on a behavioral example, scores the current candidate
text via the closed-loop validator (instead of the LM judge). Those scores
contribute to GEPA's ``sum(minibatch_scores)`` acceptance rule, so a candidate
that passes a behavioral task its predecessor failed can break a judge tie
and get accepted.

The example carries a ``closed_loop_task_id`` marker the metric routes on,
plus a placeholder ``task`` value (the suite's ``user_message``) that
``ToolModule.forward`` skips past on the behavioral branch. ``task`` and
``closed_loop_task_id`` are both marked as input keys so DSPy passes them
to ``forward()`` via ``program(**example.inputs())``.
"""

from __future__ import annotations

import dspy

from evolution.validation.task import TaskSuite


def build_behavioral_examples(suite: TaskSuite) -> list[dspy.Example]:
    """One example per task in ``suite``, stable order by ``task_id``.

    The placeholder ``task`` value carries the original ``user_message`` for
    debuggability; it isn't consumed by the behavioral metric branch.
    """
    examples = [
        dspy.Example(
            task=task.user_message,
            closed_loop_task_id=task.task_id,
        ).with_inputs("task", "closed_loop_task_id")
        for task in sorted(suite.tasks, key=lambda t: t.task_id)
    ]
    return examples
