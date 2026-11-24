"""Shared optimizer construction and constraint reporting."""

import dspy
from rich.console import Console

from evolution.core.lm import make_lm


def build_gepa(metric, iterations: int, optimizer_model: str) -> dspy.GEPA:
    """GEPA with the pipeline's standard budget policy and reflection profile."""
    return dspy.GEPA(
        metric=metric,
        auto="light" if iterations <= 5 else "medium",
        num_threads=8,
        reflection_lm=make_lm(optimizer_model, reflection=True),
    )


def print_constraints(console: Console, results) -> bool:
    """Print constraint results; return True when all passed."""
    all_pass = True
    for c in results:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False
    return all_pass
