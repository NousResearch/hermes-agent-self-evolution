"""Shared optimizer construction and constraint reporting."""

import dspy
from rich.console import Console

from evolution.core.lm import make_lm


def build_gepa(
    metric,
    iterations: int,
    optimizer_model: str,
    max_instruction_chars: int | None = None,
    proposer_guidance: str | None = None,
) -> dspy.GEPA:
    """GEPA with the pipeline's standard budget policy and reflection profile.

    max_instruction_chars caps each proposed instruction (enforced by the
    proposer's validation); proposer_guidance is appended to every proposal
    prompt (e.g. "preserve the markers verbatim").
    """
    proposer = None
    if max_instruction_chars or proposer_guidance:
        from dspy.teleprompt.gepa.instruction_proposal import InstructionProposer

        proposer = InstructionProposer(
            max_chars=max_instruction_chars,
            additional_instructions=proposer_guidance,
        )
    return dspy.GEPA(
        metric=metric,
        auto="light" if iterations <= 5 else "medium",
        num_threads=8,
        reflection_lm=make_lm(optimizer_model, reflection=True),
        instruction_proposer=proposer,
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
