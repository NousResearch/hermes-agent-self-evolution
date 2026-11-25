"""Shared optimizer construction and constraint reporting."""

from collections.abc import Mapping, Sequence
from typing import Any

import dspy
from rich.console import Console

from evolution.core.lm import make_lm


class GuardedProposer:
    """Rejects proposals missing required markers by keeping the parent instruction."""

    def __init__(self, required_markers: tuple[str, ...], **proposer_kwargs):
        from dspy.teleprompt.gepa.instruction_proposal import InstructionProposer

        self._inner = InstructionProposer(**proposer_kwargs)
        self._markers = required_markers

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposed = self._inner(candidate, reflective_dataset, components_to_update)
        for name, text in proposed.items():
            if not all(m in text for m in self._markers):
                proposed[name] = candidate[name]
        return proposed


def build_gepa(
    metric,
    iterations: int,
    optimizer_model: str,
    max_instruction_chars: int | None = None,
    proposer_guidance: str | None = None,
    marker_guard: tuple[str, ...] | None = None,
) -> dspy.GEPA:
    """GEPA with the pipeline's standard budget policy and reflection profile.

    max_instruction_chars caps each proposed instruction (enforced by the
    proposer's validation); proposer_guidance is appended to every proposal
    prompt (e.g. "preserve the markers verbatim").
    """
    proposer = None
    if marker_guard:
        proposer = GuardedProposer(
            marker_guard,
            max_chars=max_instruction_chars,
            additional_instructions=proposer_guidance,
        )
    elif max_instruction_chars or proposer_guidance:
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
