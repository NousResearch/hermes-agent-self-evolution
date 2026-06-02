"""GEPA instruction_proposer for prompt-section evolution.

Mirrors ``BudgetAwareToolProposer``: subclasses ``BudgetAwareProposer`` for
the budget-tracking infrastructure but installs a prompt-section reflection
template whose hard constraint is sentinel preservation. ``__call__`` runs the
inherited proposer LM, then passes the candidate through ``extract_and_rebuild``
so only the sentinel-delimited region survives.

On ``SentinelParseError`` the call re-raises (after incrementing
``sentinel_failures``) rather than returning the parent unchanged — GEPA's
reflective_mutation path skips the iteration, avoiding a phantom
identical-to-parent candidate that would pollute the selection pool.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import dspy

from evolution.prompts.prompt_module import (
    SentinelParseError,
    _extract_from_sentinels,
    _render_instructions,
)
from evolution.skills.budget_aware_proposer import BudgetAwareProposer

logger = logging.getLogger(__name__)


_PROMPT_PROPOSER_TEMPLATE = """\
You are revising one section ({section_name}) of an agent's system prompt.
The instruction below wraps the current candidate text between the markers
`<!-- SECTION:{section_name} -->` and `<!-- /SECTION:{section_name} -->`.

Hard constraint - sentinel preservation:
Modify only the text between those two markers. Do not change the markers
themselves, and do not add any text outside them.

Length budget: at most {target_chars} characters for the section body (between
the markers). The current body is {baseline_chars} characters.

Hard constraint - grounding citation:
Every change must quote or paraphrase a specific phrase from the feedback. If a
failure is not actionable from the section text (model error, judge
disagreement, out-of-distribution input), skip it.

Your task: rewrite the current section to fix the failures shown below,
modifying only the sentinel-delimited region for {section_name}.

Steps:
1. Read each failure in the feedback. Classify it as (a) the agent misapplied
   existing guidance -> refine the wording, (b) the agent lacked guidance it
   needed -> add it, or (c) not actionable from the section text -> skip.
2. Apply changes only for (a) and (b), only inside the sentinel region.
3. For each change, name the specific feedback phrase that grounded it.
4. Match the voice and density of the existing section.
5. If more additions are warranted than fit within {target_chars}, address the
   most-grounded failures first; GEPA will run again with the updated baseline.

If the feedback below is empty or contains no concrete failures, return the
current instruction unchanged.

Output the full instruction text (markers included, only the sentinel-delimited
region modified). No preamble, no markdown fences, no explanation.
"""


def extract_and_rebuild(candidate: str, section_name: str) -> str:
    """Extract the sentinel region from a candidate full-instructions string
    and re-render the instructions around it.

    Pure function — testable without LM mocks. Raises ``SentinelParseError``
    if the candidate didn't preserve the sentinels.
    """
    new_body = _extract_from_sentinels(candidate, section_name)
    return _render_instructions(section_name, new_body)


class _PromptProposalSignature(dspy.Signature):
    """Placeholder; overwritten per-instance via with_instructions so the
    section-specific template (section_name, target_chars, baseline_chars
    baked in) is installed."""

    current_instruction: str = dspy.InputField(
        desc="The current instruction with the sentinel-wrapped section body"
    )
    examples_with_feedback: str = dspy.InputField(
        desc="Failure feedback from the eval to ground refinements in"
    )
    improved_instruction: str = dspy.OutputField(
        desc="The revised instruction with only the sentinel region modified"
    )


class PromptSectionProposer(BudgetAwareProposer):
    """GEPA-compatible ProposalFn for prompt-section evolution."""

    component_name = "passthrough.predict"

    def __init__(
        self,
        section_name: str,
        baseline_chars: int,
        max_growth: float = 0.2,
        safety_margin: float = 0.10,
    ):
        super().__init__(
            baseline_chars=baseline_chars,
            max_growth=max_growth,
            safety_margin=safety_margin,
        )
        self.section_name = section_name
        self.sentinel_failures = 0

        template = _PROMPT_PROPOSER_TEMPLATE.format(
            section_name=section_name,
            target_chars=self.target_chars,
            baseline_chars=baseline_chars,
        )
        self.propose = dspy.Predict(
            _PromptProposalSignature.with_instructions(template)
        )

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.component_name not in components_to_update:
            return {}
        if self.component_name not in candidate:
            return {}

        current_instruction = candidate[self.component_name]
        feedback = self._format_examples(
            reflective_dataset.get(self.component_name, [])
        )
        prediction = self.propose(
            current_instruction=current_instruction,
            examples_with_feedback=feedback,
        )
        new_candidate = prediction.improved_instruction

        try:
            rebuilt = extract_and_rebuild(new_candidate, self.section_name)
        except SentinelParseError as exc:
            self.sentinel_failures += 1
            excerpt = new_candidate[:200] + ("..." if len(new_candidate) > 200 else "")
            logger.warning(
                "PromptSectionProposer: sentinel parse failure (#%d) for %r: %s. "
                "Candidate excerpt: %r",
                self.sentinel_failures, self.section_name, exc, excerpt,
            )
            raise

        return {self.component_name: rebuilt}
