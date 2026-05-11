"""GEPA instruction_proposer variant for tool description evolution.

BudgetAwareToolProposer subclasses BudgetAwareProposer with a tool-specific
reflection-prompt template (sentinel-preservation rule, sentinel-preserving
BEFORE/AFTER one-shot, length budget framed against the description not the
full instructions). __call__ invokes the inherited self.propose, then passes
the candidate through the pure function extract_and_rebuild.

On SentinelParseError: log WARNING, increment self.sentinel_failures, and
re-raise so GEPA's reflective_mutation.py exception path skips the iteration
rather than admitting a phantom unchanged-candidate that would pollute the
knee-point candidate pool with duplicate-score entries.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import dspy

from evolution.skills.budget_aware_proposer import BudgetAwareProposer
from evolution.tools.tool_module import (
    _extract_description_from_sentinels,
    _render_manifest_for_prompt,
)
from evolution.tools.tool_source import SentinelParseError, ToolManifest

logger = logging.getLogger(__name__)


_TOOL_PROPOSER_TEMPLATE = """\
You are revising the description of one tool ({target_tool_name}) inside a multi-tool manifest.
The full instruction below contains many tool descriptions in markdown form.

Hard constraint - sentinel preservation:
Modify only the text between `<!-- TARGET:{target_tool_name} -->` and `<!-- /TARGET:{target_tool_name} -->`.
Do not change the markers themselves. Do not modify any other tool's description.
Do not reorder, add, or remove tools.

Length budget: at most {target_chars} characters for the description text (between sentinels). Each character above {target_chars} is wasted.
The current description is {baseline_chars} characters; you are revising it.

Hard constraint - grounding citation:
Every addition or refinement to the description must quote or paraphrase a specific phrase from the feedback. If you cannot point to such a phrase, do not change anything for that failure.

Example of the modification you should make (the only thing that changes is the sentinel-delimited region):

BEFORE:
## search_files
<!-- TARGET:search_files -->Find things.<!-- /TARGET:search_files -->

## grep_in_terminal
Run grep across files.

AFTER:
## search_files
<!-- TARGET:search_files -->Find files in the repository by name or path pattern. Returns matching paths.<!-- /TARGET:search_files -->

## grep_in_terminal
Run grep across files.

Your task: rewrite the current instruction to fix the failures shown below, modifying only the sentinel-delimited region for {target_tool_name}.

Steps:
1. Read each failure mode in the feedback. Classify each one as:
   (a) the assistant misapplied an existing description -> refine the wording, OR
   (b) the assistant lacked information it needed -> add it, OR
   (c) neither -- the failure is not actionable from instruction text (model error, judge disagreement, out-of-distribution input). Skip it.
2. Apply changes only for (a) and (b), only inside the sentinel-delimited region for {target_tool_name}.
3. For each change, name the specific feedback phrase that grounded it.
4. New content uses the same imperative, terse style as the rest of the description.
5. If the failures call for more additions than fit within {target_chars}, address the most-grounded failures first; leave the rest for the next iteration. GEPA will run again with the updated baseline.

If the feedback below is empty or contains no concrete failures, return the current instruction unchanged.

Output the full instruction text (with all tool entries, only the sentinel-delimited region for {target_tool_name} modified). No preamble, no markdown fences, no explanation.
"""


def extract_and_rebuild(
    candidate: str,
    manifest: ToolManifest,
    target_name: str,
) -> str:
    """Parse the sentinel-delimited region from a candidate full-instructions
    string, and re-render the full manifest with that description plugged into
    the target tool's slot.

    Pure function — testable without LM mocks. Raises SentinelParseError if the
    candidate doesn't preserve the sentinels.
    """
    new_description = _extract_description_from_sentinels(candidate, target_name)
    return _render_manifest_for_prompt(manifest, target_name, new_description)


class _ToolProposalSignature(dspy.Signature):
    """Placeholder docstring — overwritten per-instance via with_instructions
    in BudgetAwareToolProposer.__init__ so the tool-specific template (with
    target_tool_name, target_chars, baseline_chars baked in) is installed."""

    current_instruction: str = dspy.InputField(
        desc="The current full rendered manifest with sentinel-wrapped target description"
    )
    examples_with_feedback: str = dspy.InputField(
        desc="Failure feedback from the eval to ground refinements in"
    )
    improved_instruction: str = dspy.OutputField(
        desc="The revised full manifest with only the sentinel-delimited region modified"
    )


class BudgetAwareToolProposer(BudgetAwareProposer):
    """GEPA-compatible ProposalFn for tool description evolution.

    Subclasses BudgetAwareProposer to inherit budget-tracking infrastructure
    but installs a tool-specific reflection template (sentinel-preservation
    hard constraint, sentinel-preserving BEFORE/AFTER one-shot, length budget
    framed against the description length). The inherited compression
    template is overwritten — the tool variant needs different anti-
    hallucination guards (sentinel preservation) and a different budget
    framing (description length, not full-manifest length).

    On parse failure, __call__ re-raises SentinelParseError rather than
    returning baseline-unchanged. Returning baseline-unchanged would create a
    phantom candidate identical to the parent and pollute the knee-point pool
    with duplicate-score entries; GEPA's reflective_mutation.py catches the
    exception and skips the iteration instead.
    """

    component_name = "selector.predict"

    def __init__(
        self,
        target_tool_name: str,
        manifest: ToolManifest,
        target_description: str,
        baseline_chars: int,
        max_growth: float = 0.2,
        safety_margin: float = 0.10,
    ):
        # Parent computes self.target_chars from baseline_chars, max_growth,
        # safety_margin; we reuse that. Mode is irrelevant here because we
        # immediately overwrite self.propose with a tool-specific signature.
        super().__init__(
            baseline_chars=baseline_chars,
            max_growth=max_growth,
            safety_margin=safety_margin,
        )
        self.target_tool_name = target_tool_name
        self.manifest = manifest
        self.target_description = target_description
        self.sentinel_failures = 0

        tool_template = _TOOL_PROPOSER_TEMPLATE.format(
            target_tool_name=target_tool_name,
            target_chars=self.target_chars,
            baseline_chars=baseline_chars,
        )
        signature = _ToolProposalSignature.with_instructions(tool_template)
        self.propose = dspy.Predict(signature)

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
            rebuilt = extract_and_rebuild(
                new_candidate, self.manifest, self.target_tool_name
            )
        except SentinelParseError as e:
            self.sentinel_failures += 1
            excerpt = new_candidate[:200] + ("..." if len(new_candidate) > 200 else "")
            logger.warning(
                "BudgetAwareToolProposer: sentinel parse failure (#%d) for %r: %s. "
                "Candidate excerpt: %r",
                self.sentinel_failures,
                self.target_tool_name,
                e,
                excerpt,
            )
            raise

        return {self.component_name: rebuilt}
