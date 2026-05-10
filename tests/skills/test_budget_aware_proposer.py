"""Tests for the GEPA-compatible BudgetAwareProposer.

The proposer must:
- Conform to gepa.core.adapter.ProposalFn (callable returning dict[str, str]).
- Bake real char numbers into the predictor's signature instructions
  (not leave format placeholders) so the LM sees a concrete budget.
- Skip components missing from candidate or reflective_dataset (defensive).
- Warn (but not raise) when the LM's proposal exceeds the target budget.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evolution.skills.budget_aware_proposer import (
    _BUDGET_AWARE_INSTRUCTIONS,
    _BUDGET_AWARE_INSTRUCTIONS_GROWTH,
    _GROWTH_EXAMPLE_AFTER,
    _GROWTH_EXAMPLE_BEFORE,
    _TIGHT_INSTRUCTION_EXAMPLE,
    BudgetAwareProposer,
)


class TestProposerConstruction:
    def test_target_chars_subtracts_safety_margin_from_growth(self):
        # Default safety_margin=0.10. With max_growth=0.20 the LM is told
        # the target is baseline * 1.10, not baseline * 1.20 — leaves
        # headroom for the observed ~8-9% overshoot.
        proposer = BudgetAwareProposer(baseline_chars=1000, max_growth=0.2)
        assert proposer.target_chars == 1100

    def test_safety_margin_can_be_disabled(self):
        # Passing safety_margin=0 reproduces the pre-PR behavior.
        proposer = BudgetAwareProposer(
            baseline_chars=1000, max_growth=0.2, safety_margin=0.0
        )
        assert proposer.target_chars == 1200

    def test_safety_margin_clamped_at_zero_growth(self):
        # If safety_margin >= max_growth the prompt should ask for zero
        # growth, not a negative target.
        proposer = BudgetAwareProposer(
            baseline_chars=1000, max_growth=0.1, safety_margin=0.5
        )
        assert proposer.target_chars == 1000

    def test_target_chars_floored_at_one_for_zero_baseline(self):
        # Defensive: a 0 baseline shouldn't yield target_chars=0.
        proposer = BudgetAwareProposer(baseline_chars=0, max_growth=0.2)
        assert proposer.target_chars == 1

    def test_signature_instructions_have_baked_in_numbers(self):
        # 1264 baseline, max_growth=0.20, default safety_margin=0.10
        # → effective prompt growth 0.10 → target 1264 * 1.10 = 1390.4 → 1390.
        proposer = BudgetAwareProposer(baseline_chars=1264, max_growth=0.2)
        instructions = proposer.propose.signature.instructions
        assert "1264" in instructions
        assert "1390" in instructions
        # No leaked format placeholders.
        assert "{baseline_chars}" not in instructions
        assert "{target_chars}" not in instructions


class TestProposerInstructions:
    """The static instruction template should hard-code the budget framing.

    The phrasing patterns here are load-bearing — they're drawn from the
    prompt-engineering literature on length-constrained generation
    (countdown framing, "at most N", loss frame, one-shot example). Lock
    the patterns so a future template-tweak doesn't silently regress to
    soft phrasing.
    """

    def test_template_has_required_format_keys(self):
        # If we rename a key in the f-string the .format() call below would
        # raise — surface it as a test rather than at GEPA call time.
        formatted = _BUDGET_AWARE_INSTRUCTIONS.format(
            baseline_chars=100, target_chars=120, tight_example="example",
        )
        assert "100" in formatted
        assert "120" in formatted
        assert "example" in formatted

    def test_template_uses_at_most_phrasing(self):
        # arXiv:2508.13805 + multi-source A/B: "at most N" beats "must be"
        # and "should not exceed". The phrasing is load-bearing.
        assert "at most" in _BUDGET_AWARE_INSTRUCTIONS

    def test_template_uses_countdown_framing_at_top(self):
        # Lead with the budget. The literature shows this raises compliance
        # vs. burying the budget mid-prompt.
        first_line = _BUDGET_AWARE_INSTRUCTIONS.splitlines()[0]
        assert "Length budget" in first_line
        assert "at most" in first_line

    def test_template_uses_loss_frame(self):
        # "Each character above N is wasted" turns brevity into a
        # directly-costed resource. Known compliance lever.
        assert "wasted" in _BUDGET_AWARE_INSTRUCTIONS

    def test_template_includes_one_shot_example_block(self):
        # The tight example is rendered into the prompt via a triple-quoted
        # block; the LM sees what good looks like at the target size.
        formatted = _BUDGET_AWARE_INSTRUCTIONS.format(
            baseline_chars=100, target_chars=200,
            tight_example=_TIGHT_INSTRUCTION_EXAMPLE,
        )
        assert '"""' in formatted
        # The curated example mentions VAULT_PATH; if the example is ever
        # swapped this assertion needs updating, but at least the test
        # forces a deliberate change.
        assert "VAULT_PATH" in formatted

    def test_tight_example_is_short_enough_to_be_a_pattern(self):
        # The example is meant to demonstrate brevity. If it ever drifts
        # past ~400 chars it stops being a useful one-shot.
        assert len(_TIGHT_INSTRUCTION_EXAMPLE) < 400, len(_TIGHT_INSTRUCTION_EXAMPLE)


class TestProposerCallable:
    def _patched_proposer(self, returned_text: str) -> BudgetAwareProposer:
        proposer = BudgetAwareProposer(baseline_chars=100, max_growth=0.2)
        proposer.propose = MagicMock(
            return_value=SimpleNamespace(improved_instruction=returned_text)
        )
        return proposer

    def test_returns_dict_mapping_component_to_new_text(self):
        proposer = self._patched_proposer("new short instruction")
        result = proposer(
            candidate={"predict": "old instruction"},
            reflective_dataset={"predict": [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "z"}]},
            components_to_update=["predict"],
        )
        assert result == {"predict": "new short instruction"}
        # And the predictor was called once with the rendered examples.
        proposer.propose.assert_called_once()
        kwargs = proposer.propose.call_args.kwargs
        assert kwargs["current_instruction"] == "old instruction"
        assert "Example 1" in kwargs["examples_with_feedback"]

    def test_skips_component_missing_from_candidate(self):
        proposer = self._patched_proposer("ignored")
        result = proposer(
            candidate={"predict": "x"},
            reflective_dataset={"predict": [{"k": "v"}], "missing": [{"k": "v"}]},
            components_to_update=["missing"],
        )
        assert result == {}
        proposer.propose.assert_not_called()

    def test_skips_component_missing_from_reflective_dataset(self):
        proposer = self._patched_proposer("ignored")
        result = proposer(
            candidate={"predict": "x", "missing": "y"},
            reflective_dataset={"predict": [{"k": "v"}]},
            components_to_update=["missing"],
        )
        assert result == {}
        proposer.propose.assert_not_called()

    def test_warns_on_oversized_proposal(self, caplog):
        # baseline=100, max_growth=0.2, default safety_margin=0.10
        # → effective 0.10 → target 100 * 1.10 = 110 chars.
        proposer = self._patched_proposer("x" * 200)
        with caplog.at_level(logging.WARNING, logger="evolution.skills.budget_aware_proposer"):
            result = proposer(
                candidate={"predict": "old"},
                reflective_dataset={"predict": [{"Inputs": "i", "Generated Outputs": "o", "Feedback": "f"}]},
                components_to_update=["predict"],
            )

        # Soft enforcement: the proposal is still returned (truncating
        # mid-sentence would corrupt instructions); we just log so we
        # can see whether the LM is honoring the budget across runs.
        assert result == {"predict": "x" * 200}
        assert any("came back at 200 chars" in rec.message for rec in caplog.records)
        assert any("target 110" in rec.message for rec in caplog.records)

    def test_logs_observation_on_every_call(self, caplog):
        """Per-call info log fires regardless of overshoot, so e2e runs
        show whether the budget signal is reaching the proposer at all
        without grepping GEPA's internal logs."""
        # 80 chars proposal, target 110 → under budget, no warning.
        proposer = self._patched_proposer("x" * 80)
        with caplog.at_level(logging.INFO, logger="evolution.skills.budget_aware_proposer"):
            proposer(
                candidate={"predict": "old"},
                reflective_dataset={"predict": [{"Inputs": "i", "Generated Outputs": "o", "Feedback": "f"}]},
                components_to_update=["predict"],
            )

        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert any(
            "BudgetAwareProposer iter" in r.message and "proposed[predict]=80" in r.message
            for r in info_records
        ), info_records


class TestFeedbackBudgetReachesPrompt:
    """Lock the contract that a [BUDGET] line in a reflective example's
    Feedback value survives _format_examples rendering and ends up in the
    string the reflection LM consumes. Without this test the feedback
    signal we ship from fitness.py could be silently dropped by a future
    rendering refactor.
    """

    def test_budget_line_in_feedback_appears_in_rendered_prompt(self):
        proposer = BudgetAwareProposer(baseline_chars=1264, max_growth=0.2)
        feedback_with_budget = (
            "judge says X.\n\n"
            "[BUDGET] Your current instruction is 1647 chars vs baseline 1264 chars "
            "(+30.3%); target ≤1390 chars."
        )
        rendered = proposer._format_examples([
            {"Inputs": "task input", "Generated Outputs": "agent output",
             "Feedback": feedback_with_budget},
        ])

        assert "[BUDGET]" in rendered
        assert "1647 chars" in rendered
        assert "+30.3%" in rendered
        assert "judge says X" in rendered  # original judge feedback preserved


class TestExampleFormatting:
    def test_renders_example_dict_as_markdown(self):
        proposer = BudgetAwareProposer(baseline_chars=100, max_growth=0.2)
        out = proposer._format_examples([
            {"Inputs": "the input", "Generated Outputs": "the output", "Feedback": "the critique"},
            {"Inputs": "input2", "Generated Outputs": "output2", "Feedback": "critique2"},
        ])
        assert "### Example 1" in out
        assert "**Inputs:** the input" in out
        assert "### Example 2" in out
        assert "**Feedback:** critique2" in out

    def test_handles_empty_dataset(self):
        proposer = BudgetAwareProposer(baseline_chars=100, max_growth=0.2)
        assert proposer._format_examples([]) == ""


class TestProposerMode:
    """Mode dispatch: compression-mode (default) and growth-mode.

    The mode parameter controls which static instruction template gets baked
    into the DSPy Signature at construction time. Only the prompt template
    differs; budget arithmetic, callable surface, and feedback formatting
    are mode-independent.
    """

    def test_default_mode_is_compression(self):
        proposer = BudgetAwareProposer(baseline_chars=1000)
        assert proposer.mode == "compression"
        assert "Cut redundant phrasing" in proposer.propose.signature.instructions

    def test_growth_mode_uses_growth_template(self):
        proposer = BudgetAwareProposer(baseline_chars=1000, mode="growth")
        instructions = proposer.propose.signature.instructions
        assert proposer.mode == "growth"
        # Hard-constraint phrase that defines the growth-mode contract.
        assert (
            "every addition or refinement must quote or paraphrase a specific phrase from the feedback"
            in instructions
        )
        # And the compression-only phrasing is NOT present in growth mode.
        assert "Cut redundant phrasing" not in instructions

    def test_compression_mode_signature_unchanged(self):
        # Regression guard: any refactor that drops the literature-backed
        # compression phrasing should fail this assertion.
        proposer = BudgetAwareProposer(baseline_chars=1000, mode="compression")
        instructions = proposer.propose.signature.instructions
        assert "Cut redundant phrasing" in instructions
        assert "at most" in instructions
        assert "wasted" in instructions

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="must be 'compression' or 'growth'"):
            BudgetAwareProposer(baseline_chars=1000, mode="bogus")

    def test_target_chars_independent_of_mode(self):
        # Same budget arithmetic produces same target_chars regardless of
        # which prompt template gets baked in.
        compression = BudgetAwareProposer(
            baseline_chars=1264, max_growth=0.2, safety_margin=0.10, mode="compression",
        )
        growth = BudgetAwareProposer(
            baseline_chars=1264, max_growth=0.2, safety_margin=0.10, mode="growth",
        )
        assert compression.target_chars == growth.target_chars == 1390

    def test_growth_mode_includes_neither_branch_in_rubric(self):
        # The (c) "neither — skip" branch is the structural anti-hallucination
        # move. Force-fitting non-actionable failures into "add an instruction"
        # is the failure mode this branch prevents.
        proposer = BudgetAwareProposer(baseline_chars=1000, mode="growth")
        instructions = proposer.propose.signature.instructions
        assert "(c) neither" in instructions
        assert "the failure is not actionable from instruction text" in instructions

    def test_growth_mode_includes_empty_feedback_sentinel(self):
        # Without a sentinel the LM has no anchor under empty feedback and
        # tends to invent.
        proposer = BudgetAwareProposer(baseline_chars=1000, mode="growth")
        instructions = proposer.propose.signature.instructions
        assert (
            "If the feedback below is empty or contains no concrete failures, return the current instruction unchanged"
            in instructions
        )

    def test_growth_template_format_keys_render_cleanly(self):
        # The template uses more format keys than compression
        # (example_before_chars / example_after_chars / two example bodies).
        # If a key gets renamed without updating __init__, format() raises;
        # surface that as a test rather than a runtime error.
        formatted = _BUDGET_AWARE_INSTRUCTIONS_GROWTH.format(
            baseline_chars=100,
            target_chars=120,
            example_before_chars=len(_GROWTH_EXAMPLE_BEFORE),
            example_after_chars=len(_GROWTH_EXAMPLE_AFTER),
            growth_example_before=_GROWTH_EXAMPLE_BEFORE,
            growth_example_after=_GROWTH_EXAMPLE_AFTER,
        )
        assert "100" in formatted
        assert "120" in formatted
        assert _GROWTH_EXAMPLE_BEFORE in formatted
        assert _GROWTH_EXAMPLE_AFTER in formatted
