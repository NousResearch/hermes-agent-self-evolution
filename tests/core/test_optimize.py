"""Tests for the shared optimizer helpers."""

from unittest.mock import patch

from evolution.core.optimize import GuardedProposer, print_constraints


class TestGuardedProposer:
    def test_marker_dropping_proposal_reverts_to_parent(self):
        with patch(
            "dspy.teleprompt.gepa.instruction_proposal.InstructionProposer.__call__",
            return_value={"predict": "no markers in this proposal"},
        ):
            proposer = GuardedProposer(("<<DESCRIPTION>>", "<</DESCRIPTION>>"))
            result = proposer(
                {"predict": "<<DESCRIPTION>>parent<</DESCRIPTION>>"},
                {},
                ["predict"],
            )

        assert result["predict"] == "<<DESCRIPTION>>parent<</DESCRIPTION>>"

    def test_marker_keeping_proposal_passes_through(self):
        good = "text <<DESCRIPTION>>d<</DESCRIPTION>> more"
        with patch(
            "dspy.teleprompt.gepa.instruction_proposal.InstructionProposer.__call__",
            return_value={"predict": good},
        ):
            proposer = GuardedProposer(("<<DESCRIPTION>>", "<</DESCRIPTION>>"))
            result = proposer({"predict": "parent"}, {}, ["predict"])

        assert result["predict"] == good

    def test_partial_marker_loss_reverted(self):
        with patch(
            "dspy.teleprompt.gepa.instruction_proposal.InstructionProposer.__call__",
            return_value={"predict": "has <<DESCRIPTION>> only"},
        ):
            proposer = GuardedProposer(("<<DESCRIPTION>>", "<</DESCRIPTION>>"))
            result = proposer({"predict": "parent"}, {}, ["predict"])

        assert result["predict"] == "parent"


class TestPrintConstraints:
    def test_returns_all_pass(self):
        from evolution.core.constraints import ConstraintResult

        results = [
            ConstraintResult(passed=True, constraint_name="a", message="ok"),
            ConstraintResult(passed=False, constraint_name="b", message="bad"),
        ]

        class FakeConsole:
            def print(self, *args, **kwargs):
                pass

        assert print_constraints(FakeConsole(), results) is False
        assert print_constraints(FakeConsole(), results[:1]) is True
