"""Tests for the generic artifact-as-instructions DSPy module."""

from unittest.mock import Mock

import dspy

from evolution.core.instruction_module import ArtifactModule


ARTIFACT = "SOME TEXT"


class TestArtifactModule:
    def test_instructions_roundtrip(self):
        """The artifact text must be extractable from the live signature."""
        mod = ArtifactModule(ARTIFACT)

        assert mod.extract() == ARTIFACT

    def test_forward_wires_to_underlying_predictor(self):
        """forward() delegates to self.predict and re-wraps the prediction."""
        mod = ArtifactModule(ARTIFACT)
        mod.predict = Mock(return_value=Mock(output="ok"))

        result = mod.forward("x")

        mod.predict.assert_called_once_with(task_input="x")
        assert result.output == "ok"
        assert isinstance(result, dspy.Prediction)
