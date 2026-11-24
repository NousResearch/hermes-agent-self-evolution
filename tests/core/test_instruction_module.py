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


class TestSkillModuleUsesArtifactModule:
    def test_subclasses_artifact_module(self):
        from evolution.skills.skill_module import SkillModule

        assert issubclass(SkillModule, ArtifactModule)

    def test_skill_text_is_the_optimizable_instructions(self):
        from evolution.skills.skill_module import SkillModule

        mod = SkillModule(ARTIFACT)

        assert mod.extract() == ARTIFACT
        assert mod.skill_text == ARTIFACT

    def test_public_constructor_takes_only_skill_text(self):
        """Other modules construct SkillModule(skill_text) positionally."""
        import inspect

        from evolution.skills.skill_module import SkillModule

        params = list(inspect.signature(SkillModule.__init__).parameters)
        assert params == ["self", "skill_text"]

    def test_forward_is_inherited_and_wired(self):
        from evolution.skills.skill_module import SkillModule

        mod = SkillModule(ARTIFACT)
        mod.predict = Mock(return_value=Mock(output="ok"))

        assert mod.forward("x").output == "ok"
        mod.predict.assert_called_once_with(task_input="x")
