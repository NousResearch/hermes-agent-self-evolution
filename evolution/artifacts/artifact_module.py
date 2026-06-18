"""Generic DSPy module for evolving any text artifact.

Mirrors SkillModule but works with arbitrary artifact text.
"""

import dspy


class ArtifactModule(dspy.Module):
    """A DSPy module wrapping an evolvable text artifact.

    The artifact_text represents the body of the artifact
    (e.g., persona rules, eval prompt, constraint rules).
    """

    def __init__(self, artifact_text: str):
        super().__init__()
        self.artifact_text = artifact_text
        self.predictor = dspy.Predict(ArtifactSignature)

    def forward(self, task_input: str) -> dspy.Prediction:
        """Apply the artifact to a task input and return the output."""
        return self.predictor(
            artifact=self.artifact_text,
            task_input=task_input,
        )


class ArtifactSignature(dspy.Signature):
    """Apply an artifact to a given task."""

    artifact = dspy.InputField(desc="The artifact text (rules, persona, prompt, etc.)")
    task_input = dspy.InputField(desc="The task or scenario to apply the artifact to")
    output = dspy.OutputField(desc="The result of applying the artifact to the task")
