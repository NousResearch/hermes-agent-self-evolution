"""Generic DSPy module that treats an arbitrary text artifact as instructions.

    mod = ArtifactModule(open("SKILL.md").read())
    pred = mod(task_input="...")
    optimized_text = mod.extract()
"""

import dspy


class ArtifactSignature(dspy.Signature):
    """Complete a task by following the given instructions."""

    task_input: str = dspy.InputField(desc="The task to complete")
    output: str = dspy.OutputField(
        desc="Your response following the given instructions"
    )


class ArtifactModule(dspy.Module):
    """A DSPy module whose optimizable parameter IS a text artifact.

    GEPA/MIPRO mutate signature INSTRUCTIONS, not module attributes or input
    fields; passing the artifact as an input field makes it invisible to the
    optimizer. So the artifact text becomes the signature's instructions.
    """

    def __init__(self, artifact_text: str):
        super().__init__()
        if not artifact_text or not artifact_text.strip():
            raise ValueError("artifact_text must be non-empty")
        self.predict = dspy.ChainOfThought(
            ArtifactSignature.with_instructions(artifact_text)
        )

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predict(task_input=task_input)
        return dspy.Prediction(output=result.output)

    def extract(self) -> str:
        """Return the current (possibly optimized) instructions.

        ChainOfThought wraps an inner dspy.Predict that owns the signature,
        hence the double ``.predict``.
        """
        return self.predict.predict.signature.instructions
