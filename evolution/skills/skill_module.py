"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter. GEPA can then
mutate the skill text and evaluate the results.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import dspy

logger = logging.getLogger(__name__)


def load_skill(skill_path: Path) -> dict:
    """Load a skill file and parse its frontmatter + body.

    Returns:
        {
            "path": Path,
            "raw": str (full file content),
            "frontmatter": str (YAML between --- markers),
            "body": str (markdown after frontmatter),
            "name": str,
            "description": str,
        }
    """
    raw = skill_path.read_text()

    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, sources) -> Optional[Path]:
    """Find a skill by name across a list of `SkillSource` adapters.

    Walks `sources` in order; returns the first SKILL.md any source resolves.
    The actual layout-specific logic lives in each adapter (Hermes, Claude
    Code, LocalDir). Use `evolution.core.discover_skill_sources()` to build
    the default list.
    """
    for source in sources:
        path = source.find_skill(skill_name)
        if path is not None:
            return path
    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill body is installed as the predictor's signature instructions —
    the same surface GEPA mutates. Reading `module.skill_text` after
    optimization returns the mutated text.
    """

    class TaskWithSkill(dspy.Signature):
        """Placeholder. SkillModule.__init__ overwrites the inner Predict's
        signature.instructions per-instance via with_instructions(skill_text);
        do not rely on this docstring at the class level."""
        task_input: str = dspy.InputField(desc="The task to complete")
        output: str = dspy.OutputField(desc="Your response following the skill instructions")

    def __init__(self, skill_text: str):
        super().__init__()
        self.predictor = dspy.ChainOfThought(self.TaskWithSkill)
        # GEPA mutates Predict.signature.instructions via named_predictors();
        # install the skill body there so forward() reads exactly what GEPA writes.
        self.predictor.predict.signature = self.predictor.predict.signature.with_instructions(skill_text)

    def forward(
        self,
        task_input: str,
        closed_loop_task_id: Optional[str] = None,
    ) -> dspy.Prediction:
        if closed_loop_task_id is not None:
            # Behavioral example: skip the predictor LM call. The metric's
            # behavioral branch reads _closed_loop_task_id off the Prediction
            # and routes to the closed-loop cache for scoring, so the LM call
            # would be wasted spend.
            return dspy.Prediction(
                output="",
                _closed_loop_task_id=closed_loop_task_id,
                _candidate_text=self.skill_text,
            )
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)

    @property
    def skill_text(self) -> str:
        return self.predictor.predict.signature.instructions


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.

    Defensive: if GEPA's reflection LM produces a body that itself
    starts with a frontmatter-looking block (mimicking YAML structure),
    strip it so we don't end up with double frontmatter. The strip is
    logged so we can spot whether the reflection prompt needs adjustment
    rather than fixing it silently.
    """
    body = evolved_body.lstrip()
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            logger.warning(
                "reassemble_skill: stripped a leading frontmatter-like block from "
                "GEPA-mutated body; the reflection LM may be mimicking YAML."
            )
            body = parts[2].lstrip()
    return f"---\n{frontmatter}\n---\n\n{body}\n"
