"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a DSPy module whose predictor
instructions are the skill text. GEPA optimises predictor instructions, so
it mutates the skill text itself and evaluates the results.
"""

import re
from pathlib import Path
from typing import Optional

import dspy


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

    # Parse YAML frontmatter
    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    # Extract name and description from frontmatter
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


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name in the hermes-agent skills directory.

    Searches recursively for a SKILL.md in a directory matching the skill name.
    """
    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    # Direct match: skills/<category>/<skill_name>/SKILL.md
    for skill_md in skills_dir.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md

    # Fuzzy match: check the name field in frontmatter
    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


class SkillModule(dspy.Module):
    """A DSPy module whose optimisable instructions ARE the skill text.

    The skill body is set as the predictor's signature instructions, which is
    exactly the component GEPA rewrites (``named_predictors()`` ->
    ``signature.with_instructions``). The only input field is ``task_input``;
    the skill text is never passed as an input, so the optimiser can edit it.
    Read the (possibly evolved) text back with :func:`extract_skill_text` or
    the ``skill_text`` property.

    Note: dspy returns instructions through ``inspect.cleandoc``, so outer
    blank lines are trimmed and tabs expanded. Compare an evolved text with
    ``SkillModule(baseline).skill_text``, not with the raw baseline string.
    """

    class TaskWithSkill(dspy.Signature):
        """Placeholder instructions; replaced by the skill text in SkillModule.__init__."""
        task_input: str = dspy.InputField(
            desc="The task to complete. Follow the procedure in your instructions.")
        output: str = dspy.OutputField(desc="Your response, following the instructions")

    def __init__(self, skill_text: str):
        super().__init__()
        if not isinstance(skill_text, str) or not skill_text.strip():
            raise ValueError("skill_text must be a nonempty string")
        self.predictor = dspy.ChainOfThought(self.TaskWithSkill.with_instructions(skill_text))

    @property
    def skill_text(self) -> str:
        return extract_skill_text(self)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)


def extract_skill_text(module: dspy.Module) -> str:
    """The skill text held as instructions by a (possibly GEPA-optimised) SkillModule."""
    predictors = module.named_predictors()
    if len(predictors) != 1:
        raise ValueError(f"expected exactly one predictor, found {len(predictors)}")
    text = predictors[0][1].signature.instructions
    if not isinstance(text, str) or not text.strip():
        raise ValueError("optimised module has empty instructions")
    return text


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
