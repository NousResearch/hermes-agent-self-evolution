"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter via signature instructions.
GEPA can then mutate the skill text and evaluate the results.
"""

import re
from pathlib import Path
from typing import Optional

import dspy


# Unique sentinel that cannot appear in any skill content.
# HTML comment format — markdown/skills never contain HTML comments.
_SKILL_BODY_SENTINEL_ = "\n\n<!-- ___SKILL_EVOLUTION_SENTINEL___ -->\n\n"


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
    """A DSPy module that wraps a skill file for optimization.

    The skill text body is embedded in the signature's instructions so that
    GEPA/MIPROv2 can actually propose mutations to it. The skill text is NOT
    passed as an InputField (which would make it invisible to DSPy optimizers).

    The original skill body is also stored separately in self.skill_body so
    it can be recovered after optimization even if the instruction text is
    replaced entirely by the optimizer.
    """

    def __init__(self, skill_text: str):
        super().__init__()
        # Store original body separately — needed for recovery after optimization
        # since optimizer may replace instruction text entirely.
        self.skill_body = skill_text

        # Embed skill text in the signature instructions so GEPA can optimize it.
        # Use a unique HTML-comment sentinel (cannot appear in any skill content).
        base_sig = self.TaskWithSkill
        base_instructions = base_sig.__doc__ or ""
        enriched_instructions = (
            f"Follow these skill instructions to complete the task:\n\n"
            f"{skill_text}"
            + _SKILL_BODY_SENTINEL_
            + base_instructions
        )
        custom_sig = base_sig.with_instructions(enriched_instructions)
        self.predictor = dspy.ChainOfThought(custom_sig)

    class TaskWithSkill(dspy.Signature):
        """Complete a task following the provided skill instructions.

        You are an AI agent following specific skill instructions to complete a task.
        Read the skill instructions carefully and follow the procedure described.
        """
        task_input: str = dspy.InputField(desc="The task to complete")
        output: str = dspy.OutputField(desc="Your response following the skill instructions")

    def forward(self, task_input: str) -> dspy.Prediction:
        # skill_text is now in the signature instructions, not passed as InputField
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
