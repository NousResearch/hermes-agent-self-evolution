"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable instruction. GEPA can then
mutate the skill text and evaluate the results.
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
    Uses os.walk with followlinks=True to handle symlinked skill directories.
    """
    import os

    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    for root, dirs, files in os.walk(str(skills_dir), followlinks=True):
        if "SKILL.md" in files and Path(root).name == skill_name:
            return Path(root) / "SKILL.md"

    # Fuzzy match: check the name field in frontmatter
    for root, dirs, files in os.walk(str(skills_dir), followlinks=True):
        if "SKILL.md" not in files:
            continue
        skill_md = Path(root) / "SKILL.md"
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill text (body) is set as the Signature's instruction, making
    it the optimizable parameter that GEPA/MIPROv2 can evolve. On each
    forward pass, the module uses the skill text as context to complete
    the given task.
    """

    def __init__(self, skill_text: str):
        super().__init__()
        self.skill_text = skill_text

        # Build a dynamic Signature with the skill text as the instruction.
        # This makes the instruction the optimizable parameter for GEPA/MIPROv2.
        signature = dspy.Signature(
            "task_input -> output",
            instructions=skill_text,
        )
        signature = signature.with_updated_fields(
            "task_input",
            desc="The task to complete following the skill instructions",
        )
        signature = signature.with_updated_fields(
            "output",
            desc="Your response following the skill instructions",
        )
        self.predictor = dspy.ChainOfThought(signature)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        # After optimization, extract the (possibly evolved) instruction
        # back as the skill text for reassembly.
        if hasattr(self.predictor, 'signature') and hasattr(self.predictor.signature, 'instructions'):
            self.skill_text = self.predictor.signature.instructions
        return dspy.Prediction(output=result.output)


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
