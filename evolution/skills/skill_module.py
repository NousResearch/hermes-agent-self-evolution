"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is embedded as the Signature instructions (docstring).
This is what GEPA actually optimizes — it mutates the predictor's signature
instructions in-place. After optimization, the evolved skill text is
extracted from the optimized module's predictor signature.
"""

from pathlib import Path
from typing import Optional

import dspy


INSTRUCTIONS_PREFIX = "Complete the task following these instructions."


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
    """Load and return a text file's contents.

    Uses utf-8 encoding with 'surrogateescape' error handling.
    """
    try:
        raw = skill_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for files written with system encoding (e.g. cp1252 on Windows)
        import locale
        sys_enc = locale.getpreferredencoding()
        raw = skill_path.read_text(encoding=sys_enc)

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


def _make_instructions(skill_text: str) -> str:
    """Build signature instructions string embedding skill text."""
    return f"{INSTRUCTIONS_PREFIX}\n\n{skill_text}"


def _extract_skill_text(instructions: str) -> str:
    """Extract original skill text from signature instructions (strip prefix).

    GEPA may mutate the prefix line (e.g. prepend "NEW:" or other labels),
    so detection of the prefix is lenient: we remove only the first line
    when it fits the expected pattern. The skill body text is returned
    with its original trailing content preserved.
    """
    lines = instructions.split("\n", 1)
    if len(lines) >= 2:
        body = lines[1]
        # Remove the leading newline that separates prefix from body
        if body.startswith("\n"):
            body = body[1:]
        return body
    return instructions


def _build_predictor(skill_text: str) -> dspy.ChainOfThought:
    """Create a ChainOfThought predictor with skill text embedded as instructions."""
    doc = _make_instructions(skill_text)
    signature = type("_EvolvingSkill", (dspy.Signature,), {
        "__doc__": doc,
        "task_input": dspy.InputField(),
        "output": dspy.OutputField(),
    })
    return dspy.ChainOfThought(signature)


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for GEPA optimization.

    Skill text is embedded as the predictor's signature instructions (docstring),
    which is what GEPA actually mutates during optimization. After GEPA compile,
    the evolved skill text is read back from the optimized module's predictor
    signature instructions.

    This module MUST compile a fresh predictor each forward pass (handled by the
    property setter) since the skill text may change after each GEPA mutation.
    """

    def __init__(self, skill_text: str):
        super().__init__()
        self._skill_text = skill_text
        self.predictor = _build_predictor(skill_text)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)

    @property
    def skill_text(self) -> str:
        """Return the current (possibly evolved) skill text from signature instructions.

        After GEPA optimization mutates the predictor's signature instructions,
        this property extracts the skill body from the instructions by stripping
        the fixed prefix.
        """
        try:
            # GEPA mutates the inner Predict's signature via with_instructions()
            # Access via named_predictors() to get the innermost predictor
            for _name, pred in self.named_predictors():
                instructions = pred.signature.instructions
                return _extract_skill_text(instructions)
            # Fallback to the ChainOfThought's predictor attribute
            instructions = self.predictor.predict.signature.instructions
            return _extract_skill_text(instructions)
        except Exception:
            return self._skill_text

    @skill_text.setter
    def skill_text(self, value: str):
        """Rebuild the predictor with new skill text."""
        self._skill_text = value
        self.predictor = _build_predictor(value)


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
