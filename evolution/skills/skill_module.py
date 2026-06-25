"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module where the
skill body itself is the optimizable instruction text. GEPA can then mutate the
actual skill procedure rather than only a wrapper prompt around an immutable
``skill_instructions`` input.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import dspy

_FRONTMATTER_RE = re.compile(
    r"\A[ \t]*---[ \t]*\r?\n(?P<frontmatter>.*?)(?:\r?\n)---[ \t]*(?:\r?\n|\Z)(?P<body>.*)\Z",
    re.DOTALL,
)


def split_frontmatter(raw: str) -> tuple[str, str]:
    """Split a SKILL.md file into ``(frontmatter, body)``.

    Only a leading YAML frontmatter block is recognized. Internal markdown
    horizontal rules are left untouched.
    """

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return "", raw.strip()
    return match.group("frontmatter").strip(), match.group("body").strip()


def strip_leading_frontmatter(text: str) -> str:
    """Return text with one leading YAML frontmatter block removed.

    Optimizers occasionally emit a full ``SKILL.md`` even though the evolution
    pipeline only asks them to mutate the body. Deployment preserves the
    baseline metadata, so optimizer-supplied leading metadata must be stripped
    before reassembly. Non-leading ``---`` blocks are preserved.
    """

    _, body = split_frontmatter(text)
    return body.strip()


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
    frontmatter, body = split_frontmatter(raw)

    # Extract name and description from frontmatter without requiring PyYAML at
    # import time. This mirrors the current lightweight parser while making the
    # frontmatter boundary detection robust.
    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        stripped = line.strip()
        if stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip().strip("'\"")
        elif stripped.startswith("description:"):
            description = stripped.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name in the hermes-agent skills directory."""

    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    # Direct match: skills/<skill_name>/SKILL.md
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


def _make_skill_signature(skill_text: str) -> type[dspy.Signature]:
    """Create a DSPy signature whose instructions are the skill body itself."""

    class TaskWithSkill(dspy.Signature):
        __doc__ = skill_text

        task_input: str = dspy.InputField(desc="The task to complete")
        output: str = dspy.OutputField(desc="Your response following the skill instructions")

    return TaskWithSkill


def _signature_instruction_text(signature: object) -> str:
    """Return a DSPy signature's current instruction text, if available."""

    if signature is None:
        return ""

    for attr in ("instructions", "__doc__"):
        text = getattr(signature, attr, None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _predictor_instruction_text(predictor: object) -> str:
    """Probe known DSPy predictor paths for optimized signature text."""

    for signature in (
        getattr(predictor, "signature", None),
        getattr(getattr(predictor, "predict", None), "signature", None),
    ):
        text = _signature_instruction_text(signature)
        if text:
            return text
    return ""


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    GEPA mutates the signature instructions. Therefore the skill body must live
    in the signature instructions, not in a normal runtime input field; otherwise
    optimization only improves a wrapper prompt while the deployed skill remains
    unchanged.
    """

    def __init__(self, skill_text: str):
        super().__init__()
        self.skill_text = skill_text.strip()
        self.predictor = dspy.ChainOfThought(_make_skill_signature(self.skill_text))

    def current_skill_text(self) -> str:
        """Return the current optimizable skill instructions for this module."""

        return _predictor_instruction_text(self.predictor) or self.skill_text

    def forward(self, task_input: str) -> dspy.Prediction:
        current_skill_text = self.current_skill_text()
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output, skill_text=current_skill_text)


def extract_evolved_skill_text(module: dspy.Module, fallback: str = "") -> str:
    """Extract evolved skill text from an optimized DSPy module.

    DSPy stores optimized signature instructions under slightly different
    attribute paths across versions. We probe the known paths before falling
    back to a ``skill_text`` attribute. The returned text is body-only and safe
    to pass to ``reassemble_skill``.
    """

    candidates: list[object] = []

    predictor = getattr(module, "predictor", None)
    if predictor is not None:
        predictor_text = _predictor_instruction_text(predictor)
        if predictor_text:
            candidates.append(predictor_text)
        candidates.extend(
            [
                getattr(predictor, "signature", None),
                getattr(getattr(predictor, "predict", None), "signature", None),
            ]
        )

    candidates.extend(
        [
            getattr(module, "signature", None),
            getattr(module, "skill_text", None),
            fallback,
        ]
    )

    for candidate in candidates:
        if candidate is None:
            continue
        text = _signature_instruction_text(candidate) if not isinstance(candidate, str) else candidate
        if isinstance(text, str) and text.strip():
            return strip_leading_frontmatter(text)

    return ""


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata) and
    replaces only the body with the evolved version.
    """

    body = strip_leading_frontmatter(evolved_body)
    return f"---\n{frontmatter.strip()}\n---\n\n{body}\n"
