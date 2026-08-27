"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module where
the skill text is the optimizable parameter. GEPA can then mutate the skill
text and evaluate the results.

Every prediction carries the exact instructions that produced it. That is what
lets the judge score the candidate instead of the baseline — DSPy hands the
metric a gold example and a prediction but no reference to the program that
made it, so the program has to say so itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import dspy

# Frontmatter keys we parse out. Read with a shallow scan rather than a YAML
# parse so a nested `metadata:` block (182 of the installed skills have one)
# cannot have its inner `name:` mistaken for the skill's own.
_TOP_LEVEL_KEYS = ("name", "description", "version")


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
            "version": str,
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

    fields = parse_frontmatter_fields(frontmatter)

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": fields.get("name", ""),
        "description": fields.get("description", ""),
        "version": fields.get("version", ""),
    }


def parse_frontmatter_fields(frontmatter: str) -> dict[str, str]:
    """Extract top-level scalar fields from YAML frontmatter.

    Only lines at zero indentation are considered, so keys nested under
    ``metadata:`` or ``hermes:`` never shadow the real ones. This is a
    deliberate shallow scan: a full YAML parse would pull in a dependency and
    still need this same "top level only" rule to behave correctly.
    """
    fields: dict[str, str] = {}
    for line in frontmatter.split("\n"):
        if not line.strip() or line.startswith((" ", "\t", "#", "-")):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in _TOP_LEVEL_KEYS and key not in fields:
            fields[key] = value.strip().strip("'\"")
    return fields


def bump_version(frontmatter: str) -> str:
    """Increment the skill's semantic patch version, adding one if absent.

    An evolved skill that keeps its predecessor's version number is
    indistinguishable from it downstream — caches, deploy tooling and anyone
    reading a diff all assume the version tracks the content.
    """
    fields = parse_frontmatter_fields(frontmatter)
    current = fields.get("version", "").strip()

    new_version = _next_version(current)

    lines = frontmatter.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("version:"):
            lines[i] = f"version: {new_version}"
            return "\n".join(lines)

    # No version field — add one after `name:` if present, else at the top.
    for i, line in enumerate(lines):
        if line.startswith("name:"):
            lines.insert(i + 1, f"version: {new_version}")
            return "\n".join(lines)
    return f"version: {new_version}\n" + "\n".join(lines)


def _next_version(current: str) -> str:
    parts = current.split(".") if current else []
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return f"{parts[0]}.{parts[1]}.1"
    if current.isdigit():
        return f"{current}.0.1"
    return "1.0.1"


def find_skill(skill_name: str, *roots: Path) -> Optional[Path]:
    """Find a skill by name across every skills tree given.

    A real install keeps skills in more than one place: the repo ships them,
    the user tree adds more, and each profile has its own. Searching only the
    repo missed every skill a profile actually runs — including the ones the
    weekly rotation was pointed at.

    Args:
        skill_name: Directory name or frontmatter ``name`` of the skill.
        roots: Directories to search. Each may be a skills directory itself or
            a parent containing one.
    """
    search_dirs: list[Path] = []
    for root in roots:
        if not root:
            continue
        root = Path(root)
        if root.name == "skills" and root.is_dir():
            search_dirs.append(root)
        elif (root / "skills").is_dir():
            search_dirs.append(root / "skills")
        elif root.is_dir():
            search_dirs.append(root)

    # Pass 1: directory name match, across all trees before falling back.
    for skills_dir in search_dirs:
        for skill_md in sorted(skills_dir.rglob("SKILL.md")):
            if skill_md.parent.name == skill_name:
                return skill_md

    # Pass 2: frontmatter name match.
    for skills_dir in search_dirs:
        for skill_md in sorted(skills_dir.rglob("SKILL.md")):
            try:
                head = skill_md.read_text(errors="replace")[:800]
            except OSError:
                continue
            fields = parse_frontmatter_fields(head.split("---")[1] if "---" in head else head)
            if fields.get("name") == skill_name:
                return skill_md

    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill text is embedded as the signature's instructions — the part that
    DSPy optimizers (GEPA, MIPROv2) actually mutate. After
    ``optimizer.compile()``, call ``get_evolved_text()`` to extract the result.
    """

    def __init__(self, skill_text: str, bundle_context: str = ""):
        super().__init__()
        # Store original for reference/diff.
        self.skill_text = skill_text
        self.bundle_context = bundle_context

        instructions = skill_text
        if bundle_context:
            # Supporting files are shown to the optimizer but are not part of
            # the artifact being edited, so they are appended here and stripped
            # back off in get_evolved_text().
            instructions = f"{skill_text}\n\n{_BUNDLE_MARKER}\n{bundle_context}"

        sig = dspy.Signature(
            "task_input: str -> output: str",
            instructions=instructions,
        )
        self.predictor = dspy.ChainOfThought(sig)

    def current_instructions(self) -> str:
        """The live instruction text, including any optimizer mutations.

        DSPy optimizers mutate ``predictor.signature.instructions``, not
        instance attributes. ``ChainOfThought`` wraps an inner ``Predict`` at
        ``self.predictor.predict`` — that is where the signature lives.
        """
        return self.predictor.predict.signature.instructions

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        # Attach the instructions that produced this output so the metric can
        # judge the candidate rather than the baseline.
        return dspy.Prediction(
            output=result.output,
            skill_text=self.get_evolved_text(),
        )

    def get_evolved_text(self) -> str:
        """The evolved skill text, with any appended bundle context removed."""
        text = self.current_instructions()
        if _BUNDLE_MARKER in text:
            text = text.split(_BUNDLE_MARKER, 1)[0]
        return text.rstrip()


_BUNDLE_MARKER = "--- SUPPORTING FILES (reference only, not part of this skill file) ---"


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
