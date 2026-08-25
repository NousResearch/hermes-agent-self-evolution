"""Wrap Hermes prompt-builder sections as DSPy optimizable modules.

Phase 3 optimizes named text constants from ``agent/prompt_builder.py`` (for
example ``MEMORY_GUIDANCE`` or ``DEFAULT_AGENT_IDENTITY``) without importing the
Hermes runtime. Static AST loading keeps this safe and dependency-light.
"""

import ast
from pathlib import Path
from typing import Optional

import dspy


PROMPT_BUILDER_RELATIVE = Path("agent") / "prompt_builder.py"


def _literal_string(node: ast.AST) -> Optional[str]:
    """Return a statically evaluable string expression, if possible."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def list_prompt_sections(hermes_agent_path: Path) -> dict[str, str]:
    """List uppercase string constants in ``agent/prompt_builder.py``.

    Returns ``{SECTION_NAME: text}`` for constants that are useful prompt
    sections. Non-string values and private/internal constants are ignored.
    """
    path = hermes_agent_path / PROMPT_BUILDER_RELATIVE
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sections: dict[str, str] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = _literal_string(node.value)
        if not value:
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name.isupper() and not name.startswith("_"):
                sections[name] = value

    return sections


def load_prompt_section(hermes_agent_path: Path, section_name: str) -> dict[str, object]:
    """Load one named prompt-builder section.

    Returns a dict with ``name``, ``text`` and ``path`` keys.
    """
    sections = list_prompt_sections(hermes_agent_path)
    if section_name not in sections:
        available = ", ".join(sorted(sections))
        raise ValueError(f"Prompt section {section_name!r} not found. Available: {available}")
    return {
        "name": section_name,
        "text": sections[section_name],
        "path": hermes_agent_path / PROMPT_BUILDER_RELATIVE,
    }


class PromptSectionModule(dspy.Module):
    """DSPy module that uses a prompt section as mutable instructions."""

    class TaskWithPromptSection(dspy.Signature):
        task_input: str = dspy.InputField(desc="The task or situation to handle")
        output: str = dspy.OutputField(desc="Your response following the prompt section")

    def __init__(self, section_text: str):
        super().__init__()
        signature = self.TaskWithPromptSection.with_instructions(section_text)
        self.predictor = dspy.ChainOfThought(signature)

    @property
    def section_text(self) -> str:
        """The current prompt-section instructions mutated by GEPA/MIPRO."""
        return self.predictor.predict.signature.instructions  # type: ignore[attr-defined]

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)
