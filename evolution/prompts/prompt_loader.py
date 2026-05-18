"""Discover and rewrite individual system-prompt section constants.

Phase 3 evolves the named string constants inside
``hermes-agent/agent/prompt_builder.py`` (e.g. ``DEFAULT_AGENT_IDENTITY``,
``MEMORY_GUIDANCE``). Each section is treated as a stand-alone artifact
whose value is an optimizable string. PLATFORM_HINTS is a dict of platform
name -> string; each platform value is exposed as its own section name
(``PLATFORM_HINTS:<platform>``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_PROMPT_BUILDER_RELATIVE_PATH = Path("agent") / "prompt_builder.py"

EVOLVABLE_SECTION_NAMES = (
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
)

PLATFORM_HINTS_NAME = "PLATFORM_HINTS"


@dataclass
class PromptSection:
    """A named system-prompt section discovered in prompt_builder.py."""

    name: str  # logical name, e.g. DEFAULT_AGENT_IDENTITY or PLATFORM_HINTS:whatsapp
    var_name: str  # underlying Python identifier
    platform_key: Optional[str]  # set when section is a value from PLATFORM_HINTS
    text: str
    source_path: Path


def _eval_string_node(node: ast.AST) -> Optional[str]:
    """Return the literal value of an AST node that evaluates to a string."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, str):
        return value
    return None


def load_sections_from_file(prompt_builder_path: Path) -> list[PromptSection]:
    """Parse prompt_builder.py and return every evolvable section we find."""
    try:
        source = prompt_builder_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    sections: list[PromptSection] = []
    wanted = set(EVOLVABLE_SECTION_NAMES) | {PLATFORM_HINTS_NAME}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue

        if target.id == PLATFORM_HINTS_NAME:
            if not isinstance(node.value, ast.Dict):
                continue
            for key_node, value_node in zip(node.value.keys, node.value.values):
                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                    continue
                text = _eval_string_node(value_node)
                if not text:
                    continue
                sections.append(
                    PromptSection(
                        name=f"{PLATFORM_HINTS_NAME}:{key_node.value}",
                        var_name=PLATFORM_HINTS_NAME,
                        platform_key=key_node.value,
                        text=text,
                        source_path=prompt_builder_path,
                    ),
                )
            continue

        text = _eval_string_node(node.value)
        if not text:
            continue
        sections.append(
            PromptSection(
                name=target.id,
                var_name=target.id,
                platform_key=None,
                text=text,
                source_path=prompt_builder_path,
            ),
        )
    return sections


def discover_prompt_sections(hermes_agent_path: Path) -> list[PromptSection]:
    path = hermes_agent_path / DEFAULT_PROMPT_BUILDER_RELATIVE_PATH
    if not path.exists():
        return []
    return load_sections_from_file(path)


def find_prompt_section(name: str, hermes_agent_path: Path) -> Optional[PromptSection]:
    for section in discover_prompt_sections(hermes_agent_path):
        if section.name == name:
            return section
    return None


def identity_traits_present(text: str) -> dict[str, bool]:
    """Phase 3 constraint helper: ensures evolved identity preserves core traits."""
    if not text:
        return {"helpful": False, "direct": False, "admits_uncertainty": False}
    lowered = text.lower()
    return {
        "helpful": "help" in lowered,
        "direct": "direct" in lowered or "concise" in lowered or "clear" in lowered,
        "admits_uncertainty": (
            "uncertain" in lowered
            or "admit" in lowered
            or "don't know" in lowered
            or "do not know" in lowered
        ),
    }
