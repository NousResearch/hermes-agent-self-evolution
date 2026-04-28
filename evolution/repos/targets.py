"""Repository target discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TargetSpec:
    target_type: str
    name: str
    file_path: str
    selector: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def scan_skill_targets(repo_path: str | Path) -> list[TargetSpec]:
    """Discover Hermes skills under <repo>/skills/**/SKILL.md."""
    root = Path(repo_path)
    skills_dir = root / "skills"
    if not skills_dir.exists():
        return []

    targets: list[TargetSpec] = []
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        text = skill_file.read_text()
        frontmatter = _parse_frontmatter(text)
        name = frontmatter.get("name") or skill_file.parent.name
        description = frontmatter.get("description", "")
        rel_path = skill_file.relative_to(root).as_posix()
        targets.append(
            TargetSpec(
                target_type="skill",
                name=name,
                file_path=rel_path,
                selector=None,
                metadata={"description": description},
            )
        )
    return targets


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.strip().startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    fields: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"name", "description"}:
            fields[key] = value.strip().strip("'\"")
    return fields
