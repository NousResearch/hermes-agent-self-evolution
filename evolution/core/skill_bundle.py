"""Skills as bundles, not single files.

Nearly half the installed library — 96 of 201 skills — ships supporting files
alongside ``SKILL.md``: reference documents, scripts, templates. That is
progressive disclosure, and it is deliberate: the entry file stays small and
points at detail the agent loads only when it needs it.

An optimizer that reads only ``SKILL.md`` is therefore blind to half of what
many skills actually are. It can delete a reference to a file it cannot see,
or duplicate that file's content inline and blow the size budget doing it.

This module loads the whole bundle, tracks which supporting files the entry
file references, and reports when an evolved variant breaks one of those
links — a structural regression no size or judge check would catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Files that are bundle payload rather than documentation of the bundle.
_IGNORED_NAMES = frozenset({".DS_Store", "Thumbs.db"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})

# Markdown links and bare relative paths that point at a sibling file.
_MD_LINK = re.compile(r"\[[^\]]*\]\(\s*(?!https?://|mailto:|#)([^)\s]+)")
_BARE_PATH = re.compile(
    r"(?<![\w/.-])((?:\./)?(?:[\w.-]+/)*[\w.-]+\.(?:md|py|sh|json|ya?ml|txt|csv|sql|js|ts))"
)

# How much of a supporting file to show the optimizer. Enough to know what the
# file is for; not so much that the context fills with reference material.
_EXCERPT_CHARS = 1_200


@dataclass
class SupportingFile:
    """One non-entry file in a skill directory."""

    path: Path
    relpath: str
    size: int
    referenced: bool = False

    def excerpt(self) -> str:
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if len(text) <= _EXCERPT_CHARS:
            return text
        return text[:_EXCERPT_CHARS] + "\n… (truncated)"


@dataclass
class SkillBundle:
    """A skill directory: the entry file plus everything it ships with."""

    name: str
    entry_path: Path
    entry_text: str
    supporting: list[SupportingFile] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.entry_path.parent

    @property
    def is_bundle(self) -> bool:
        """True when the skill is more than a lone ``SKILL.md``."""
        return bool(self.supporting)

    @property
    def referenced_files(self) -> list[SupportingFile]:
        return [f for f in self.supporting if f.referenced]

    @property
    def total_size(self) -> int:
        return len(self.entry_text) + sum(f.size for f in self.supporting)

    def describe(self) -> str:
        if not self.supporting:
            return "single file"
        linked = len(self.referenced_files)
        return (
            f"{len(self.supporting)} supporting file(s), {linked} referenced, "
            f"{self.total_size:,} chars total"
        )

    def context_for_optimizer(self, max_files: int = 6) -> str:
        """A compact description of the bundle to hand the optimizer.

        Only referenced files are included, and only as excerpts. The point is
        for the optimizer to know these files exist and roughly what is in
        them, so it stops inlining their content or dropping the links.
        """
        files = self.referenced_files[:max_files]
        if not files:
            return ""

        blocks = [
            "The skill ships these supporting files. They are loaded on demand "
            "and must keep working — preserve the references to them and do not "
            "inline their contents:",
        ]
        for f in files:
            blocks.append(f"\n--- {f.relpath} ({f.size:,} chars) ---\n{f.excerpt()}")
        return "\n".join(blocks)


def load_bundle(entry_path: Path, name: Optional[str] = None) -> SkillBundle:
    """Load a skill directory as a bundle."""
    entry_path = Path(entry_path)
    try:
        entry_text = entry_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        entry_text = ""

    root = entry_path.parent
    supporting: list[SupportingFile] = []

    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path == entry_path:
                continue
            if path.name in _IGNORED_NAMES or path.suffix in _IGNORED_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                rel = str(path.relative_to(root))
                size = path.stat().st_size
            except (OSError, ValueError):
                continue
            supporting.append(SupportingFile(path=path, relpath=rel, size=size))

    bundle = SkillBundle(
        name=name or root.name,
        entry_path=entry_path,
        entry_text=entry_text,
        supporting=supporting,
    )
    mark_references(bundle, entry_text)
    return bundle


def extract_references(text: str) -> set[str]:
    """Relative file paths the text points at, normalized without ``./``."""
    found: set[str] = set()
    for match in _MD_LINK.finditer(text):
        found.add(_normalize(match.group(1)))
    for match in _BARE_PATH.finditer(text):
        found.add(_normalize(match.group(1)))
    return {f for f in found if f}


def _normalize(raw: str) -> str:
    cleaned = raw.strip().strip("`'\"")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.split("#", 1)[0].strip()


def resolve_reference(bundle: SkillBundle, ref: str) -> Optional[str]:
    """The supporting file a reference identifies, or None if it matches none.

    Matching is by full relative path first, then by bare filename. The
    leniency is deliberate: the question these checks answer is "does the
    skill still point the agent at this document", and an optimizer that
    rewrites ``references/api.md`` to ``api.md`` has kept the pointer even
    though it changed the path. Treating that as a deletion would fail
    perfectly good rewrites.
    """
    if not ref:
        return None
    for f in bundle.supporting:
        if f.relpath == ref:
            return f.relpath
    basename = Path(ref).name
    for f in bundle.supporting:
        if Path(f.relpath).name == basename:
            return f.relpath
    return None


def _resolved_set(bundle: SkillBundle, text: str) -> set[str]:
    resolved = set()
    for ref in extract_references(text):
        target = resolve_reference(bundle, ref)
        if target:
            resolved.add(target)
    return resolved


def mark_references(bundle: SkillBundle, text: str) -> None:
    """Flag which supporting files the given text actually links to."""
    resolved = _resolved_set(bundle, text)
    for f in bundle.supporting:
        f.referenced = f.relpath in resolved


def broken_references(bundle: SkillBundle, evolved_text: str) -> list[str]:
    """Supporting files the baseline referenced that the evolved text dropped.

    Losing one of these is a real regression: the agent is told to consult a
    document the skill no longer points it at. It reads as a size *win* to
    every other check, which is exactly why it needs its own.
    """
    baseline = _resolved_set(bundle, bundle.entry_text)
    evolved = _resolved_set(bundle, evolved_text)
    return sorted(baseline - evolved)


def invented_references(bundle: SkillBundle, evolved_text: str) -> list[str]:
    """Paths the evolved text points at that do not exist in the bundle.

    An optimizer asked to be helpful will happily invent ``references/api.md``.
    Shipping that sends the agent after a file that was never written.
    """
    baseline_refs = extract_references(bundle.entry_text)

    # Only flag paths the evolution introduced — a baseline that already
    # pointed at a missing file is a pre-existing problem, not this run's.
    new_refs = extract_references(evolved_text) - baseline_refs
    return sorted(r for r in new_refs if resolve_reference(bundle, r) is None)
