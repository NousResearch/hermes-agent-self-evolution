"""PromptSource Protocol — adapters that read and write named prompt sections.

Phase 3 integrates via in-place splice-and-restore (see
``HermesPromptSectionInstaller``), so the runtime override seam lives in
the installer, not here. The contract is deliberately just read + write: the
driver reads the baseline and persists/splices an evolved value, and nothing
more is shared across implementers. Enumeration (``list_sections`` →
``SectionDescriptor``) is a concrete convenience on ``HermesPromptSource`` for
a future ``--list-sections`` affordance, not part of the contract every
adapter must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SectionDescriptor:
    """Metadata about an evolvable prompt section.

    ``applicability`` is informational at design time; it's not used for
    runtime filtering in v1, but downstream joint-optimization work will
    consume it (e.g., model-family-targeted sections only get evaluated
    against that family).
    """

    name: str
    current_text: str
    source_path: Path
    applicability: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class PromptSource(Protocol):
    """Adapter contract for prompt-section evolution targets: read + write.

    Kept minimal on purpose — these are the only members the evolution driver
    exercises. Concrete adapters may offer more (e.g. ``HermesPromptSource``
    also enumerates sections), but those are not part of the shared contract.
    """

    def read(self, section_name: str) -> str:
        """Return the canonical baseline text of the named section."""
        ...

    def write(self, section_name: str, new_text: str) -> None:
        """Persist evolved text to the canonical source.

        Used both at deploy time and as the splice primitive the
        closed-loop installer drives during validation (the installer
        owns the backup/restore around the mutation).
        """
        ...
