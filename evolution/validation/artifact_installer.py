"""ArtifactInstaller Protocol — how the validator gets an artifact onto disk.

v1 only ships ``HermesToolDescriptionInstaller`` (splice into an
existing tool's description via ``HermesToolSource``). v2 will add
skill installers (drop a SKILL.md into the sandboxed HERMES_HOME's
skills dir) without changing ``ClosedLoopValidator``.
"""

from __future__ import annotations

import ast
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

from evolution.tools.hermes_source import HermesToolSource
from evolution.tools.tool_source import MCPManifestSource, ToolManifest


class ArtifactInstaller(Protocol):
    """Install (and uninstall) a candidate artifact in the agent's environment."""

    target_path: Path
    """The single file the installer mutates. Used by the validator for
    backup + flock + checksum book-keeping."""

    def install(self, artifact_source: Path) -> str:
        """Apply ``artifact_source`` to ``target_path``. Returns the sha256
        of ``target_path``'s on-disk bytes after installation so the
        validator can verify the file wasn't mutated between tasks."""
        ...


class HermesToolDescriptionInstaller:
    """Splice an evolved tool description into a Hermes ``*_SCHEMA`` file.

    The artifact source is the full evolved tool-module file (same
    layout as the baseline). We reuse ``HermesToolSource`` to do the
    AST parse + byte-precise splice; this class only manages the
    target_path bookkeeping and the post-install checksum.
    """

    def __init__(self, hermes_repo: Path, tool_name: str) -> None:
        self.hermes_repo = hermes_repo
        self.tool_name = tool_name
        self._source = HermesToolSource(hermes_repo / "tools")
        self.target_path = self._locate_target()

    def _locate_target(self) -> Path:
        manifest = self._source.find_manifest(self.hermes_repo / "tools")
        if manifest is None:
            raise FileNotFoundError(
                f"No Hermes tools manifest found under {self.hermes_repo / 'tools'}"
            )
        entry = manifest.find_tool(self.tool_name)
        if entry.source_location is None:
            raise ValueError(
                f"Tool {self.tool_name!r} has no statically-resolved source location"
            )
        return Path(entry.source_location[0])

    def install(self, artifact_source: Path) -> str:
        """Splice the target tool's description from ``artifact_source`` into
        the live install. Always splices against the LIVE manifest's
        source_location (the live target_path), so the description from
        the evolved artifact is the only thing carried over — the byte
        offsets come from re-parsing the current on-disk file.
        """
        new_description = self._extract_description(artifact_source)
        live_manifest = self._source.find_manifest(self.hermes_repo / "tools")
        if live_manifest is None:
            raise FileNotFoundError(
                "live Hermes manifest disappeared between init and install"
            )
        self._source.apply_evolved(
            source_path=self.target_path,
            evolved_manifest=live_manifest,
            target_tool=self.tool_name,
            new_description=new_description,
        )
        return sha256_of(self.target_path)

    def _extract_description(self, artifact_source: Path) -> str:
        """Return the description string for ``self.tool_name`` from
        ``artifact_source``. Dispatches on suffix so the installer can
        consume either a Hermes tool-module .py file (e.g., a
        hand-edited baseline) or an MCP-shape manifest .json (the
        output ``evolve_tool`` produces and that ``--benchmark-cmd``
        threads through as ``EVOLVED_PATH`` / ``BASELINE_PATH``).
        """
        if artifact_source.suffix == ".json":
            manifest = MCPManifestSource(artifact_source.parent).find_manifest(artifact_source)
            if manifest is None:
                raise ValueError(f"Could not parse {artifact_source} as MCP manifest JSON")
            return manifest.find_tool(self.tool_name).description

        # Default: Hermes tool-module .py file. The parse uses
        # HermesToolSource pointed at a temp-dir root holding only this
        # file; we extract the description before the temp dir is cleaned
        # up so source_location-bound reads against the temp dir can't
        # happen later.
        with tempfile.TemporaryDirectory(prefix="cl_install_") as tmp:
            tmp_root = Path(tmp)
            staged = tmp_root / artifact_source.name
            staged.write_bytes(artifact_source.read_bytes())
            manifest = HermesToolSource(tmp_root).find_manifest(tmp_root)
            if manifest is None:
                raise ValueError(
                    f"Could not parse {artifact_source} as a Hermes tool module"
                )
            return manifest.find_tool(self.tool_name).description


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic file write — same primitive HermesToolSource uses internally.
    Crash mid-write leaves the original file intact (or absent), never
    half-written.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def verify_python_parses(path: Path) -> None:
    """Raise SyntaxError if ``path`` doesn't parse as Python.

    Used to validate ``.cl_backup`` before trusting it for restore — a
    truncated backup from a SIGKILL-during-backup-write must not be
    silently restored over the original.
    """
    ast.parse(path.read_bytes())
