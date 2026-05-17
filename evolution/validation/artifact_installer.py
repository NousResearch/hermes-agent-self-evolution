"""ArtifactInstaller Protocol — how the validator gets an artifact onto disk.

Two concrete installers ship: ``HermesToolDescriptionInstaller`` (splice
an evolved description into a Hermes tool-module ``*_SCHEMA`` file in
place) and ``SkillFileInstaller`` (write an evolved SKILL.md into a
caller-provided writable workdir, decoupled from the user's actual
HERMES_HOME / read-only plugin cache).

Installers that need the runner to stage extra state into its per-task
sandbox (only ``SkillFileInstaller`` today, which needs the candidate
SKILL.md visible to ``hermes -z``) expose an optional ``skills_src``
attribute the validator threads through ``TaskRunContext``.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Protocol

from evolution.tools.hermes_source import HermesToolSource
from evolution.tools.tool_source import ToolManifest


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

    def verify_backup(self, backup_path: Path) -> None:
        """Validate the backup before trusting it for restore.

        Default behavior for Python-source artifacts: raise SyntaxError if
        the backup doesn't parse. For non-Python artifacts (skills), the
        installer overrides with a format-appropriate check (e.g., UTF-8
        decodability + non-empty)."""
        ...


class HermesToolDescriptionInstaller:
    """Splice an evolved tool description into a Hermes ``*_SCHEMA`` file.

    The artifact source is the full evolved tool-module file (same
    layout as the baseline). We reuse ``HermesToolSource`` to do the
    AST parse + byte-precise splice; this class only manages the
    target_path bookkeeping and the post-install checksum.

    Constraint: the target tool's schema must be declared as a static
    dict literal (``NAME_SCHEMA = {...}``). Schemas built by a
    function call (e.g. ``EXECUTE_CODE_SCHEMA = build_schema()``) end
    up in ``HermesToolSource.dropped_tools`` and aren't installable —
    closed-loop validation for those tools needs the upstream module
    refactored to a static literal first.
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

    def verify_backup(self, backup_path: Path) -> None:
        verify_python_parses(backup_path)

    def _extract_description(self, artifact_source: Path) -> str:
        """Return the description string for ``self.tool_name`` from
        ``artifact_source``. Dispatches on suffix so the installer can
        consume either a Hermes tool-module .py file (e.g., a
        hand-edited baseline) or an MCP-shape manifest .json (the
        output ``evolve_tool`` produces and that ``--benchmark-cmd``
        threads through as ``EVOLVED_PATH`` / ``BASELINE_PATH``).
        """
        if artifact_source.suffix == ".json":
            # MCPManifestSource.find_manifest double-resolves relative paths
            # against its root; from_json_file takes the path as given.
            manifest = ToolManifest.from_json_file(artifact_source)
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


class SkillFileInstaller:
    """Write an evolved SKILL.md into a writable workdir for closed-loop validation.

    The user's actual skill may live in a read-only location (the Claude
    Code plugin cache, a system-installed skill bundle, or a user
    HERMES_HOME we don't want to mutate). The installer copies the
    entire baseline skill directory once at construction into a
    caller-owned ``workdir``, then ``install()`` writes candidate text
    over the resulting target SKILL.md. The original location is never
    touched.

    ``skills_src`` is the directory the runner copies into its per-task
    sandbox so ``hermes -z`` discovers the candidate skill. Exposed
    here so the validator can thread it through ``TaskRunContext``
    without a Hermes-specific code path.
    """

    def __init__(
        self,
        *,
        skill_source_path: Path,
        skill_name: str,
        workdir: Path,
    ) -> None:
        if not skill_source_path.is_file():
            raise FileNotFoundError(
                f"skill_source_path not found: {skill_source_path}"
            )
        if not workdir.is_dir():
            raise NotADirectoryError(
                f"workdir not found or not a directory: {workdir}"
            )
        self.skill_name = skill_name
        self.workdir = workdir
        self.skills_src: Path = workdir / "skills"
        skill_dest_dir = self.skills_src / skill_name
        skill_dest_dir.parent.mkdir(parents=True, exist_ok=True)
        source_dir = skill_source_path.parent
        # Copy the entire skill directory (SKILL.md + any sibling files
        # the skill references) so the candidate has the same surrounding
        # context as the baseline.
        shutil.copytree(source_dir, skill_dest_dir, dirs_exist_ok=False)
        self.target_path: Path = skill_dest_dir / skill_source_path.name

    def install(self, artifact_source: Path) -> str:
        """Atomically overwrite ``target_path`` with ``artifact_source`` contents.

        The artifact source is the candidate SKILL.md as written by the
        cache's ``artifact_writer``. We just copy bytes — no parse, no
        splice — since for skills the whole file is the artifact.
        """
        atomic_write_bytes(self.target_path, artifact_source.read_bytes())
        return sha256_of(self.target_path)

    def verify_backup(self, backup_path: Path) -> None:
        """Skills are UTF-8 text; reject empty or non-UTF-8 backups."""
        data = backup_path.read_bytes()
        if not data:
            raise ValueError(f"skill backup at {backup_path} is empty")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"skill backup at {backup_path} is not valid UTF-8: {exc}"
            ) from exc


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
