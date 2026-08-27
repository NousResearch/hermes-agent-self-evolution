"""Hermes install layout discovery.

Every path the evolution system reads out of a Hermes install is resolved
here, and none of it goes through ``Path.home()``.

That distinction is not cosmetic. Evolution runs inside the ``hermes``
container via ``docker exec``, where ``HOME`` is ``/root`` but the Hermes data
directory is bind-mounted somewhere else entirely (``/opt/data`` on the
reference deployment). Importers that built their paths from ``Path.home()``
silently resolved to directories that do not exist there, mined zero examples,
and took the whole run down with a "no relevant examples" exit — without ever
reporting that the source itself was missing.

Resolution order for the data root:

1. ``HERMES_DATA_DIR`` — explicit override, wins over everything
2. ``HERMES_HOME``     — the variable the Hermes CLI itself honors
3. ``~/.hermes``       — the standard single-user install

A root only counts as a Hermes install if it actually looks like one, so a
stale env var pointing at an empty directory is reported rather than silently
producing empty datasets.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Files/directories that mark a directory as a real Hermes data root. A root
# needs at least one; a fresh install has config.yaml before it has state.db.
_ROOT_MARKERS = ("state.db", "config.yaml", "skills", "profiles")


class HermesInstallNotFound(RuntimeError):
    """Raised when no Hermes data root can be located or the one named is empty."""


@dataclass(frozen=True)
class HermesProfile:
    """One Hermes profile (``ali``, ``usman``, …) or the root default profile."""

    name: str
    root: Path

    @property
    def state_db(self) -> Path:
        """Conversation + session store. The real session list lives here."""
        return self.root / "state.db"

    @property
    def verification_db(self) -> Path:
        """Verified command outcomes — exit codes from real test/build runs."""
        return self.root / "verification_evidence.db"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    def has_state_db(self) -> bool:
        return self.state_db.is_file()


@dataclass(frozen=True)
class HermesInstall:
    """A resolved Hermes data directory and everything evolution reads from it."""

    root: Path
    source: str  # which resolution rule produced this root, for error messages

    # ── shared (non per-profile) stores ──────────────────────────────────

    @property
    def cron_dir(self) -> Path:
        return self.root / "cron"

    @property
    def cron_executions_db(self) -> Path:
        """Scheduled-job outcomes: one row per execution with status + error."""
        return self.cron_dir / "executions.db"

    @property
    def cron_jobs_json(self) -> Path:
        """Job definitions. Binds each scheduled job to the skills it uses."""
        return self.cron_dir / "jobs.json"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    # ── profiles ─────────────────────────────────────────────────────────

    def profiles(self) -> list[HermesProfile]:
        """All profiles, newest-data-first is not guaranteed — sorted by name.

        A root with no ``profiles/`` directory is a single-profile install; the
        root itself is returned as the ``default`` profile so callers never
        need to special-case the two layouts.
        """
        found: list[HermesProfile] = []
        if self.profiles_dir.is_dir():
            for entry in sorted(self.profiles_dir.iterdir()):
                if entry.is_dir():
                    found.append(HermesProfile(name=entry.name, root=entry))
        if not found:
            found.append(HermesProfile(name="default", root=self.root))
        return found

    def profile(self, name: Optional[str] = None) -> HermesProfile:
        """Return one profile by name, or the default when name is None."""
        available = self.profiles()
        if name is None:
            return available[0]
        for prof in available:
            if prof.name == name:
                return prof
        names = ", ".join(p.name for p in available) or "(none)"
        raise HermesInstallNotFound(
            f"Hermes profile {name!r} not found under {self.root}. Available: {names}"
        )

    def profiles_with_state(self) -> list[HermesProfile]:
        """Only profiles that actually have a state database to read."""
        return [p for p in self.profiles() if p.has_state_db()]


def _looks_like_hermes_root(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in _ROOT_MARKERS)


def _platform_candidates() -> list[tuple[Path, str]]:
    """Default install locations, in the order Hermes itself would use them.

    ``~/.hermes`` is the common case, but a Windows install lands under
    ``%LOCALAPPDATA%`` and an XDG-configured Linux install under
    ``$XDG_DATA_HOME``. Checking only the POSIX default made discovery fail on
    those machines with a message that pointed at the wrong directory.
    Locations contributed by NousResearch/hermes-agent-self-evolution#178.
    """
    home = Path.home()
    found: list[tuple[Path, str]] = [(home / ".hermes", "~/.hermes")]

    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        found.append((Path(local_appdata) / "hermes", "%LOCALAPPDATA%/hermes"))

    xdg_data = os.getenv("XDG_DATA_HOME")
    if xdg_data:
        found.append((Path(xdg_data) / "hermes", "$XDG_DATA_HOME/hermes"))

    if sys.platform == "darwin":
        found.append(
            (home / "Library" / "Application Support" / "hermes",
             "~/Library/Application Support/hermes")
        )

    return found


def find_hermes_install(explicit: Optional[str | Path] = None) -> HermesInstall:
    """Locate the Hermes data directory.

    Args:
        explicit: A path supplied by the caller (``--hermes-data-dir``). When
            given it is used as-is and never falls back, so a typo surfaces as
            an error instead of silently reading a different install.

    Raises:
        HermesInstallNotFound: When nothing resolves, or the resolved directory
            does not look like a Hermes install.
    """
    candidates: list[tuple[Path, str]] = []

    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise HermesInstallNotFound(f"Hermes data dir does not exist: {path}")
        if not _looks_like_hermes_root(path):
            raise HermesInstallNotFound(
                f"{path} exists but does not look like a Hermes data dir "
                f"(expected one of: {', '.join(_ROOT_MARKERS)})"
            )
        return HermesInstall(root=path, source="explicit")

    for env_var in ("HERMES_DATA_DIR", "HERMES_HOME"):
        raw = os.getenv(env_var)
        if raw:
            candidates.append((Path(raw).expanduser(), f"${env_var}"))

    candidates.extend(_platform_candidates())

    tried = []
    for path, source in candidates:
        if _looks_like_hermes_root(path):
            return HermesInstall(root=path, source=source)
        tried.append(f"{source} -> {path}")

    raise HermesInstallNotFound(
        "Cannot locate a Hermes data directory. Tried:\n  "
        + "\n  ".join(tried)
        + "\nSet HERMES_DATA_DIR to the directory containing state.db."
    )


def try_find_hermes_install(
    explicit: Optional[str | Path] = None,
) -> Optional[HermesInstall]:
    """Non-raising variant for callers that treat a missing install as optional."""
    try:
        return find_hermes_install(explicit)
    except HermesInstallNotFound:
        return None
