"""Auto-mode write-back: safely overwrite a live skill with an evolved version.

Write-back is only invoked when:
  - mode == "auto"
  - decision.auto_merge is True
  - constraints all passed
  - improvement >= min_improvement
  - no regression

For safety, a timestamped backup is written alongside the live skill before
overwrite. If the write fails mid-way, the backup can be used to roll back.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class WriteBackResult:
    """Outcome of an auto-merge write-back."""
    merged: bool                 # True if the live skill was overwritten
    live_path: Path              # Path to the skill that was (or would be) overwritten
    backup_path: Optional[Path]  # Path to the .bak file (None if merged=False)
    reason: str                  # Human-readable explanation

    def to_dict(self) -> dict:
        d = asdict(self)
        d["live_path"] = str(self.live_path)
        d["backup_path"] = str(self.backup_path) if self.backup_path else None
        return d


def write_back_skill(
    live_path: Path,
    evolved_text: str,
    *,
    mode: str,
    auto_merge: bool,
    timestamp: Optional[str] = None,
    backup_dir: Optional[Path] = None,
) -> WriteBackResult:
    """Atomically overwrite a live skill with evolved content.

    Args:
        live_path: Path to the live SKILL.md to overwrite.
        evolved_text: Full evolved skill text (frontmatter + body).
        mode: "auto" or "propose". Only "auto" triggers write-back.
        auto_merge: The gate decision. False => no write.
        timestamp: YYYYMMDD_HHMMSS for backup naming. Auto-generated if None.
        backup_dir: Optional directory to store backups. Defaults to
                    `live_path.parent / ".backups"`.

    Returns:
        WriteBackResult describing what happened.

    Raises:
        FileNotFoundError: if live_path does not exist (nothing to back up).
    """
    if mode != "auto":
        return WriteBackResult(
            merged=False,
            live_path=live_path,
            backup_path=None,
            reason=f"mode={mode!r} is not 'auto' — no write-back",
        )
    if not auto_merge:
        return WriteBackResult(
            merged=False,
            live_path=live_path,
            backup_path=None,
            reason="gate rejected auto-merge — no write-back",
        )
    if not live_path.exists():
        raise FileNotFoundError(
            f"Cannot write back: live_path does not exist: {live_path}"
        )

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if backup_dir is None:
        backup_dir = live_path.parent / ".backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{live_path.name}.{ts}.bak"

    # 1. Back up current content
    shutil.copy2(live_path, backup_path)

    # 2. Atomic overwrite: write to temp, rename into place.
    tmp_path = live_path.with_suffix(live_path.suffix + ".tmp")
    tmp_path.write_text(evolved_text)
    tmp_path.replace(live_path)  # atomic on POSIX

    return WriteBackResult(
        merged=True,
        live_path=live_path,
        backup_path=backup_path,
        reason=f"mode=auto + gate approved — wrote {len(evolved_text)} chars, backup at {backup_path.name}",
    )
