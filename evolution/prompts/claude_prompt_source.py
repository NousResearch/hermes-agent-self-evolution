"""ClaudeCodePromptSource — read/write a sentinel-delimited region in a CLAUDE.md.

Implements the PromptSource contract (read/write) for the Claude Code backend. The
evolvable region is delimited by ``<!-- evolve:NAME start -->`` / ``<!-- evolve:NAME
end -->`` so GEPA evolves only that block and the user's hand-written CLAUDE.md content
outside it survives byte-for-byte. Used to seed GEPA (read the baseline region) and to
deploy the evolved region on ``--apply``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic write — crash mid-write leaves the original intact, never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _open(section_name: str) -> str:
    return f"<!-- evolve:{section_name} start -->"


def _close(section_name: str) -> str:
    return f"<!-- evolve:{section_name} end -->"


class ClaudeCodePromptSource:
    """Read/write the named evolve-region in a CLAUDE.md file."""

    name = "claude_prompt_source"

    def __init__(self, claude_md_path: Path) -> None:
        self.claude_md_path = Path(claude_md_path)

    def _locate(self, text: str, section_name: str) -> tuple[int, int]:
        """Return (inner_start, inner_end) byte offsets of the region body.

        Raises KeyError if markers are absent, ValueError if malformed/duplicated.
        """
        open_m, close_m = _open(section_name), _close(section_name)
        if text.count(open_m) == 0 or text.count(close_m) == 0:
            raise KeyError(
                f"evolve region {section_name!r} not found in {self.claude_md_path}"
            )
        if text.count(open_m) > 1 or text.count(close_m) > 1:
            raise ValueError(
                f"evolve region {section_name!r} appears multiple times in "
                f"{self.claude_md_path}"
            )
        inner_start = text.index(open_m) + len(open_m)
        inner_end = text.index(close_m)
        if inner_end < inner_start:
            raise ValueError(
                f"closing marker precedes opening marker for {section_name!r}"
            )
        return inner_start, inner_end

    def read(self, section_name: str) -> str:
        if not self.claude_md_path.is_file():
            raise FileNotFoundError(
                f"CLAUDE.md not found at {self.claude_md_path}. Point --claude-md at an "
                f"existing file containing an '<!-- evolve:{section_name} start -->' region, "
                f"or pass --baseline-override-file to seed a new region."
            )
        text = self.claude_md_path.read_text(encoding="utf-8")
        start, end = self._locate(text, section_name)
        return text[start:end].strip()

    def write(self, section_name: str, new_text: str) -> None:
        """Replace the region body with ``new_text``.

        When the file or markers are absent, append a fresh delimited block at EOF
        (leaving any existing content intact), so first-time evolution targets a
        CLAUDE.md that doesn't yet carry the region.
        """
        text = (
            self.claude_md_path.read_text(encoding="utf-8")
            if self.claude_md_path.exists()
            else ""
        )
        open_m, close_m = _open(section_name), _close(section_name)
        if open_m in text and close_m in text:
            start, end = self._locate(text, section_name)
            updated = text[:start] + f"\n{new_text}\n" + text[end:]
        else:
            block = f"{open_m}\n{new_text}\n{close_m}\n"
            updated = text + ("\n" if text and not text.endswith("\n") else "") + block
        _atomic_write_text(self.claude_md_path, updated)
