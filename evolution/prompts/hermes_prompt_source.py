"""HermesPromptSource — read/write named string constants in Hermes prompt_builder.py.

Walks ``agent/prompt_builder.py`` for top-level ``NAME = "..."`` (or
concatenated-string) assignments. v1 supports string-typed constants
only; dict-typed constants (like ``PLATFORM_HINTS``) raise KeyError on
read.
"""

from __future__ import annotations

import ast
import logging
import os
import shutil
import tempfile
from pathlib import Path

from evolution.prompts.prompt_source import SectionDescriptor

logger = logging.getLogger(__name__)


class HermesPromptSource:
    """Read/write named string constants in Hermes prompt_builder.py."""

    name = "hermes_prompt_source"

    def __init__(self, hermes_repo: Path) -> None:
        self.hermes_repo = Path(hermes_repo)
        self.prompt_builder_path = self.hermes_repo / "agent" / "prompt_builder.py"
        if not self.prompt_builder_path.is_file():
            raise FileNotFoundError(
                f"prompt_builder.py not found at {self.prompt_builder_path}"
            )

    def read(self, section_name: str) -> str:
        constants = self._parse_string_constants()
        if section_name not in constants:
            raise KeyError(
                f"section {section_name!r} not found in {self.prompt_builder_path} "
                f"(v1 only supports top-level string-typed constants). "
                f"Available: {sorted(constants)}"
            )
        return constants[section_name][0]

    def list_sections(self) -> list[SectionDescriptor]:
        constants = self._parse_string_constants()
        return [
            SectionDescriptor(
                name=name,
                current_text=text,
                source_path=self.prompt_builder_path,
            )
            for name, (text, _node) in sorted(constants.items())
        ]

    def _parse_string_constants(self) -> dict[str, tuple[str, ast.Constant]]:
        """Return ``{name: (value, value_ast_node)}`` for every top-level
        string-typed assignment in prompt_builder.py.

        Concatenated-string forms like ``X = ("a" "b" "c")`` are folded to
        a single ``ast.Constant`` by the parser, so they read back as one
        string. The AST node is retained so ``write`` can splice by byte
        offset.
        """
        source = self.prompt_builder_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self.prompt_builder_path))
        out: dict[str, tuple[str, ast.Constant]] = {}
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                out[target.id] = (value.value, value)
        return out

    def write(self, section_name: str, new_text: str) -> None:
        """Splice ``new_text`` into the named constant in place.

        Uses ``repr()`` for the replacement literal so the new text
        round-trips byte-equal regardless of embedded newlines, quotes,
        or backslashes. Other constants are left verbatim.

        The write is atomic (tempfile + ``os.replace``) and guarded: the
        new bytes must parse as Python before the original is replaced.
        A botched splice (only possible if AST extraction were wrong)
        raises and leaves ``prompt_builder.py`` untouched, rather than
        leaving the user's Hermes unstartable.
        """
        constants = self._parse_string_constants()
        if section_name not in constants:
            raise KeyError(
                f"section {section_name!r} not found in {self.prompt_builder_path}"
            )
        _, value_node = constants[section_name]
        data = self.prompt_builder_path.read_bytes()
        start_offset = _byte_offset(data, value_node.lineno, value_node.col_offset)
        end_offset = _byte_offset(
            data, value_node.end_lineno, value_node.end_col_offset
        )
        replacement = repr(new_text).encode("utf-8")
        new_bytes = data[:start_offset] + replacement + data[end_offset:]

        try:
            ast.parse(new_bytes, filename=str(self.prompt_builder_path))
        except SyntaxError as exc:
            raise RuntimeError(
                f"Refusing to write {self.prompt_builder_path}: spliced output "
                f"would not parse as Python ({exc}). Original file untouched."
            ) from exc

        self._atomic_write_bytes(self.prompt_builder_path, new_bytes)

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            shutil.copymode(path, tmp_path)
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


def _byte_offset(data: bytes, lineno: int, col_offset: int) -> int:
    """Convert an AST position (1-based line, 0-based byte column) to an
    absolute byte offset into ``data``."""
    lines = data.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        raise ValueError(f"lineno {lineno} out of range [1, {len(lines)}]")
    return sum(len(line) for line in lines[: lineno - 1]) + col_offset
