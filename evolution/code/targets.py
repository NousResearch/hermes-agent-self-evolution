"""Choosing which code to evolve, and what to gate it against.

Phase 4 has a target-selection problem the earlier phases do not. A skill is a
named artifact you ask for by name; a "tool implementation" is a region of a
large codebase, and pointing an optimizer at the wrong file wastes an
expensive run.

Targets are therefore explicit by default — the operator names the files. The
suggestion path is offered as ranked evidence, not as an automatic choice,
because the mapping from an observed failure to the source file responsible is
a heuristic and should be labelled as one rather than dressed up as inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from evolution.code.admission import RecordedCommandCheck
from evolution.core.hermes_paths import HermesInstall
from evolution.core.outcome_signals import VerificationSignal
from evolution.core.state_db import tool_usage_histogram

# Files that are never sensible evolution targets, whatever the signals say.
_NEVER_EVOLVE = (
    "tests/",
    "test_",
    "conftest.py",
    "setup.py",
    "__init__.py",
    "migrations/",
)

# A target file above this size is too large for a mutator to rewrite coherently
# and too large for a human to review as a diff.
MAX_TARGET_CHARS = 60_000


@dataclass
class CodeTarget:
    """One or more files to evolve together, and why."""

    paths: list[str]
    label: str
    rationale: str = ""
    evidence: dict = field(default_factory=dict)

    def total_chars(self, repo: Path) -> int:
        total = 0
        for rel in self.paths:
            try:
                total += len((repo / rel).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        return total

    def read(self, repo: Path) -> dict[str, str]:
        """The baseline contents, as the organism's file map."""
        out: dict[str, str] = {}
        for rel in self.paths:
            out[rel] = (repo / rel).read_text(encoding="utf-8", errors="replace")
        return out

    def describe(self, repo: Optional[Path] = None) -> str:
        size = f", {self.total_chars(repo):,} chars" if repo else ""
        return f"{self.label} ({len(self.paths)} file(s){size}) — {self.rationale}"


class TargetError(ValueError):
    """A requested target cannot be evolved, with the reason stated."""


def resolve_targets(
    hermes_repo: Path,
    paths: Sequence[str],
    label: str = "",
    allow_large: bool = False,
) -> CodeTarget:
    """Validate operator-specified paths into a target.

    Every rejection names the file and the reason. A silent skip here would
    mean an operator asking to evolve three files and getting two, with the
    report still reading as a success.
    """
    hermes_repo = Path(hermes_repo).resolve()
    if not paths:
        raise TargetError("no target paths given")

    resolved: list[str] = []
    for raw in paths:
        candidate = (hermes_repo / raw).resolve()

        try:
            rel = str(candidate.relative_to(hermes_repo))
        except ValueError:
            raise TargetError(f"{raw} is outside the repo {hermes_repo}")

        if not candidate.is_file():
            raise TargetError(f"{rel} is not a file in {hermes_repo}")

        if any(part in rel for part in _NEVER_EVOLVE):
            raise TargetError(
                f"{rel} is excluded from evolution (tests, packaging and "
                "migrations must be written by humans)"
            )

        resolved.append(rel)

    target = CodeTarget(
        paths=resolved,
        label=label or Path(resolved[0]).stem,
        rationale="explicitly requested",
    )

    size = target.total_chars(hermes_repo)
    if size > MAX_TARGET_CHARS and not allow_large:
        raise TargetError(
            f"target is {size:,} chars, over the {MAX_TARGET_CHARS:,} limit. "
            "A mutator cannot rewrite this coherently and a reviewer cannot "
            "read the diff. Narrow the target, or pass --allow-large."
        )

    return target


def suggest_targets(
    install: HermesInstall,
    hermes_repo: Path,
    limit: int = 10,
) -> list[CodeTarget]:
    """Rank plausible targets from production signals.

    This is evidence for a human to act on, not an automatic selection. The
    ranking combines two things the install actually records: how heavily a
    tool is used (leverage — a fix to a hot path is worth more) and how often
    verification of work involving it failed (need).

    The tool-name-to-file mapping is a filename heuristic. It is reported as
    such, and ``resolve_targets`` still has to approve whatever a human picks
    from this list.
    """
    hermes_repo = Path(hermes_repo)
    histogram = tool_usage_histogram(install)
    if not histogram:
        return []

    failures = _failure_terms(install)
    suggestions: list[CodeTarget] = []

    for tool_name, uses in list(histogram.items())[: limit * 3]:
        path = _guess_tool_file(hermes_repo, tool_name)
        if path is None:
            continue

        failure_hits = sum(1 for term in failures if tool_name in term)
        suggestions.append(
            CodeTarget(
                paths=[path],
                label=tool_name,
                rationale=(
                    f"{uses:,} calls in production"
                    + (f", {failure_hits} failed verifications mention it" if failure_hits else "")
                    + " (file matched by name — confirm before evolving)"
                ),
                evidence={"uses": uses, "failure_mentions": failure_hits},
            )
        )
        if len(suggestions) >= limit:
            break

    suggestions.sort(
        key=lambda t: (t.evidence.get("failure_mentions", 0), t.evidence.get("uses", 0)),
        reverse=True,
    )
    return suggestions


def _failure_terms(install: HermesInstall) -> list[str]:
    """Commands from verification events that did not pass."""
    return [
        f"{ev.kind} {ev.command}".lower()
        for ev in VerificationSignal(install).events()
        if not ev.passed
    ]


_TOOL_FILE_HINTS = ("agent", "tools", "hermes_cli")


def _guess_tool_file(hermes_repo: Path, tool_name: str) -> Optional[str]:
    """Best-effort file for a tool, by filename match. Heuristic by design."""
    if tool_name.startswith("mcp__"):
        return None  # MCP tools live outside the repo entirely.

    stem = re.sub(r"[^a-z0-9_]", "", tool_name.lower())
    if not stem:
        return None

    for hint in _TOOL_FILE_HINTS:
        base = hermes_repo / hint
        if not base.is_dir():
            continue
        for candidate in sorted(base.rglob(f"{stem}.py")):
            try:
                rel = str(candidate.relative_to(hermes_repo))
            except ValueError:
                continue
            if not any(part in rel for part in _NEVER_EVOLVE):
                return rel
    return None


def recorded_checks_for(
    install: HermesInstall,
    limit: int = 8,
    only_passing: bool = True,
) -> list[RecordedCommandCheck]:
    """Turn recorded verification events into replayable gate checks.

    Only events that *passed* are used by default. A command that was already
    failing before the mutation tells us nothing about the mutation, and
    admitting on "it still fails the same way" is not a gate.

    Commands are de-duplicated: the same ``pytest tests/`` recorded forty times
    is one check, not forty.
    """
    seen: set[str] = set()
    checks: list[RecordedCommandCheck] = []

    for event in VerificationSignal(install).events():
        if only_passing and not event.passed:
            continue
        command = (event.command or "").strip()
        if not command or command in seen:
            continue
        if not _is_replayable(command):
            continue
        seen.add(command)
        checks.append(
            RecordedCommandCheck(
                name=f"recorded:{event.kind or 'check'}:{command[:40]}",
                command=command,
                expected_exit=0,
            )
        )
        if len(checks) >= limit:
            break

    return checks


# Commands that would mutate state, reach the network, or touch the operator's
# machine rather than the sandbox. Replaying these is not verification.
_UNSAFE_REPLAY = re.compile(
    r"\b(rm|mv|cp|dd|mkfs|shutdown|reboot|kill|pkill|systemctl|docker|kubectl|"
    r"git\s+(push|commit|reset|clean)|curl|wget|ssh|scp|rsync|npm\s+publish|"
    r"pip\s+install|apt|brew|sudo)\b",
    re.IGNORECASE,
)


def _is_replayable(command: str) -> bool:
    """Whether a recorded command is safe to re-run inside the sandbox."""
    if _UNSAFE_REPLAY.search(command):
        return False
    # Shell metacharacters that chain into something else.
    if any(tok in command for tok in ("&&", "||", ";", "|", ">", "<", "`", "$(")):
        return False
    return True
