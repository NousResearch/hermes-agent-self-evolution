"""CLI for verifier-gated Phase 4 patch search.

The proposer is an external process.  It receives public JSON context on stdin
and returns patch proposals as JSON lines.  Hidden commands are retained by this
process and are never serialized into proposer context.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import click

from evolution.code.verified_search import (
    CommandCheck,
    ExternalCLIProposer,
    GateTask,
    VerifiedPatchSearch,
    compare_adaptive_frozen,
)


def _split_argv(value: str) -> tuple[str, ...]:
    """Split a command string while preserving quoted Windows paths."""

    stripped = value.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise click.BadParameter("JSON command must be an array of strings")
        return tuple(parsed)
    tokens = shlex.split(stripped, posix=os.name != "nt")
    cleaned = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
            token = token[1:-1]
        cleaned.append(token)
    return tuple(cleaned)


def _parse_check(value: str, default_name: str) -> CommandCheck:
    """Parse ``name::command`` or a bare command without invoking a shell."""

    if "::" in value:
        name, command = value.split("::", 1)
    else:
        name, command = default_name, value
    argv = _split_argv(command)
    if not argv:
        raise click.BadParameter("check command is empty")
    return CommandCheck(name=name.strip() or default_name, argv=argv)


def _resolve_local_command_paths(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Make existing relative command paths survive proposer cwd isolation."""

    resolved = []
    current = Path.cwd()
    for token in argv:
        path = Path(token)
        candidate = current / path
        if not path.is_absolute() and candidate.exists():
            resolved.append(str(candidate.resolve()))
        else:
            resolved.append(token)
    return tuple(resolved)


@click.command()
@click.option(
    "--repo",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
    help="Stable repository to copy for candidate evaluation.",
)
@click.option("--task-id", required=True, help="Audit identifier for this repair task.")
@click.option(
    "--proposer-command",
    required=True,
    help="External JSON-lines proposer command (no shell operators).",
)
@click.option(
    "--visible",
    multiple=True,
    required=True,
    help="Public check as name::command; repeatable.",
)
@click.option(
    "--sealed",
    multiple=True,
    required=True,
    help="Verifier-only check as name::command; repeatable.",
)
@click.option("--full-suite", required=True, help="Required full-suite command.")
@click.option(
    "--allow",
    "allowed_paths",
    multiple=True,
    required=True,
    help="Allowed patch path glob; repeatable.",
)
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--cycles", default=2, show_default=True, type=click.IntRange(min=1))
@click.option("--budget", default=4, show_default=True, type=click.IntRange(min=1))
@click.option("--max-files", default=4, show_default=True, type=click.IntRange(min=1))
@click.option("--max-lines", default=200, show_default=True, type=click.IntRange(min=1))
@click.option("--min-sealed-delta", default=0.0, show_default=True, type=float)
@click.option("--seed", default=20260621, show_default=True, type=int)
@click.option(
    "--compare-frozen",
    is_flag=True,
    help="Run an equal-budget frozen control in a separate run directory.",
)
def main(
    repo: Path,
    task_id: str,
    proposer_command: str,
    visible: tuple[str, ...],
    sealed: tuple[str, ...],
    full_suite: str,
    allowed_paths: tuple[str, ...],
    run_dir: Path,
    cycles: int,
    budget: int,
    max_files: int,
    max_lines: int,
    min_sealed_delta: float,
    seed: int,
    compare_frozen: bool,
) -> None:
    task = GateTask(
        task_id=task_id,
        visible_checks=tuple(
            _parse_check(value, f"visible-{index}")
            for index, value in enumerate(visible, 1)
        ),
        sealed_checks=tuple(
            _parse_check(value, f"sealed-{index}")
            for index, value in enumerate(sealed, 1)
        ),
        full_suite_check=_parse_check(full_suite, "full-suite"),
        allowed_paths=tuple(allowed_paths),
        max_changed_files=max_files,
        max_changed_lines=max_lines,
        min_sealed_delta=min_sealed_delta,
    )
    proposer_argv = _resolve_local_command_paths(_split_argv(proposer_command))
    if not proposer_argv:
        raise click.BadParameter("proposer command is empty")

    if compare_frozen:
        report = compare_adaptive_frozen(
            repo,
            task,
            run_dir,
            lambda: ExternalCLIProposer(proposer_argv),
            cycles=cycles,
            budget=budget,
            seed=seed,
        ).as_dict()
    else:
        report = VerifiedPatchSearch(repo, task, run_dir, seed=seed).run(
            ExternalCLIProposer(proposer_argv),
            cycles=cycles,
            budget=budget,
            adaptive=True,
        ).as_dict()
    click.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
