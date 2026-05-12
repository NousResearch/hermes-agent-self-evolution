"""CLI entry point for the closed-loop validation harness."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from evolution.validation.artifact_installer import HermesToolDescriptionInstaller
from evolution.validation.hermes_runner import (
    DEFAULT_TASK_TIMEOUT_SECONDS,
    HermesAgentRunner,
)
from evolution.validation.task import TaskSuite
from evolution.validation.validator import ClosedLoopValidator, ValidationInputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)

console = Console()


@click.command()
@click.option(
    "--tool",
    "tool_name",
    required=True,
    help="The Hermes tool whose description is being validated (e.g. 'patch').",
)
@click.option(
    "--hermes-repo",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to your hermes-agent checkout. The tool file inside its tools/ "
    "directory is mutated and restored.",
)
@click.option(
    "--tasks",
    "tasks_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to a JSONL task-suite file.",
)
@click.option(
    "--baseline",
    "baseline_artifact",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to the baseline tool-module file (e.g. the unmutated copy from "
    "hermes-agent/tools/).",
)
@click.option(
    "--evolved",
    "evolved_artifact",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to the evolved tool-module file (an evolve_tool --apply output, "
    "or a hand-crafted candidate for harness validation).",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Where to write the validation report. Defaults to "
    "output/validation/<tool>/<timestamp>/.",
)
@click.option(
    "--task-timeout-seconds",
    default=DEFAULT_TASK_TIMEOUT_SECONDS,
    type=click.IntRange(min=1),
    help=f"Per-task wall-clock cap for hermes -z (default {DEFAULT_TASK_TIMEOUT_SECONDS}s). "
    "Timeouts count as abstentions, not failures.",
)
def main(
    tool_name: str,
    hermes_repo: Path,
    tasks_path: Path,
    baseline_artifact: Path,
    evolved_artifact: Path,
    output_dir: Path | None,
    task_timeout_seconds: int,
) -> None:
    """Closed-loop validation: run baseline + evolved through real hermes
    sessions on a task suite, report whether evolved measurably shifted
    behavior. Exit 0 on pass, 1 on regression.
    """
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / "validation" / tool_name / ts

    suite = TaskSuite.from_jsonl(tasks_path)
    installer = HermesToolDescriptionInstaller(
        hermes_repo=hermes_repo,
        tool_name=tool_name,
    )
    runner = HermesAgentRunner(timeout_seconds=task_timeout_seconds)
    validator = ClosedLoopValidator(installer=installer, runner=runner)

    console.print(
        f"\n[bold cyan]Closed-loop validation[/bold cyan] — tool: [bold]{tool_name}[/bold]"
    )
    console.print(f"  Hermes repo: {hermes_repo}")
    console.print(f"  Task suite:  {tasks_path} ({len(suite.tasks)} tasks, sha256 {suite.sha256[:12]}…)")
    console.print(f"  Baseline:    {baseline_artifact}")
    console.print(f"  Evolved:     {evolved_artifact}")
    console.print(f"  Output dir:  {output_dir}")
    console.print(
        "  [dim]Each task is one hermes -z invocation; expect ~$0.05-0.50 per task.[/dim]"
    )

    report = validator.validate(ValidationInputs(
        tool_name=tool_name,
        suite=suite,
        baseline_artifact=baseline_artifact,
        evolved_artifact=evolved_artifact,
    ))

    output_dir.mkdir(parents=True, exist_ok=True)
    report.write_json(output_dir / "validation_report.json")
    console.print()
    report.render_console(console)
    console.print(f"\n  Output saved to {output_dir}/")

    sys.exit(0 if report.decision == "pass" else 1)


if __name__ == "__main__":
    main()
