"""Unified CLI — run all evolution phases in the correct order."""

from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from evolution.loop.history import load_history, append_run, RunRecord
from evolution.loop.scheduler import select_pending_phases
from evolution.loop.orchestrator import run_phases

console = Console()

HISTORY_PATH = Path(__file__).resolve().parent / ".run_history.jsonl"


def _build_phase_funcs(hermes_repo: str, iterations: int,
                       write_back: bool) -> dict:
    """Build the mapping of phase_name to evolve function."""
    funcs = {}

    def _tools():
        from evolution.tools.evolve_tool_descriptions import evolve_tools
        return evolve_tools(
            hermes_repo=hermes_repo, iterations=iterations,
            write_back=write_back, dry_run=False,
        )

    def _guidance():
        from evolution.prompts.evolve_guidance import evolve_guidance
        return evolve_guidance(
            hermes_repo=hermes_repo, iterations=iterations,
            write_back=write_back, dry_run=False,
        )

    def _params():
        from evolution.tools.evolve_tool_descriptions import evolve_tools
        return evolve_tools(
            hermes_repo=hermes_repo, iterations=iterations,
            write_back=write_back, dry_run=False,
            evolve_params=True,
        )

    funcs["tools"] = _tools
    funcs["guidance"] = _guidance
    funcs["params"] = _params
    return funcs


@click.command()
@click.option("--phases", default=None, help="Comma-separated phases to run (default: all pending)")
@click.option("--force", is_flag=True, help="Run all phases even if no drift detected")
@click.option("--iterations", default=10, help="GEPA iterations per phase")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--dry-run", is_flag=True, help="Enumerate phases without evolving")
@click.option("--write-back", is_flag=True, help="Apply changes to hermes-agent repo")
def main(phases, force, iterations, hermes_repo, dry_run, write_back):
    """Run the continuous evolution loop -- evolve all pending targets."""
    console.print("[bold cyan]Hermes Self-Evolution -- Continuous Loop[/bold cyan]\n")

    history = load_history(HISTORY_PATH)
    latest = history.latest_per_phase()

    only_phases = phases.split(",") if phases else None
    pending = select_pending_phases(latest, force=force, only=only_phases)

    if not pending:
        console.print("[green]✓ All phases up-to-date. Nothing to evolve.[/green]")
        return

    console.print(f"  Phases to run: {', '.join(pending)}")
    console.print(f"  Iterations: {iterations}")
    console.print(f"  Hermes repo: {hermes_repo or '(auto-detect)'}")
    console.print(f"  Dry-run: {dry_run}")
    console.print(f"  Write-back: {write_back}")

    if dry_run:
        console.print("\n[bold green]DRY RUN — would execute phases above.[/bold green]")
        return

    phase_funcs = _build_phase_funcs(hermes_repo, iterations, write_back)

    console.print(f"\n[bold]Running {len(pending)} phases...[/bold]")
    result = run_phases(phase_funcs, phases=pending)

    metrics = result.summary_metrics()
    console.print(f"\n[bold]Results:[/bold] {result.status.upper()}")
    console.print(f"  Phases run: {metrics['phases_run']}")
    console.print(f"  Phases skipped: {metrics['phases_skipped']}")
    console.print(f"  Phases failed: {metrics['phases_failed']}")
    console.print(f"  Total improvement: {metrics['total_improvement']:+.3f}")
    console.print(f"  Total time: {result.total_elapsed}s")

    import subprocess
    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(hermes_repo) if hermes_repo else None,
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
    except Exception:
        pass

    record = RunRecord(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        timestamp=datetime.now().isoformat(),
        phases={name: data for name, data in result.phases.items()},
        git_sha=git_sha,
        elapsed_seconds=result.total_elapsed,
    )
    append_run(HISTORY_PATH, record)
    console.print(f"  History saved to {HISTORY_PATH}")

    if write_back:
        console.print("\n[bold green]✓ Changes written to hermes-agent repo.[/bold green]")
    else:
        console.print("\n[yellow]⚠ Report-only — no changes written. Use --write-back to apply.[/yellow]")


if __name__ == "__main__":
    main()
