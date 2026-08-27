"""Drive a scheduled evolution sweep.

Replaces the out-of-repo shell driver. Same responsibilities — pick skills,
run them, report — with the failure modes the deployed version had closed off:

* a run that evaluates nothing exits non-zero instead of reporting success;
* the summary always lands somewhere, even when every remote channel is down;
* what was *deferred* is reported, so a bounded run never reads as full
  coverage;
* a per-run budget caps spend, and hitting it is stated rather than silently
  truncating the queue.

Usage:
    python -m evolution.monitor.run_rotation --skills-per-run 4 --dry-run
    python -m evolution.monitor.run_rotation --skills-per-run 4 --notify \\
        --optimizer-model openai-codex/gpt-5.6-luna \\
        --eval-model openai-codex/gpt-5.6-luna
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from evolution.core.hermes_paths import try_find_hermes_install
from evolution.core.notify import Notifier, RunSummary
from evolution.monitor.rotation import (
    DEFAULT_COOLDOWN_DAYS,
    RotationState,
    build_plan,
    coverage_estimate,
)
from evolution.skills.evolve_skill import EvolutionError, evolve

# Phase 4 is deliberately absent. Code evolution needs explicitly chosen
# targets and ends at a pull request; sweeping it on a timer would mean
# machine-authored code changes nobody asked for.
SWEEPABLE_PHASES = ("skills", "tools", "prompts")

console = Console()

DEFAULT_STATE_FILE = "rotation_state.json"


def run_rotation(
    skills_per_run: int = 4,
    hermes_data_dir: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    profile: Optional[str] = None,
    iterations: int = 10,
    eval_source: str = "sessiondb",
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    state_path: Optional[str] = None,
    output_root: str = "./output",
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
    time_budget_min: float = 0.0,
    create_pr: bool = False,
    run_tests: bool = False,
    phases: Optional[list[str]] = None,
    dry_run: bool = False,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> RunSummary:
    """Run one sweep and return its summary."""
    summary = RunSummary(subject="Hermes self-evolution — scheduled sweep")

    install = try_find_hermes_install(hermes_data_dir)
    if install is None:
        summary.failed.append(
            ("<sweep>", "no Hermes data directory found; set HERMES_DATA_DIR")
        )
        return summary

    console.print("\n[bold cyan]🧬 Scheduled evolution sweep[/bold cyan]")
    console.print(f"  Hermes data: {install.root} (via {install.source})")

    state = RotationState(
        Path(state_path) if state_path else Path(output_root) / DEFAULT_STATE_FILE
    ).load()

    plan = build_plan(
        install=install,
        state=state,
        skills_per_run=skills_per_run,
        profile=profile,
        cooldown_days=cooldown_days,
    )

    console.print(f"  Skills available: {plan.total_skills}")
    console.print(f"  Coverage at this cadence: {coverage_estimate(plan.total_skills, skills_per_run)}")
    console.print()

    table = Table(title="Queue")
    table.add_column("#", justify="right", style="dim")
    table.add_column("skill")
    table.add_column("why it was chosen")
    for i, candidate in enumerate(plan.selected, 1):
        table.add_row(str(i), f"{candidate.profile}/{candidate.name}", candidate.reason)
    console.print(table)

    if not plan.selected:
        summary.notes.append(
            f"Nothing queued: all {plan.total_skills} skills are inside the "
            f"{cooldown_days}-day cooldown and none are failing in production."
        )
        return summary

    # Deferred work is stated, never implied. A bounded run that prints only
    # its successes reads as complete coverage.
    if plan.deferred:
        summary.notes.append(
            f"{len(plan.deferred)} skill(s) deferred this run "
            f"(next up: {plan.deferred[0].profile}/{plan.deferred[0].name})"
        )

    deadline = time.time() + time_budget_min * 60 if time_budget_min > 0 else None

    for candidate in plan.selected:
        label = f"{candidate.profile}/{candidate.name}"

        if deadline and time.time() > deadline:
            summary.skipped.append((label, f"time budget of {time_budget_min:.0f} min exhausted"))
            continue

        console.print(f"\n[bold]── {label} ──[/bold]")
        try:
            result = evolve(
                skill_name=candidate.name,
                iterations=iterations,
                eval_source=eval_source,
                optimizer_model=optimizer_model,
                eval_model=eval_model,
                hermes_repo=hermes_repo,
                hermes_data_dir=hermes_data_dir,
                profile=candidate.profile,
                run_tests=run_tests,
                dry_run=dry_run,
                api_base=api_base,
                api_key=api_key,
                create_pr=create_pr,
                output_root=output_root,
            )
        except EvolutionError as exc:
            console.print(f"[red]✗ {exc}[/red]")
            summary.failed.append((label, str(exc)))
            state.record_result(candidate.profile, candidate.name, "ERROR")
            continue
        except Exception as exc:  # noqa: BLE001 — one bad skill must not end the sweep
            console.print(f"[red]✗ unexpected: {type(exc).__name__}: {exc}[/red]")
            summary.failed.append((label, f"{type(exc).__name__}: {exc}"))
            state.record_result(candidate.profile, candidate.name, "ERROR")
            continue

        if result.get("dry_run"):
            summary.succeeded.append(f"{label} (dry run)")
            continue

        verdict = result.get("verdict", "HOLD")
        state.record_result(candidate.profile, candidate.name, verdict)
        if verdict == "SHIP":
            summary.succeeded.append(f"{label} — {result.get('verdict_reason', '')}")
        else:
            summary.skipped.append((label, result.get("verdict_reason", verdict)))

    _sweep_other_phases(
        phases=phases or ["skills"],
        summary=summary,
        hermes_data_dir=hermes_data_dir,
        hermes_repo=hermes_repo,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        output_root=output_root,
        api_base=api_base,
        api_key=api_key,
        dry_run=dry_run,
    )

    # Persist the scheduler's memory only after the sweep, so a crash mid-run
    # leaves the queue intact for the next attempt rather than marking skills
    # as attempted when they were not.
    if not dry_run:
        state.save()
        summary.notes.append(f"Rotation state: {state.path}")

    return summary


def _sweep_other_phases(
    phases, summary, hermes_data_dir, hermes_repo, optimizer_model,
    eval_model, output_root, api_base, api_key, dry_run,
) -> None:
    """Run one Phase 2 and/or Phase 3 pass, when the sweep asks for them.

    These have no rotation state: there is one tool catalogue and one system
    prompt, so "which one next" does not arise the way it does for 324 skills.
    """
    if "tools" in phases:
        try:
            from evolution.tools.evolve_tool import evolve_tools

            console.print("\n[bold]── phase 2: tool descriptions ──[/bold]")
            result = evolve_tools(
                hermes_data_dir=hermes_data_dir,
                hermes_repo=hermes_repo,
                optimizer_model=optimizer_model,
                eval_model=eval_model,
                api_base=api_base,
                api_key=api_key,
                output_root=output_root,
                dry_run=dry_run,
            )
            verdict = getattr(result, "verdict", "HOLD")
            if verdict in ("SHIP", "DRY_RUN"):
                summary.succeeded.append(f"tool-descriptions — {verdict}")
            else:
                summary.skipped.append(("tool-descriptions", verdict))
        except Exception as exc:  # noqa: BLE001 — one phase must not end the sweep
            summary.failed.append(("tool-descriptions", f"{type(exc).__name__}: {exc}"))

    if "prompts" in phases:
        try:
            from evolution.prompts.evolve_prompt import evolve_prompt_section
            from evolution.prompts.prompt_sections import load_live_prompts
            from evolution.core.hermes_paths import try_find_hermes_install

            install = try_find_hermes_install(hermes_data_dir)
            live = load_live_prompts(install) if install else []
            if not live or not live[0].sections:
                summary.skipped.append(("prompt-sections", "no live prompt recorded"))
                return

            # The largest section of the most-used variant: the most context
            # budget riding on one artifact.
            section = max(live[0].sections, key=lambda s: s.size)
            console.print(f"\n[bold]── phase 3: prompt section '{section.title}' ──[/bold]")
            result = evolve_prompt_section(
                section_name=section.title,
                hermes_data_dir=hermes_data_dir,
                hermes_repo=hermes_repo,
                optimizer_model=optimizer_model,
                eval_model=eval_model,
                output_root=output_root,
                api_base=api_base,
                api_key=api_key,
                dry_run=dry_run,
            )
            verdict = result.get("verdict", "HOLD")
            if verdict == "SHIP" or result.get("dry_run"):
                summary.succeeded.append(f"prompt/{section.title} — {verdict}")
            else:
                summary.skipped.append((f"prompt/{section.title}", result.get("reason", verdict)))
        except Exception as exc:  # noqa: BLE001
            summary.failed.append(("prompt-sections", f"{type(exc).__name__}: {exc}"))


@click.command()
@click.option("--skills-per-run", default=4, help="How many skills to evolve this sweep")
@click.option("--hermes-data-dir", default=None, help="Hermes data directory (state.db, profiles)")
@click.option("--hermes-repo", default=None, help="hermes-agent repo path")
@click.option("--profile", default=None, help="Restrict the sweep to one profile")
@click.option("--iterations", default=10, help="GEPA full evaluations per skill")
@click.option("--eval-source", default="sessiondb",
              type=click.Choice(["sessiondb", "synthetic", "golden"]))
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--state-path", default=None, help="Where to keep rotation state")
@click.option("--output-root", default="./output")
@click.option("--cooldown-days", default=DEFAULT_COOLDOWN_DAYS,
              help="Skip skills evolved this recently unless they are failing")
@click.option("--time-budget-min", default=0.0, type=float,
              help="Stop starting new skills after this many minutes (0 = no limit)")
@click.option("--create-pr", is_flag=True, help="Open a PR for each SHIP verdict")
@click.option("--run-tests", is_flag=True, help="Run the hermes-agent test suite as a gate")
@click.option("--phases", default="skills",
              help="Comma-separated phases to sweep: skills, tools, prompts. "
                   "Code evolution is never swept — it needs chosen targets.")
@click.option("--notify/--no-notify", default=True, help="Deliver the sweep summary")
@click.option("--dry-run", is_flag=True, help="Validate setup for each skill without optimizing")
@click.option("--api-base", default=None)
@click.option("--api-key", default=None)
def main(skills_per_run, hermes_data_dir, hermes_repo, profile, iterations, eval_source,
         optimizer_model, eval_model, state_path, output_root, cooldown_days,
         time_budget_min, create_pr, run_tests, phases, notify, dry_run, api_base, api_key):
    """Run one scheduled evolution sweep."""
    selected = [p.strip() for p in phases.split(",") if p.strip()]
    unknown = [p for p in selected if p not in SWEEPABLE_PHASES]
    if unknown:
        console.print(
            f"[red]✗ Unknown phase(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(SWEEPABLE_PHASES)}.[/red]"
        )
        sys.exit(1)

    summary = run_rotation(
        skills_per_run=skills_per_run,
        hermes_data_dir=hermes_data_dir,
        hermes_repo=hermes_repo,
        profile=profile,
        iterations=iterations,
        eval_source=eval_source,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        state_path=state_path,
        output_root=output_root,
        cooldown_days=cooldown_days,
        time_budget_min=time_budget_min,
        create_pr=create_pr,
        run_tests=run_tests,
        phases=selected,
        dry_run=dry_run,
        api_base=api_base,
        api_key=api_key,
    )

    console.print("\n[bold]── Sweep summary ──[/bold]")
    console.print(summary.render())

    if notify:
        outcome = Notifier.from_env(Path(output_root) / "_status").send(
            summary.subject, summary.render()
        )
        style = "green" if outcome.delivered else "red"
        console.print(f"\n[{style}]Notification: {outcome.render()}[/{style}]")
        if not outcome.delivered:
            # Report it, but never let delivery decide the run's exit code.
            console.print("[yellow]Summary was not delivered to any remote channel.[/yellow]")

    sys.exit(summary.exit_code)


if __name__ == "__main__":
    main()
