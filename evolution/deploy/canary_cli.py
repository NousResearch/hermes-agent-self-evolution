"""Inspect, evaluate and roll back deployed canaries.

A canary that nothing ever checks is just an undocumented edit to a live
install. This is the other half of ``evolve_skill --canary``: it reads the
outcomes Hermes has recorded since a variant went out and decides whether to
keep it.

Usage:
    python -m evolution.deploy.canary_cli --list
    python -m evolution.deploy.canary_cli --evaluate
    python -m evolution.deploy.canary_cli --evaluate --apply
    python -m evolution.deploy.canary_cli --rollback my-skill
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from evolution.core.hermes_paths import try_find_hermes_install
from evolution.core.notify import Notifier, RunSummary
from evolution.deploy.canary import (
    PROMOTE,
    ROLLBACK,
    WAIT,
    CanaryLedger,
    evaluate_canary,
    prompt_variant_comparison,
    rollback_canary,
)

console = Console()

LEDGER_RELATIVE = Path("evolution") / "canary-ledger.json"


def _ledger_for(install, explicit: Optional[str]) -> CanaryLedger:
    if explicit:
        return CanaryLedger(Path(explicit))
    return CanaryLedger(Path(install.root) / LEDGER_RELATIVE)


def cmd_list(ledger: CanaryLedger) -> int:
    records = ledger.all()
    if not records:
        console.print("[yellow]No canaries recorded.[/yellow]")
        return 0

    table = Table(title=f"Canaries — {ledger.path}")
    table.add_column("skill")
    table.add_column("status")
    table.add_column("deployed")
    table.add_column("baseline", justify="right")
    table.add_column("variant")
    for r in records:
        rate = (
            f"{r.baseline_success_rate:.0%} / {r.baseline_observations}"
            if r.baseline_success_rate is not None
            else "—"
        )
        table.add_row(r.skill_name, r.status, r.deployed_at[:16], rate, r.variant_sha[:10])
    console.print(table)
    return 0


def cmd_evaluate(
    install,
    ledger: CanaryLedger,
    apply: bool,
    min_observations: int,
    notify: bool,
) -> int:
    active = ledger.active()
    if not active:
        console.print("[yellow]No active canaries to evaluate.[/yellow]")
        return 0

    summary = RunSummary(subject="Hermes canary evaluation")
    exit_code = 0

    for record in active:
        verdict = evaluate_canary(install, record, min_observations=min_observations)
        colour = {PROMOTE: "green", ROLLBACK: "red", WAIT: "yellow"}[verdict.decision]
        console.print(f"\n[bold]{record.skill_name}[/bold]")
        console.print(f"  [{colour}]{verdict.render()}[/{colour}]")

        # Interactive sessions carry no skill label, so prompt-hash groups that
        # appeared only after this deployment are directional evidence, not
        # attribution. Reported separately so the two are never conflated.
        new_variants = prompt_variant_comparison(install, record)
        if new_variants:
            best = max(new_variants.items(), key=lambda kv: kv[1]["sessions"])
            console.print(
                f"  [dim]since deploy: {len(new_variants)} new prompt variant(s); "
                f"largest {best[0][:12]} at {best[1]['success_rate']:.0%} over "
                f"{best[1]['sessions']} sessions (directional only)[/dim]"
            )

        if verdict.decision == ROLLBACK:
            summary.failed.append((record.skill_name, verdict.reason))
            exit_code = 1
            if apply:
                if rollback_canary(record, ledger):
                    console.print(f"  [green]rolled back to {record.backup_path}[/green]")
                    summary.notes.append(f"{record.skill_name}: rolled back")
                else:
                    console.print("  [red]rollback FAILED — restore the backup by hand[/red]")
                    summary.notes.append(f"{record.skill_name}: ROLLBACK FAILED")
            else:
                console.print("  [dim]re-run with --apply to roll it back[/dim]")

        elif verdict.decision == PROMOTE:
            summary.succeeded.append(f"{record.skill_name} — {verdict.reason}")
            if apply:
                ledger.update_status(
                    record.skill_name, record.variant_sha, "promoted", verdict.reason
                )
                console.print("  [green]promoted; no longer under observation[/green]")
        else:
            summary.skipped.append((record.skill_name, verdict.reason))

    if notify:
        outcome = Notifier.from_env().send(summary.subject, summary.render())
        style = "green" if outcome.delivered else "red"
        console.print(f"\n[{style}]Notification: {outcome.render()}[/{style}]")

    return exit_code


def cmd_rollback(ledger: CanaryLedger, skill_name: str) -> int:
    matches = [r for r in ledger.active() if r.skill_name == skill_name]
    if not matches:
        console.print(f"[red]✗ No active canary for '{skill_name}'.[/red]")
        return 1

    failed = False
    for record in matches:
        if rollback_canary(record, ledger):
            console.print(f"[green]✓ {skill_name} restored from {record.backup_path}[/green]")
        else:
            console.print(f"[red]✗ Could not restore {skill_name} — backup missing[/red]")
            failed = True
    return 1 if failed else 0


@click.command()
@click.option("--list", "do_list", is_flag=True, help="Show every recorded canary")
@click.option("--evaluate", "do_evaluate", is_flag=True, help="Judge each active canary")
@click.option("--rollback", "rollback_skill", default=None, help="Force a rollback by skill name")
@click.option("--apply", is_flag=True,
              help="Act on the verdict: roll back regressions, close out promotions. "
                   "Without it, --evaluate only reports.")
@click.option("--min-observations", default=20, type=int,
              help="Post-deploy runs required before a verdict is issued")
@click.option("--hermes-data-dir", default=None, help="Hermes data directory")
@click.option("--ledger", "ledger_path", default=None, help="Explicit ledger file")
@click.option("--notify/--no-notify", default=False, help="Deliver the evaluation summary")
def main(do_list, do_evaluate, rollback_skill, apply, min_observations,
         hermes_data_dir, ledger_path, notify):
    """Manage skill canaries deployed by `evolve_skill --canary`."""
    install = try_find_hermes_install(hermes_data_dir)
    if install is None and not ledger_path:
        console.print(
            "[red]✗ No Hermes data directory found. Set HERMES_DATA_DIR, or pass "
            "--ledger.[/red]"
        )
        sys.exit(1)

    ledger = _ledger_for(install, ledger_path)

    if rollback_skill:
        sys.exit(cmd_rollback(ledger, rollback_skill))
    if do_evaluate:
        if install is None:
            console.print("[red]✗ --evaluate needs a Hermes data dir for outcome data.[/red]")
            sys.exit(1)
        sys.exit(cmd_evaluate(install, ledger, apply, min_observations, notify))
    if do_list:
        sys.exit(cmd_list(ledger))

    console.print("Nothing to do. Pass --list, --evaluate or --rollback <skill>.")
    sys.exit(1)


if __name__ == "__main__":
    main()
