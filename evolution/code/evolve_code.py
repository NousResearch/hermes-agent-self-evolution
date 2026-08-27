"""Phase 4 — evolve tool implementation code.

Mirrors the shape of skill evolution, with three differences that follow from
mutating executable code rather than prose.

*Admission before scoring.* A candidate is gated in a sandbox — targeted tests
visible to the mutator, full suite and replayed real commands held out — before
its quality is measured at all. See :mod:`evolution.code.admission`.

*No automatic deployment, ever.* Skill evolution can canary a variant into a
live install and roll it back on a regression. Code cannot: a bad tool
implementation is executing inside the agent before any outcome signal exists
to roll back on. Phase 4 ends at a pull request, and that is deliberate.

*The engine is out of process.* The mutation loop runs in an AGPL sidecar; see
:mod:`evolution.code.sidecar` for why.

Usage:
    python -m evolution.code.evolve_code --suggest
    python -m evolution.code.evolve_code --target agent/tool_executor.py --iterations 5
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from evolution.code.admission import (
    AdmissionGate,
    AdmissionVerdict,
    build_default_gate,
    materialize_candidate,
)
from evolution.code.sidecar import (
    CodeCandidate,
    SidecarFailed,
    SidecarJob,
    SidecarNotAvailable,
    run_sidecar,
    sidecar_available,
)
from evolution.code.targets import (
    CodeTarget,
    TargetError,
    recorded_checks_for,
    resolve_targets,
    suggest_targets,
)
from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.hermes_paths import try_find_hermes_install
from evolution.core.objectives import ObjectiveVector, ObjectiveWeights, select_best
from evolution.core.report import ABReport, arm_from_scores
from evolution.deploy.pr import PRPublisher, build_pr_body

console = Console()


class CodeEvolutionError(RuntimeError):
    """A Phase 4 run could not complete, with a reportable reason."""


def list_suggestions(hermes_repo: str, hermes_data_dir: Optional[str]) -> int:
    """Print ranked target suggestions from production signals."""
    install = try_find_hermes_install(hermes_data_dir)
    if install is None:
        console.print("[red]✗ No Hermes data directory found. Set HERMES_DATA_DIR.[/red]")
        return 1

    repo = resolve_hermes_agent_path(hermes_repo)
    suggestions = suggest_targets(install, repo)
    if not suggestions:
        console.print("[yellow]No targets could be suggested from the recorded signals.[/yellow]")
        return 1

    console.print(f"\n[bold]Ranked candidates from {install.root}[/bold]")
    console.print(
        "[dim]Evidence, not a selection. The tool-to-file mapping is a filename "
        "heuristic — confirm before evolving.[/dim]\n"
    )
    table = Table()
    table.add_column("tool")
    table.add_column("file")
    table.add_column("uses", justify="right")
    table.add_column("failed verifications", justify="right")
    for s in suggestions:
        table.add_row(
            s.label,
            s.paths[0],
            f"{s.evidence.get('uses', 0):,}",
            str(s.evidence.get("failure_mentions", 0)),
        )
    console.print(table)
    return 0


def evolve_code(
    target_paths: list[str],
    hermes_repo: Optional[str] = None,
    hermes_data_dir: Optional[str] = None,
    task: str = "",
    iterations: int = 5,
    population: int = 4,
    model: str = "",
    targeted_tests: Optional[list[str]] = None,
    allow_large: bool = False,
    output_root: str = "./output",
    create_pr: bool = False,
    pr_base: str = "main",
    push: bool = True,
    sidecar_path: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Run one Phase 4 evolution over a code target."""
    console.print("\n[bold cyan]🧬 Phase 4 — code evolution[/bold cyan]\n")

    repo = resolve_hermes_agent_path(hermes_repo)
    if not repo.is_dir():
        raise CodeEvolutionError(f"hermes-agent repo not found: {repo}")
    console.print(f"  Repo: {repo}")

    install = try_find_hermes_install(hermes_data_dir)
    if install:
        console.print(f"  Hermes data: {install.root} (via {install.source})")

    # ── target ───────────────────────────────────────────────────────────
    try:
        target = resolve_targets(repo, target_paths, allow_large=allow_large)
    except TargetError as exc:
        raise CodeEvolutionError(str(exc)) from exc
    console.print(f"  Target: {target.describe(repo)}")

    baseline_files = target.read(repo)

    # ── admission gate ───────────────────────────────────────────────────
    recorded = recorded_checks_for(install) if install else []
    gate = build_default_gate(
        hermes_repo=repo,
        targeted_tests=targeted_tests or [],
        recorded=recorded,
    )
    console.print(
        f"  Gate: {len(gate.visible)} visible check(s), {len(gate.hidden)} held out "
        f"({len(recorded)} replayed from recorded evidence)"
    )

    # ── sidecar ──────────────────────────────────────────────────────────
    available, detail = sidecar_available(sidecar_path)
    console.print(f"  Sidecar: {detail if available else '[red]unavailable[/red]'}")
    if not available and not dry_run:
        raise CodeEvolutionError(detail)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / "code" / target.label / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        console.print(f"  Would evolve {len(target.paths)} file(s) over {iterations} iterations")
        console.print(f"  Would gate each candidate, then {'open a PR' if create_pr else 'write output only'}")
        return {"dry_run": True, "target": target.label, "output_dir": str(output_dir)}

    # Establish that the baseline itself passes the gate. If it does not, every
    # candidate will look like an improvement for the wrong reason, and the run
    # is measuring a broken starting point rather than the mutation.
    console.print("\n[bold]Checking the baseline against the gate[/bold]")
    baseline_verdict = _gate_candidate(gate, repo, baseline_files, output_dir / "sandbox-baseline")
    for check in baseline_verdict.visible + baseline_verdict.hidden:
        style = "green" if check.passed else "red"
        console.print(f"  [{style}]{check.render()}[/{style}]")
    if not baseline_verdict.admitted:
        raise CodeEvolutionError(
            "The baseline does not pass its own admission gate "
            f"({baseline_verdict.reason}). Fix the repo before evolving it — "
            "otherwise every candidate scores against a broken reference."
        )

    # ── run the engine ───────────────────────────────────────────────────
    job = SidecarJob(
        repo_root=str(repo),
        files=baseline_files,
        task=task or f"Improve {', '.join(target.paths)} without changing its public behaviour.",
        output_dir=str(output_dir / "sidecar"),
        iterations=iterations,
        population=population,
        model=model,
        targeted_tests=list(targeted_tests or []),
    )

    console.print(f"\n[bold cyan]Running the evolution sidecar ({iterations} iterations)...[/bold cyan]\n")
    try:
        result = run_sidecar(job, explicit=sidecar_path)
    except (SidecarNotAvailable, SidecarFailed) as exc:
        raise CodeEvolutionError(str(exc)) from exc

    console.print(f"\n  Sidecar returned {len(result.candidates)} candidate(s) in {result.elapsed_s:.0f}s")
    if not result.candidates:
        raise CodeEvolutionError("The sidecar produced no candidates.")

    # ── gate every candidate ─────────────────────────────────────────────
    console.print("\n[bold]Admission[/bold]")
    admitted: list[tuple[CodeCandidate, AdmissionVerdict]] = []
    rejected: list[tuple[CodeCandidate, AdmissionVerdict]] = []

    for candidate in result.candidates:
        verdict = _gate_candidate(
            gate, repo, candidate.files, output_dir / f"sandbox-{candidate.id}"
        )
        (admitted if verdict.admitted else rejected).append((candidate, verdict))
        style = "green" if verdict.admitted else "yellow"
        changed = ", ".join(candidate.changed_paths(baseline_files)) or "(no change)"
        console.print(f"  [{style}]{candidate.id}[/{style}] {verdict.summary()} — {changed}")

    if not admitted:
        console.print(
            f"\n[yellow]⚠ No candidate passed the gate ({len(rejected)} rejected).[/yellow]"
        )
        _write_outputs(output_dir, target, baseline_files, result, admitted, rejected, None)
        return {
            "target": target.label,
            "verdict": "HOLD",
            "verdict_reason": f"all {len(rejected)} candidates rejected by the gate",
            "output_dir": str(output_dir),
            "admitted": 0,
            "rejected": len(rejected),
        }

    # ── select among survivors ───────────────────────────────────────────
    baseline_chars = sum(len(c) for c in baseline_files.values())
    vectors = [
        ObjectiveVector(
            quality=max(0.0, min(1.0, cand.score)),
            size_chars=cand.total_chars(),
            size_budget=int(baseline_chars * 1.25),
            baseline_chars=baseline_chars,
            max_growth=0.25,
        )
        for cand, _ in admitted
    ]
    best_index = select_best(vectors, ObjectiveWeights(quality=1.0, size=0.6)) or 0
    winner, winner_verdict = admitted[best_index]

    report = ABReport(
        subject=f"code — {target.label}",
        baseline=arm_from_scores("baseline", [0.0], baseline_chars),
        evolved=arm_from_scores("evolved", [winner.score], winner.total_chars()),
        metric_name="sidecar fitness, gated by admission",
        constraints_passed=True,
        extra={
            "target files": ", ".join(target.paths),
            "candidates": f"{len(admitted)} admitted / {len(rejected)} rejected",
            "held-out checks": len(gate.hidden),
            "sidecar elapsed": f"{result.elapsed_s:.0f}s",
        },
    )
    report.caveats.append(
        "The baseline arm is a fixed reference, not a re-measured sample: the "
        "sidecar scores candidates against the baseline it was given. Treat the "
        "delta as directional and read the diff."
    )

    console.print(f"\n[bold]Winner:[/bold] {winner.id} (score {winner.score:.3f})")
    console.print(f"  changed: {', '.join(winner.changed_paths(baseline_files)) or '(none)'}")

    _write_outputs(output_dir, target, baseline_files, result, admitted, rejected, winner)
    report.write(output_dir)
    console.print(f"\n  Output saved to {output_dir}/")

    outcome = {
        "target": target.label,
        "verdict": "SHIP",
        "verdict_reason": f"{winner.id} admitted with score {winner.score:.3f}",
        "winner": winner.id,
        "score": winner.score,
        "admitted": len(admitted),
        "rejected": len(rejected),
        "output_dir": str(output_dir),
        "pr": None,
    }

    # ── deploy: pull request only ────────────────────────────────────────
    if create_pr:
        outcome["pr"] = _open_pr(
            repo=repo,
            target=target,
            winner=winner,
            verdict=winner_verdict,
            report=report,
            timestamp=timestamp,
            pr_base=pr_base,
            push=push,
        )
    else:
        console.print("\n  [dim]Review the diff, then re-run with --create-pr.[/dim]")

    return outcome


# ── helpers ─────────────────────────────────────────────────────────────


def _gate_candidate(
    gate: AdmissionGate,
    repo: Path,
    files: dict[str, str],
    sandbox: Path,
) -> AdmissionVerdict:
    """Materialise a candidate and run the gate against it, then clean up."""
    try:
        materialize_candidate(repo, files, sandbox)
    except (OSError, ValueError) as exc:
        return AdmissionVerdict(admitted=False, reason=f"could not build sandbox: {exc}")

    try:
        return gate.admit(sandbox)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _write_outputs(
    output_dir: Path,
    target: CodeTarget,
    baseline: dict[str, str],
    result,
    admitted: list,
    rejected: list,
    winner: Optional[CodeCandidate],
) -> None:
    """Persist baseline, every candidate, and the admission record."""
    (output_dir / "baseline").mkdir(parents=True, exist_ok=True)
    for rel, content in baseline.items():
        path = output_dir / "baseline" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    record = {
        "target": target.label,
        "paths": target.paths,
        "admitted": [],
        "rejected": [],
    }
    for bucket, entries in (("admitted", admitted), ("rejected", rejected)):
        for candidate, verdict in entries:
            cand_dir = output_dir / bucket / candidate.id
            cand_dir.mkdir(parents=True, exist_ok=True)
            for rel, content in candidate.files.items():
                path = cand_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            record[bucket].append(
                {
                    "id": candidate.id,
                    "score": candidate.score,
                    "changed": candidate.changed_paths(baseline),
                    "summary": verdict.summary(),
                    "reason": verdict.reason,
                    "visible": [c.render() for c in verdict.visible],
                    # Held-out check *names* are recorded for the operator, who
                    # is allowed to see them — only the mutator is not.
                    "hidden": [c.render() for c in verdict.hidden],
                }
            )

    if winner:
        record["winner"] = winner.id
    (output_dir / "admission.json").write_text(json.dumps(record, indent=2))


def _open_pr(repo, target, winner, verdict, report, timestamp, pr_base, push) -> dict:
    """Open a PR for the winning candidate. The only deploy path Phase 4 has."""
    publisher = PRPublisher(repo=repo, base_branch=pr_base, draft=True)

    changed = winner.changed_paths(
        {p: (repo / p).read_text(encoding="utf-8", errors="replace") for p in target.paths}
    )
    if not changed:
        console.print("[yellow]⚠ Winner is byte-identical to the baseline; no PR.[/yellow]")
        return {"created": False, "detail": "winner is identical to the baseline"}

    constraint_lines = [f"✓ **{c.name}** — passed" for c in verdict.visible + verdict.hidden if c.passed]
    body = build_pr_body(
        skill_name=f"code/{target.label}",
        report_markdown=report.to_markdown(),
        constraint_lines=constraint_lines,
        run_metadata={
            "files": ", ".join(target.paths),
            "held-out checks passed": len([c for c in verdict.hidden if c.passed]),
            "engine": "darwinian_evolver via AGPL sidecar",
        },
    )
    body += (
        "\n\n> **This is machine-authored executable code.** It passed a sandboxed "
        "admission gate including held-out checks, which is evidence that it does "
        "not obviously break anything — not evidence that it is correct. Read every "
        "line before merging.\n"
    )

    # Multi-file publish: write each changed file, then commit once.
    console.print(f"\n[bold]Opening pull request[/bold] against {repo}")
    first = True
    result = None
    for rel in changed:
        result = publisher.publish(
            skill_name=f"{target.label}-{rel.replace('/', '-')}",
            target_path=repo / rel,
            content=winner.files[rel],
            title=f"evolve(code): {target.label} — {len(changed)} file(s)",
            body=body,
            timestamp=timestamp,
            push=push and first,
        )
        first = False

    style = "green" if result and result.created else "yellow"
    console.print(f"  [{style}]{result.render() if result else 'no files changed'}[/{style}]")
    return {
        "created": bool(result and result.created),
        "branch": result.branch if result else "",
        "url": result.url if result else "",
        "detail": result.detail if result else "",
    }


# ── CLI ─────────────────────────────────────────────────────────────────


@click.command()
@click.option("--target", "targets", multiple=True, help="Repo-relative file to evolve (repeatable)")
@click.option("--suggest", is_flag=True, help="Rank plausible targets from production signals and exit")
@click.option("--task", default="", help="What the mutator should try to improve")
@click.option("--iterations", default=5, help="Evolution iterations")
@click.option("--population", default=4, help="Candidates per generation")
@click.option("--model", default="", help="Model for the sidecar's mutator")
@click.option("--test", "targeted_tests", multiple=True,
              help="Test path the mutator may see failures from (repeatable)")
@click.option("--hermes-repo", default=None, help="hermes-agent repo path")
@click.option("--hermes-data-dir", default=None, help="Hermes data dir (state.db, verification evidence)")
@click.option("--sidecar", "sidecar_path", default=None, help="Path to the AGPL evolution sidecar")
@click.option("--allow-large", is_flag=True, help="Permit a target over the size limit")
@click.option("--output-root", default="./output")
@click.option("--create-pr", is_flag=True, help="Open a draft PR for the winner")
@click.option("--pr-base", default="main")
@click.option("--no-push", is_flag=True, help="Commit locally without pushing")
@click.option("--dry-run", is_flag=True, help="Validate setup without evolving")
def main(targets, suggest, task, iterations, population, model, targeted_tests,
         hermes_repo, hermes_data_dir, sidecar_path, allow_large, output_root,
         create_pr, pr_base, no_push, dry_run):
    """Evolve Hermes tool implementation code (Phase 4)."""
    if suggest:
        sys.exit(list_suggestions(hermes_repo, hermes_data_dir))

    if not targets:
        console.print("[red]✗ Provide --target, or --suggest to see candidates.[/red]")
        sys.exit(1)

    try:
        evolve_code(
            target_paths=list(targets),
            hermes_repo=hermes_repo,
            hermes_data_dir=hermes_data_dir,
            task=task,
            iterations=iterations,
            population=population,
            model=model,
            targeted_tests=list(targeted_tests),
            allow_large=allow_large,
            output_root=output_root,
            create_pr=create_pr,
            pr_base=pr_base,
            push=not no_push,
            sidecar_path=sidecar_path,
            dry_run=dry_run,
        )
    except CodeEvolutionError as exc:
        console.print(f"\n[red]✗ {exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
