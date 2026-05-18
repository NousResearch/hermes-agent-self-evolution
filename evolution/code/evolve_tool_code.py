"""Evolve a tool implementation file via LLM-driven mutation.

Usage:
    python -m evolution.code.evolve_tool_code \
        --tool file_tools --bug-brief "read_file fails on symlinks" \
        --iterations 3 --engine internal \
        --optimizer-model 'ollama/qwen2.5:7b' \
        --hermes-repo /path/to/hermes-agent
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from evolution.code.code_organism import CodeOrganism, OrganismError
from evolution.code.fitness import CompositeFitness, compute_fitness
from evolution.code.mutator import select_engine
from evolution.code.sandbox import WorktreeSandbox
from evolution.code.signature_freeze import compare_files, compare_sources
from evolution.core.config import EvolutionConfig
from evolution.core.phase_gate import write_phase_gate_result
from evolution.core.reproducibility import (
    build_code_reproducibility_manifest,
    write_manifest,
)

console = Console()

_TOOL_RELATIVE_PATHS: dict[str, str] = {
    "file_tools": "tools/file_tools.py",
    "file_operations": "tools/file_operations.py",
    "memory_tool": "tools/memory_tool.py",
    "web_tools": "tools/web_tools.py",
    "skill_manager_tool": "tools/skill_manager_tool.py",
    "session_search_tool": "tools/session_search_tool.py",
    "path_security": "tools/path_security.py",
    "todo_tool": "tools/todo_tool.py",
    "url_safety": "tools/url_safety.py",
}


def _resolve_tool_path(tool_name: str, hermes_root: Path) -> Path:
    rel = _TOOL_RELATIVE_PATHS.get(tool_name)
    if rel is None:
        candidates = list((hermes_root / "tools").glob("*.py"))
        matches = [c for c in candidates if tool_name in c.stem]
        if len(matches) == 1:
            rel = str(matches[0].relative_to(hermes_root))
        else:
            raise click.BadParameter(
                f"Unknown tool {tool_name!r}. Known: {sorted(_TOOL_RELATIVE_PATHS)} "
                f"or a substring of a file in agent/tools/*.py",
            )
    abs_path = hermes_root / rel
    if not abs_path.is_file():
        raise click.BadParameter(f"tool file not found: {abs_path}")
    return Path(rel)


@click.command()
@click.option("--tool", required=True, help="Tool name (e.g. file_tools) or relative path")
@click.option("--bug-brief", required=True, help="Plain-English description of the bug to fix")
@click.option("--iterations", default=3, show_default=True, type=int)
@click.option("--engine", default="internal", show_default=True,
              type=click.Choice(["internal", "darwinian-evolver"]))
@click.option("--optimizer-model", default="ollama/qwen2.5:7b", show_default=True)
@click.option("--bug-repro-test", default=None,
              help="Path to a pytest test that reproduces the bug (under hermes repo)")
@click.option("--pytest-target", default=None,
              help="Specific test path/file to run for fitness gate instead of full suite")
@click.option("--hermes-repo", default=None, help="Override hermes-agent repo path")
@click.option("--dry-run", is_flag=True, default=False)
def main(
    tool: str,
    bug_brief: str,
    iterations: int,
    engine: str,
    optimizer_model: str,
    bug_repro_test: Optional[str],
    pytest_target: Optional[str],
    hermes_repo: Optional[str],
    dry_run: bool,
):
    evolve(
        tool_name=tool,
        bug_brief=bug_brief,
        iterations=iterations,
        engine_name=engine,
        optimizer_model=optimizer_model,
        bug_repro_test=bug_repro_test,
        pytest_target=pytest_target,
        hermes_repo=hermes_repo,
        dry_run=dry_run,
    )


def evolve(
    *,
    tool_name: str,
    bug_brief: str,
    iterations: int = 3,
    engine_name: str = "internal",
    optimizer_model: str = "ollama/qwen2.5:7b",
    bug_repro_test: str | None = None,
    pytest_target: str | None = None,
    hermes_repo: str | None = None,
    dry_run: bool = False,
):
    config = EvolutionConfig(optimizer_model=optimizer_model)
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    console.print(f"\n[bold cyan]Phase 4 — Code Evolution[/bold cyan] -> [bold]{tool_name}[/bold]\n")

    rel_path = _resolve_tool_path(tool_name, config.hermes_agent_path)
    console.print(f"  Tool file: {rel_path}")
    console.print(f"  Bug brief: {bug_brief[:120]}{'...' if len(bug_brief) > 120 else ''}")
    console.print(f"  Engine:    {engine_name}")
    console.print(f"  Iterations: {iterations}")

    organism = CodeOrganism(
        repo_root=config.hermes_agent_path,
        relative_tool_path=rel_path,
    )
    console.print(f"  Baseline SHA256: {organism.baseline_sha256[:12]}...")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        organism.cleanup()
        return

    mutator = select_engine(engine_name, model_name=optimizer_model)
    freeze_baseline = compare_sources(organism.baseline_source, organism.baseline_source)
    console.print(
        f"  Baseline invariants: {len(freeze_baseline.signature_violations)} sig violations "
        f"(expected 0), {freeze_baseline.error_handling_baseline} try/except blocks",
    )

    best_fitness: CompositeFitness | None = None
    best_iteration: int = -1
    best_candidate_source: str | None = None

    for i in range(iterations):
        console.print(f"\n[bold]Iteration {i + 1}/{iterations}[/bold]")

        try:
            candidate_source = mutator.propose(
                baseline_source=organism.baseline_source,
                bug_brief=bug_brief,
                iteration=i,
            )
        except Exception as exc:
            console.print(f"  [red]Mutator failed: {exc}[/red]")
            continue

        if not candidate_source.strip():
            console.print("  [yellow]Mutator returned empty source; skipping[/yellow]")
            continue

        try:
            candidate = organism.create_candidate(
                iteration=i,
                mutated_source=candidate_source,
                commit_message=f"hermes-evo: {tool_name} iter {i}",
            )
        except OrganismError as exc:
            console.print(f"  [red]Worktree creation failed: {exc}[/red]")
            continue

        sandbox = WorktreeSandbox(candidate.worktree_path)
        console.print(f"  Worktree: {candidate.worktree_path.name}")

        pytest_result = sandbox.run_pytest(
            test_target=pytest_target, timeout_seconds=300,
        )
        console.print(
            f"  pytest: {'PASS' if pytest_result.passed else 'FAIL'} "
            f"({pytest_result.duration_seconds:.1f}s)",
        )
        if not pytest_result.passed:
            console.print(f"  [dim]{pytest_result.stdout[-500:]}"[:500])
            console.print(f"  [dim]{pytest_result.stderr[-500:]}"[:500])

        ruff_result = sandbox.run_ruff()
        console.print(
            f"  ruff:   {'PASS' if ruff_result.passed else 'FAIL'}",
        )

        freeze = compare_files(organism.baseline_tool_path, candidate.tool_path)
        freeze_icon = "PASS" if freeze.passed else "FAIL"
        console.print(f"  freeze: {freeze_icon}", highlight=False)
        if not freeze.passed:
            for v in freeze.signature_violations[:3]:
                console.print(f"    [red]- {v}[/red]")
            for v in freeze.registry_violations[:3]:
                console.print(f"    [red]- {v}[/red]")
            if freeze.error_handling_decreased:
                console.print(
                    f"    [red]- error handling decreased "
                    f"({freeze.error_handling_baseline} -> {freeze.error_handling_candidate})[/red]",
                )

        bug_repro_baseline_result = None
        bug_repro_candidate_result = None
        if bug_repro_test:
            bug_repro_baseline_result = WorktreeSandbox(
                organism.repo_root,
            ).run_pytest(test_target=bug_repro_test, timeout_seconds=120)
            bug_repro_candidate_result = sandbox.run_pytest(
                test_target=bug_repro_test, timeout_seconds=120,
            )
            br_status = (
                "FAIL->PASS" if (not bug_repro_baseline_result.passed
                                 and bug_repro_candidate_result.passed)
                else "no transition"
            )
            console.print(f"  bug-repro: {br_status}")

        fitness = compute_fitness(
            pytest_candidate=pytest_result,
            ruff_candidate=ruff_result,
            freeze=freeze,
            bug_repro_baseline=bug_repro_baseline_result,
            bug_repro_candidate=bug_repro_candidate_result,
        )
        console.print(f"  fitness: {fitness.score:.2f}")

        if fitness.score > (best_fitness.score if best_fitness else -1.0):
            best_fitness = fitness
            best_iteration = i
            best_candidate_source = candidate_source

    # ── Report + artifacts ──────────────────────────────────────────────
    if best_fitness is None or best_fitness.score == 0.0:
        console.print("\n[red]No viable candidate produced across all iterations.[/red]")
        organism.cleanup()
        return

    console.print(f"\n[bold green]Best candidate: iteration {best_iteration}, "
                  f"fitness {best_fitness.score:.2f}[/bold green]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "code" / tool_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "baseline_source.py").write_text(organism.baseline_source, encoding="utf-8")
    (output_dir / "candidate_source.py").write_text(best_candidate_source or "", encoding="utf-8")

    freeze_final = compare_sources(organism.baseline_source, best_candidate_source or "")
    metrics = {
        "tool_name": tool_name,
        "engine": engine_name,
        "iterations": iterations,
        "best_iteration": best_iteration,
        "best_fitness": best_fitness.to_dict(),
        "freeze_final": freeze_final.to_dict(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    manifest = build_code_reproducibility_manifest(
        tool_name=tool_name,
        tool_source_path=organism.baseline_tool_path,
        baseline_source=organism.baseline_source,
        config=config,
        best_fitness_score=best_fitness.score,
        best_iteration=best_iteration,
        iterations=iterations,
        engine=engine_name,
        evolution_repo_path=Path(__file__).resolve().parents[2],
    )
    write_manifest(output_dir / "reproducibility_manifest.json", manifest)

    from evolution.core.phase_gate import evaluate_phase4_gate

    gate_result = evaluate_phase4_gate(
        fitness=best_fitness,
        freeze=freeze_final,
        config=config,
        manifest=manifest,
    )
    write_phase_gate_result(output_dir / "phase4_gate.json", gate_result)

    if gate_result.passed:
        console.print("[green]Phase 4 gate passed[/green]")
    else:
        console.print("[red]Phase 4 gate failed[/red]")
        for f in gate_result.failures:
            console.print(f"  - {f}")

    console.print(
        f"\n  [bold]Artifacts saved to {output_dir}/[/bold]\n"
        f"  Candidate source is NOT auto-merged. Review {output_dir / 'candidate_source.py'} "
        f"before applying.",
    )

    organism.cleanup()


if __name__ == "__main__":
    main()
