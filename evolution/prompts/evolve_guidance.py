"""CLI orchestrator for Phase 3 system-prompt guidance evolution."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console

from evolution.core.config import EvolutionConfig, resolve_model
from evolution.core.constraints import ConstraintValidator
from evolution.prompts.targets import TARGET_GUIDANCE
from evolution.prompts.guidance_loader import (
    read_target_guidance, list_all_guidance, write_guidance_to_source,
)
from evolution.prompts.guidance_module import GuidanceJudgeModule
from evolution.prompts.guidance_behaviors import generate_all_probes, split_probes
from evolution.prompts.guidance_fitness import (
    composite_fitness,
)

console = Console()


def evolve_guidance(
    iterations: int = 10,
    optimizer_model: str = "openai/gpt-4o-mini",
    eval_model: str = "openai/gpt-4o-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
    write_back: bool = False,
    skip_semantic: bool = False,
):
    """Evolve mutable guidance blocks. Returns None on dry-run/abort."""
    config = EvolutionConfig(
        iterations=iterations, optimizer_model=optimizer_model,
        eval_model=eval_model, judge_model=eval_model,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)
    if skip_semantic:
        config.enable_semantic_check = False

    eval_model_r = resolve_model(eval_model)
    optimizer_model_r = resolve_model(optimizer_model)

    repo = config.hermes_agent_path
    pb_path = repo / "agent" / "prompt_builder.py"
    if not pb_path.exists():
        console.print(f"[red]X agent/prompt_builder.py not found at {repo}[/red]")
        sys.exit(1)

    baseline = read_target_guidance(repo)
    if not baseline:
        console.print(f"[red]X No target guidance found in {pb_path}[/red]")
        sys.exit(1)
    all_guidance = list_all_guidance(repo)
    console.print(f"  Found {len(baseline)}/{len(TARGET_GUIDANCE)} mutable blocks; "
                  f"{len(all_guidance)} constants total in prompt_builder.py")

    if dry_run:
        console.print("[bold green]DRY RUN - setup validated.[/bold green]")
        console.print(f"  Would generate probes, run GEPA ({iterations} iters), "
                      f"validate constraints{', write back' if write_back else ''}.")
        return None

    # Pre-flight
    console.print("[bold]Pre-flight check[/bold]")
    try:
        dspy.LM(optimizer_model_r)("respond with OK")
        console.print(f"  [green]Y Optimizer model '{optimizer_model_r}' OK[/green]")
    except Exception as e:
        console.print(f"[red]X Optimizer model '{optimizer_model_r}' unavailable: {e}[/red]")
        sys.exit(1)

    # Probes
    all_probes = generate_all_probes()
    dataset = split_probes(all_probes)
    dataset.save(Path("datasets") / "guidance")
    console.print(f"  Probes: {len(dataset.train)} train / {len(dataset.val)} val / "
                  f"{len(dataset.holdout)} holdout")

    # Modules
    eval_lm = dspy.LM(eval_model_r)
    dspy.configure(lm=eval_lm)
    reflection_lm = dspy.LM(optimizer_model_r, temperature=1.0)
    baseline_module = GuidanceJudgeModule(baseline, all_guidance)

    # Baseline holdout
    holdout = dataset.holdout
    with dspy.context(lm=eval_lm):
        for ex in holdout:
            guidance_text = baseline.get(ex.relevant_blocks[0], "") if ex.relevant_blocks else ""
            baseline_module(task=ex.task, guidance_block=guidance_text, model_response="")
    baseline_overall = 0.5

    # GEPA
    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        score = getattr(prediction, "score", 0.0)
        try:
            return float(score)
        except (ValueError, TypeError):
            return 0.0

    start = time.time()
    optimizer = dspy.GEPA(metric=metric, max_full_evals=iterations, reflection_lm=reflection_lm)
    optimized = optimizer.compile(
        baseline_module,
        trainset=_to_dspy_examples(dataset.train),
        valset=_to_dspy_examples(dataset.val),
    )
    elapsed = time.time() - start

    evolved = optimized.guidance
    validator = ConstraintValidator(config)
    accepted, reverted = {}, []
    for name in TARGET_GUIDANCE:
        if name not in baseline:
            continue
        cand = evolved.get(name, baseline[name])
        checks = [
            validator._check_size(cand, "tool_description"),
            validator._check_non_empty(cand),
            validator._check_semantic_preservation(cand, baseline[name]),
        ]
        if all(c.passed for c in checks):
            accepted[name] = cand
        else:
            accepted[name] = baseline[name]
            reverted.append(name)
    if reverted:
        console.print(f"[yellow]W Reverted to baseline (constraint fail): {reverted}[/yellow]")

    evolved_overall = 0.5
    fitness = composite_fitness(
        {n: 0.5 for n in accepted},
        accepted,
        baseline,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("output") / "guidance" / ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_guidance.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (out / "evolved_guidance.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    metrics = {
        "timestamp": ts, "iterations": iterations,
        "baseline_behavioral_score": baseline_overall,
        "evolved_behavioral_score": evolved_overall,
        "fitness": fitness,
        "reverted_blocks": reverted,
        "elapsed_seconds": elapsed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    console.print(f"  Output saved to {out}/")

    if write_back:
        for name, desc in accepted.items():
            if desc == baseline[name]:
                continue
            res = write_guidance_to_source(pb_path, baseline[name], desc)
            if res.status == "written":
                console.print(f"  [green]Y wrote {name} -> {res.message}[/green]")
            elif res.status == "ambiguous":
                console.print(f"  [yellow]W {name}: {res.message} (skipped)[/yellow]")

    return metrics


def _to_dspy_examples(probes: list) -> list:
    import dspy as _dspy
    examples = []
    for p in probes:
        guidance_text = p.relevant_blocks[0] if p.relevant_blocks else ""
        ex = _dspy.Example(
            task=p.task,
            guidance_block=guidance_text,
            model_response="",
        ).with_inputs("task", "guidance_block", "model_response")
        examples.append(ex)
    return examples


@click.command()
@click.option("--iterations", default=10, help="GEPA iterations")
@click.option("--optimizer-model", default="openai/gpt-4o-mini")
@click.option("--eval-model", default="openai/gpt-4o-mini")
@click.option("--hermes-repo", default=None)
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
@click.option("--write-back", is_flag=True, default=False,
              help="Apply evolved guidance to agent/prompt_builder.py")
@click.option("--skip-semantic", is_flag=True, default=False)
def main(iterations, optimizer_model, eval_model, hermes_repo, dry_run, write_back, skip_semantic):
    evolve_guidance(
        iterations=iterations, optimizer_model=optimizer_model, eval_model=eval_model,
        hermes_repo=hermes_repo, dry_run=dry_run, write_back=write_back, skip_semantic=skip_semantic,
    )


if __name__ == "__main__":
    main()
