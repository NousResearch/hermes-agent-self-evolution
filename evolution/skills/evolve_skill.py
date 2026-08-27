"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/

The run is structured so that the thing being optimized, the thing being
measured, and the thing being reported are all the same thing:

  * the size budget comes from the installed skill corpus, not a constant;
  * the objective carries that budget, so the optimizer feels size pressure
    during the search rather than meeting it at a gate afterwards;
  * one metric drives GEPA *and* the holdout comparison, so the reported
    delta is the quantity that was optimized;
  * the verdict is stated against a noise band, and a run that cannot be
    evaluated exits non-zero and says why.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
import litellm
from rich.console import Console
from rich.table import Table

from evolution.core.config import (
    EvolutionConfig,
    resolve_hermes_agent_path,
    skill_search_roots,
)
from evolution.core.agent_runner import (
    AgentEvaluator,
    HermesAgentBackend,
    tasks_from_examples,
)
from evolution.core.corpus import derive_size_budget, measure_corpus
from evolution.core.dataset_builder import (
    SyntheticDatasetBuilder,
    EvalDataset,
    GoldenDatasetLoader,
)
from evolution.core.external_importers import (
    NoSessionDataError,
    build_dataset_from_external,
)
from evolution.core.fitness import LLMJudge, make_fitness_metric
from evolution.core.constraints import ConstraintValidator
from evolution.core.dspy_lm import make_dspy_lm
from evolution.core.hermes_paths import HermesInstallNotFound, try_find_hermes_install
from evolution.core.notify import Notifier, RunSummary
from evolution.core.outcome_signals import skill_production_health
from evolution.core.objectives import ObjectiveVector, select_best, summarize_front
from evolution.core.report import ABReport, arm_from_eval_run, arm_from_scores
from evolution.core.skill_bundle import load_bundle
from evolution.deploy.canary import CanaryLedger, deploy_canary
from evolution.deploy.pr import PRPublisher, build_pr_body
from evolution.skills.skill_module import (
    SkillModule,
    bump_version,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


# Default per-request timeout for every LLM call DSPy makes. Without an explicit
# value, litellm waits forever on silent connection drops (some providers /
# corporate gateways drop long-poll requests without sending a TCP RST), and the
# whole optimization loop hangs. 90s is generous for sonnet/opus reasoning calls
# while still cutting a hung connection. Override with LITELLM_REQUEST_TIMEOUT.
litellm.request_timeout = float(os.environ.get("LITELLM_REQUEST_TIMEOUT", "90"))


class EvolutionError(RuntimeError):
    """A run could not complete. Carries a message fit for a notification."""


def _is_successful_improvement(baseline_text: str, evolved_text: str, improvement: float) -> bool:
    """Return True only when optimization produced a real artifact change and a score win."""
    return evolved_text != baseline_text and improvement > 0


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    hermes_data_dir: Optional[str] = None,
    profile: Optional[str] = None,
    run_tests: bool = False,
    canary: bool = False,
    agent_eval: bool = False,
    agent_eval_reps: int = 1,
    dry_run: bool = False,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    size_percentile: int = 90,
    create_pr: bool = False,
    pr_base: str = "main",
    pr_draft: bool = True,
    push: bool = True,
    output_root: Optional[str] = None,
) -> dict:
    """Run the full optimization loop. Returns a result dict.

    Raises:
        EvolutionError: When the run cannot produce a usable result. The caller
            turns this into a non-zero exit and a notification — never a silent
            empty output.
    """
    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        hermes_data_dir=hermes_data_dir,
        profiles=[profile] if profile else None,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
        agent_eval=agent_eval,
        agent_eval_reps=agent_eval_reps,
        api_base=api_base,
        api_key=api_key,
        size_percentile=size_percentile,
        create_pr=create_pr,
        pr_base_branch=pr_base,
        pr_draft=pr_draft,
    )
    if output_root:
        config.output_dir = Path(output_root)

    console.print(
        f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — "
        f"Evolving skill: [bold]{skill_name}[/bold]\n"
    )

    # ── 1. Locate the Hermes install and the skill ──────────────────────
    install = None
    try:
        install = try_find_hermes_install(hermes_data_dir)
    except HermesInstallNotFound as exc:
        raise EvolutionError(str(exc)) from exc

    if install:
        console.print(f"  Hermes data: {install.root} (via {install.source})")
    elif eval_source == "sessiondb":
        raise EvolutionError(
            "--eval-source sessiondb needs a Hermes data directory but none was "
            "found. Set HERMES_DATA_DIR (inside the container this is not $HOME)."
        )

    roots = skill_search_roots(config, install, profile)
    if not roots:
        raise EvolutionError(
            "No skills directories found. Checked the hermes-agent repo and the "
            "Hermes data dir; set --hermes-repo and/or --hermes-data-dir."
        )

    skill_path = find_skill(skill_name, *roots)
    if not skill_path:
        searched = "\n    ".join(str(r) for r in roots)
        raise EvolutionError(
            f"Skill '{skill_name}' not found. Searched:\n    {searched}"
        )

    skill = load_skill(skill_path)
    bundle = load_bundle(skill_path, name=skill_name)

    console.print(f"  Loaded: {skill_path}")
    console.print(f"  Name: {skill['name'] or skill_name}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Bundle: {bundle.describe()}")
    if skill["description"]:
        console.print(f"  Description: {skill['description'][:80]}...")

    if install:
        health = skill_production_health(install, skill_name)
        if health.has_evidence:
            console.print(f"  Production health: {health.describe()}")

    # ── 2. Derive the size budget from the real corpus ──────────────────
    corpus_roots = [r for r in roots]
    corpus = measure_corpus(*corpus_roots)
    size_budget, budget_reason = derive_size_budget(
        baseline_chars=len(skill["raw"]),
        stats=corpus,
        percentile=config.size_percentile,
        fallback=config.max_skill_size,
    )
    console.print(f"  Size budget: {size_budget:,} chars ({budget_reason})")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would gate on constraints, size budget {size_budget:,}")
        console.print(f"  Would {'open a PR' if create_pr else 'write output only'}")
        return {"skill_name": skill_name, "dry_run": True, "size_budget": size_budget}

    # ── 3. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")
    dataset = _build_dataset(
        config=config,
        skill=skill,
        skill_name=skill_name,
        eval_source=eval_source,
        dataset_path=dataset_path,
        install=install,
        profile=profile,
    )
    console.print(
        f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / "
        f"{len(dataset.holdout)} holdout"
    )
    if not dataset.train:
        raise EvolutionError("Dataset has no training examples after splitting.")

    # ── 4. Validate constraints on the baseline ─────────────────────────
    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config, size_budget=size_budget)
    for c in validator.validate_all(skill["raw"], "skill", bundle=bundle):
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "yellow"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")

    # ── 5. Configure the optimizer ──────────────────────────────────────
    console.print("\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    lm = make_dspy_lm(eval_model, api_base=config.api_base, api_key=config.api_key)
    reflection_lm = make_dspy_lm(
        optimizer_model,
        temperature=1.0,
        max_tokens=3000,
        api_base=config.api_base,
        api_key=config.api_key,
    )
    dspy.configure(lm=lm)

    baseline_module = SkillModule(
        skill["body"],
        bundle_context=bundle.context_for_optimizer() if bundle.is_bundle else "",
    )

    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # One metric, used for the search and for the holdout comparison. These
    # were different functions before — GEPA optimized a judge composite while
    # the reported delta came from keyword overlap — so the headline number
    # measured something nobody was optimizing.
    judge = LLMJudge(config)
    search_vectors: list[ObjectiveVector] = []
    metric = make_fitness_metric(
        config=config,
        baseline_text=skill["body"],
        size_budget=size_budget,
        weights=config.objective_weights,
        judge=judge,
        on_vector=search_vectors.append,
    )

    # ── 6. Run GEPA ─────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()

    try:
        # DSPy >=3.1 replaced max_steps with a metric-call budget, and requires an
        # explicit reflection LM — that's the model that reads execution traces and
        # proposes mutations, so it's what --optimizer-model is for.
        optimizer = dspy.GEPA(
            metric=metric,
            max_full_evals=iterations,
            reflection_lm=reflection_lm,
        )
        optimized_module = optimizer.compile(
            baseline_module, trainset=trainset, valset=valset
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]GEPA unavailable ({e}); falling back to MIPROv2[/yellow]")
        try:
            optimizer = dspy.MIPROv2(metric=metric, auto="light")
            optimized_module = optimizer.compile(baseline_module, trainset=trainset)
        except Exception as fallback_error:  # noqa: BLE001
            raise EvolutionError(
                f"Both GEPA and MIPROv2 failed: {e} / {fallback_error}"
            ) from fallback_error

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 7. Extract and assemble the evolved skill ───────────────────────
    evolved_body = optimized_module.get_evolved_text()
    evolved_frontmatter = bump_version(skill["frontmatter"])
    evolved_full = reassemble_skill(evolved_frontmatter, evolved_body)

    # ── 8. Holdout evaluation ───────────────────────────────────────────
    holdout = dataset.to_dspy_examples("holdout") or dataset.to_dspy_examples("val")
    console.print(f"\n[bold]Evaluating on holdout set ({len(holdout)} examples)[/bold]")

    baseline_arm = evolved_arm = None
    baseline_scores: list[float] = []
    evolved_scores: list[float] = []

    if config.agent_eval:
        # Run the real agent with each skill loaded, rather than scoring a bare
        # completion. Strictly better evidence, and strictly more expensive.
        baseline_run, evolved_run = _score_arms_with_agent(
            config=config,
            holdout=holdout,
            baseline_text=skill["body"],
            evolved_text=evolved_body,
        )
        baseline_arm = arm_from_eval_run("baseline", baseline_run, len(skill["raw"]))
        evolved_arm = arm_from_eval_run("evolved", evolved_run, len(evolved_full))
        baseline_scores = baseline_arm.scores
        evolved_scores = evolved_arm.scores
    else:
        baseline_scores, evolved_scores = _score_arms(
            holdout, baseline_module, optimized_module, metric, lm
        )

    improvement = (
        (sum(evolved_scores) / len(evolved_scores) if evolved_scores else 0.0)
        - (sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0)
    )

    # ── 9. Constraint gate on the evolved artifact ──────────────────────
    console.print("\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(
        evolved_full, "skill", baseline_text=skill["raw"], bundle=bundle
    )
    constraint_lines = []
    failures = []
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        constraint_lines.append(f"{icon} **{c.constraint_name}** — {c.message}")
        if not c.passed:
            failures.append(f"{c.constraint_name}: {c.message}")

    # The test-suite gate. This existed and was never called; --run-tests was
    # accepted, threaded into config, and ignored, so the README's first
    # guardrail was documentation rather than behaviour.
    if config.run_pytest and config.hermes_agent_path:
        console.print("\n[bold]Running hermes-agent test suite[/bold]")
        result = validator.run_test_suite(Path(config.hermes_agent_path))
        icon = "✓" if result.passed else "✗"
        color = "green" if result.passed else "red"
        console.print(f"  [{color}]{icon} test_suite[/{color}]: {result.message}")
        if result.details:
            console.print(f"    {result.details}")
        constraint_lines.append(f"{icon} **test_suite** — {result.message}")
        if not result.passed:
            failures.append(f"test_suite: {result.message}")

    # ── 10. Report against a noise band ─────────────────────────────────
    report = ABReport(
        subject=skill_name,
        baseline=baseline_arm or arm_from_scores("baseline", baseline_scores, len(skill["raw"])),
        evolved=evolved_arm or arm_from_scores("evolved", evolved_scores, len(evolved_full)),
        metric_name=(
            "graded agent runs (real AIAgent, real toolsets)"
            if config.agent_eval
            else "judge composite × size objective (identical to the search metric)"
        ),
        constraints_passed=not failures,
        constraint_failures=failures,
        extra={
            "size budget": f"{size_budget:,} chars ({budget_reason})",
            "optimizer": f"GEPA, {iterations} full evals, {optimizer_model}",
            "eval source": eval_source,
            "bundle": bundle.describe(),
        },
    )
    if search_vectors:
        best_idx = select_best(search_vectors, config.objective_weights)
        if best_idx is not None:
            report.extra["best search vector"] = search_vectors[best_idx].as_dict()
        # The front, not just the winner: it shows what was traded away.
        front = summarize_front(search_vectors)
        report.extra["pareto front"] = f"{len(front)} of {len(search_vectors)} evaluations"

    verdict, reason = report.verdict()
    console.print()
    console.print(_results_table(skill, evolved_body, report, elapsed, iterations))
    console.print(f"\n[bold]Verdict: {verdict}[/bold] — {reason}")
    for caveat in report.auto_caveats():
        console.print(f"  [dim]caveat: {caveat}[/dim]")

    # ── 11. Persist ─────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.resolved_output_dir() / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evolved_skill.md").write_text(evolved_full)
    (output_dir / "baseline_skill.md").write_text(skill["raw"])
    report.write(output_dir)

    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "eval_source": eval_source,
        "size_budget": size_budget,
        "size_budget_reason": budget_reason,
        "baseline_size": len(skill["raw"]),
        "evolved_size": len(evolved_full),
        "improvement": improvement,
        "verdict": verdict,
        "verdict_reason": reason,
        "constraints_passed": not failures,
        "constraint_failures": failures,
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    console.print(f"\n  Output saved to {output_dir}/")

    result = {
        **metrics,
        "output_dir": str(output_dir),
        "evolved_path": str(output_dir / "evolved_skill.md"),
        "skill_path": str(skill_path),
        "report_markdown": report.to_markdown(),
        "constraint_lines": constraint_lines,
        "pr": None,
    }

    # ── 12. Deploy ──────────────────────────────────────────────────────
    if failures:
        console.print("[red]✗ Constraints failed — not deploying[/red]")
        return result

    if verdict != "SHIP":
        console.print(f"[yellow]⚠ Verdict is {verdict} — not deploying[/yellow]")
        return result

    if canary:
        result["canary"] = _deploy_canary(
            install=install,
            skill_name=skill_name,
            skill_path=skill_path,
            evolved_full=evolved_full,
            output_dir=output_dir,
        )

    if create_pr:
        result["pr"] = _open_pr(
            config=config,
            skill_name=skill_name,
            skill_path=skill_path,
            evolved_full=evolved_full,
            report=report,
            constraint_lines=constraint_lines,
            timestamp=timestamp,
            push=push,
        )
    elif not canary:
        console.print(
            "  [dim]Review the diff: "
            f"diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md[/dim]"
        )
        console.print("  [dim]Re-run with --create-pr to open a pull request.[/dim]")

    return result


# ── helpers ─────────────────────────────────────────────────────────────


def _build_dataset(
    config: EvolutionConfig,
    skill: dict,
    skill_name: str,
    eval_source: str,
    dataset_path: Optional[str],
    install,
    profile: Optional[str],
) -> EvalDataset:
    if eval_source == "golden":
        if not dataset_path:
            raise EvolutionError("--eval-source golden requires --dataset-path")
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
        return dataset

    if eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        try:
            dataset = build_dataset_from_external(
                skill_name=skill_name,
                skill_text=skill["raw"],
                sources=["hermes", "claude-code", "copilot"],
                output_path=save_path,
                model=config.eval_model,
                api_base=config.api_base,
                api_key=config.api_key,
                install=install,
                profiles=[profile] if profile else None,
            )
        except NoSessionDataError as exc:
            raise EvolutionError(str(exc)) from exc
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
        return dataset

    if eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(artifact_text=skill["raw"], artifact_type="skill")
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
        return dataset

    if dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
        return dataset

    raise EvolutionError("Specify --dataset-path or use --eval-source synthetic")



def _score_arms_with_agent(config, holdout, baseline_text: str, evolved_text: str):
    """Score both arms by running the real Hermes agent with each skill loaded.

    Fails loudly when the agent is unreachable. Quietly falling back to
    completion-level scoring would report a number the operator believes came
    from real agent runs, which is worse than not running at all.

    Both arms see the identical task list, built once — re-deriving tasks per
    arm would let sampling differences masquerade as a result.
    """
    backend = HermesAgentBackend(
        hermes_repo=Path(config.hermes_agent_path) if config.hermes_agent_path else Path("."),
        model=config.eval_model,
    )
    available, detail = backend.available()
    if not available:
        raise EvolutionError(
            f"--agent-eval was requested but the Hermes agent is unavailable: {detail}. "
            "Point --hermes-repo at a checkout whose run_agent module imports, or "
            "drop --agent-eval to score completions instead."
        )

    tasks = tasks_from_examples(holdout)
    if not tasks:
        raise EvolutionError("--agent-eval needs holdout examples; none were usable.")

    for task in tasks:
        task.toolsets = tuple(config.agent_toolsets)

    evaluator = AgentEvaluator(backend, reps=config.agent_eval_reps)
    console.print(
        f"  Agent evaluation: {len(tasks)} task(s) x {config.agent_eval_reps} rep(s) "
        f"per arm, toolsets {', '.join(config.agent_toolsets)}"
    )
    console.print(f"  Backend: {detail}")

    baseline_run = evaluator.evaluate(baseline_text, tasks, label="baseline")
    evolved_run = evaluator.evaluate(evolved_text, tasks, label="evolved")
    return baseline_run, evolved_run


def _score_arms(holdout, baseline_module, evolved_module, metric, lm):
    """Score both arms with the same metric, same examples, same order."""
    baseline_scores: list[float] = []
    evolved_scores: list[float] = []

    for ex in holdout:
        with dspy.context(lm=lm):
            try:
                baseline_pred = baseline_module(task_input=ex.task_input)
                baseline_scores.append(_metric_score(metric(ex, baseline_pred)))
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [yellow]baseline example failed: {exc}[/yellow]")

            try:
                evolved_pred = evolved_module(task_input=ex.task_input)
                evolved_scores.append(_metric_score(metric(ex, evolved_pred)))
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [yellow]evolved example failed: {exc}[/yellow]")

    return baseline_scores, evolved_scores


def _metric_score(value) -> float:
    """Metrics return a Prediction(score=…) for GEPA, or a bare float."""
    score = getattr(value, "score", value)
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _results_table(skill, evolved_body, report: ABReport, elapsed, iterations) -> Table:
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    delta = report.delta
    color = "green" if delta > 0 else "red"
    noise = "within noise" if report.within_noise else "beyond noise"
    table.add_row(
        "Holdout score",
        f"{report.baseline.mean:.3f}",
        f"{report.evolved.mean:.3f}",
        f"[{color}]{delta:+.3f}[/{color}] ({noise})",
    )
    table.add_row(
        "Skill size",
        f"{len(skill['raw']):,}",
        f"{report.evolved.size_chars:,}",
        f"{report.size_delta:+,} chars",
    )
    table.add_row(
        "Observations",
        str(report.baseline.n),
        str(report.evolved.n),
        f"±{report.noise_band:.3f} noise band",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")
    return table



def _deploy_canary(install, skill_name, skill_path, evolved_full, output_dir) -> dict:
    """Install the variant over the live skill, keeping a backup for rollback.

    Only skills get this. A skill change shows up in cron outcomes and can be
    reverted before much depends on it; evolved *code* is already executing
    inside the agent before any outcome signal exists, which is why Phase 4
    stops at a pull request.
    """
    if install is None:
        console.print(
            "[yellow]⚠ --canary needs a Hermes data directory to record the "
            "deployment against; none was found. Not deploying.[/yellow]"
        )
        return {"deployed": False, "detail": "no Hermes data dir"}

    ledger = CanaryLedger(Path(install.root) / "evolution" / "canary-ledger.json")
    try:
        record = deploy_canary(
            install=install,
            ledger=ledger,
            skill_name=skill_name,
            target_path=Path(skill_path),
            variant_text=evolved_full,
            backup_dir=Path(output_dir) / "canary-backup",
        )
    except OSError as exc:
        console.print(f"[red]✗ Canary deployment failed: {exc}[/red]")
        return {"deployed": False, "detail": str(exc)}

    console.print(f"\n[bold green]✓ Canary deployed[/bold green] to {skill_path}")
    console.print(
        f"  baseline backed up to {record.backup_path}\n"
        f"  pre-deploy success rate: "
        + (f"{record.baseline_success_rate:.0%} over {record.baseline_observations} runs"
           if record.baseline_success_rate is not None else "no prior evidence")
    )
    console.print(
        "  [dim]Check it later: python -m evolution.deploy.canary_cli --evaluate[/dim]"
    )
    return {
        "deployed": True,
        "variant_sha": record.variant_sha,
        "backup": record.backup_path,
        "ledger": str(ledger.path),
    }


def _open_pr(
    config: EvolutionConfig,
    skill_name: str,
    skill_path: Path,
    evolved_full: str,
    report: ABReport,
    constraint_lines: list[str],
    timestamp: str,
    push: bool,
) -> dict:
    """Open a PR against whichever repo actually contains the skill file."""
    repo = _git_root(skill_path)
    if repo is None:
        console.print(
            f"[yellow]⚠ {skill_path} is not inside a git repository — "
            "cannot open a PR. Output is saved for manual review.[/yellow]"
        )
        return {"created": False, "detail": "skill file is not in a git repo"}

    publisher = PRPublisher(
        repo=repo, base_branch=config.pr_base_branch, draft=config.pr_draft
    )
    body = build_pr_body(
        skill_name=skill_name,
        report_markdown=report.to_markdown(),
        constraint_lines=constraint_lines,
        run_metadata={
            "optimizer model": config.optimizer_model,
            "eval model": config.eval_model,
            "iterations": config.iterations,
        },
    )
    verdict, _ = report.verdict()
    title = f"evolve({skill_name}): {report.delta:+.3f} holdout, {verdict}"

    console.print(f"\n[bold]Opening pull request[/bold] against {repo}")
    result = publisher.publish(
        skill_name=skill_name,
        target_path=skill_path,
        content=evolved_full,
        title=title,
        body=body,
        timestamp=timestamp,
        push=push,
    )
    style = "green" if result.created else "yellow"
    console.print(f"  [{style}]{result.render()}[/{style}]")
    return {"created": result.created, "branch": result.branch, "url": result.url, "detail": result.detail}


def _git_root(path: Path) -> Optional[Path]:
    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    return None


# ── CLI ─────────────────────────────────────────────────────────────────


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic",
              type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1",
              help="Model for GEPA reflections. Use openai-codex/<model> to route via Hermes Codex OAuth.")
@click.option("--eval-model", default="openai/gpt-4.1-mini",
              help="Model for evaluations. Use openai-codex/<model> to route via Hermes Codex OAuth.")
@click.option("--hermes-repo", default=None, help="Path to the hermes-agent source repo")
@click.option("--hermes-data-dir", default=None,
              help="Hermes data directory (state.db, profiles, cron). Defaults to "
                   "$HERMES_DATA_DIR, $HERMES_HOME, then ~/.hermes.")
@click.option("--profile", default=None, help="Restrict skills and session mining to one profile")
@click.option("--size-percentile", default=90, type=int,
              help="Corpus percentile used as the size budget (default 90)")
@click.option("--run-tests", is_flag=True, help="Run the hermes-agent pytest suite as a gate")
@click.option("--agent-eval", is_flag=True,
              help="Score the holdout by running the real Hermes agent with each skill "
                   "loaded, instead of a single completion. Slower and far better evidence.")
@click.option("--agent-eval-reps", default=1, type=int,
              help="Repetitions per task per arm. >1 gives a measured noise band.")
@click.option("--canary", is_flag=True,
              help="On a SHIP verdict, install the variant over the live skill with a "
                   "backup, and record it for later evaluation and auto-rollback.")
@click.option("--create-pr", is_flag=True, help="Open a pull request when the verdict is SHIP")
@click.option("--pr-base", default="main", help="Base branch for the pull request")
@click.option("--no-draft", is_flag=True, help="Open a ready-for-review PR instead of a draft")
@click.option("--no-push", is_flag=True, help="Commit the branch locally but do not push")
@click.option("--output-root", default=None, help="Directory for run output (default ./output)")
@click.option("--notify/--no-notify", default=False, help="Send a run summary through configured channels")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--api-base", default=None, help="API base URL (for custom endpoints like vLLM)")
@click.option("--api-key", default=None, help="API key (for custom endpoints)")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model,
         hermes_repo, hermes_data_dir, profile, size_percentile, run_tests, agent_eval,
         agent_eval_reps, canary, create_pr, pr_base, no_draft, no_push, output_root, notify,
         dry_run, api_base, api_key):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    summary = RunSummary(subject=f"Hermes self-evolution — {skill}")

    try:
        result = evolve(
            skill_name=skill,
            iterations=iterations,
            eval_source=eval_source,
            dataset_path=dataset_path,
            optimizer_model=optimizer_model,
            eval_model=eval_model,
            hermes_repo=hermes_repo,
            hermes_data_dir=hermes_data_dir,
            profile=profile,
            run_tests=run_tests,
            canary=canary,
            agent_eval=agent_eval,
            agent_eval_reps=agent_eval_reps,
            dry_run=dry_run,
            api_base=api_base,
            api_key=api_key,
            size_percentile=size_percentile,
            create_pr=create_pr,
            pr_base=pr_base,
            pr_draft=not no_draft,
            push=not no_push,
            output_root=output_root,
        )
    except EvolutionError as exc:
        console.print(f"\n[red]✗ {exc}[/red]")
        summary.failed.append((skill, str(exc)))
        _finish(summary, notify)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — unexpected, but still must be reported
        console.print(f"\n[red]✗ Unexpected failure: {type(exc).__name__}: {exc}[/red]")
        summary.failed.append((skill, f"{type(exc).__name__}: {exc}"))
        _finish(summary, notify)
        raise

    if result.get("dry_run"):
        summary.succeeded.append(f"{skill} (dry run)")
    elif result.get("verdict") == "SHIP":
        summary.succeeded.append(f"{skill} — {result['verdict_reason']}")
    else:
        summary.skipped.append((skill, result.get("verdict_reason", "held")))

    if result.get("output_dir"):
        summary.notes.append(f"Output: {result['output_dir']}")
    canary_info = result.get("canary") or {}
    if canary_info.get("deployed"):
        summary.notes.append(f"Canary deployed; ledger {canary_info.get('ledger')}")

    pr = result.get("pr") or {}
    if pr.get("created"):
        summary.notes.append(f"PR: {pr.get('url') or pr.get('branch')}")
    elif pr.get("detail"):
        summary.notes.append(f"PR not opened: {pr['detail']}")

    _finish(summary, notify)
    sys.exit(summary.exit_code)


def _finish(summary: RunSummary, notify: bool) -> None:
    """Deliver the summary if asked, and always say whether delivery worked."""
    if not notify:
        return
    outcome = Notifier.from_env().send(summary.subject, summary.render())
    style = "green" if outcome.delivered else "red"
    console.print(f"[{style}]Notification: {outcome.render()}[/{style}]")


if __name__ == "__main__":
    main()
