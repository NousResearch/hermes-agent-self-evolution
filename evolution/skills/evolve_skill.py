"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.errors import (
    EXIT_EMFILE,
    EMFILE_HINT,
    classify_error,
    is_emfile,
)
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


def run_holdout_evaluation(
    baseline_module,
    optimized_module,
    holdout_examples,
    metric,
    lm=None,
    samples: int = 3,
) -> tuple[list[float], list[float]]:
    """Score baseline vs evolved module on the holdout set.

    Multi-sample aggregation: each example is scored ``samples`` times and the
    median is used (smooths LLM stochasticity). ``lm`` is applied via
    dspy.context when provided (real LLM runs); None is for tests/fake modules.
    Returns (baseline_scores, evolved_scores) — one median score per example.
    """
    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        ex_baseline_scores = []
        ex_evolved_scores = []
        cm = dspy.context(lm=lm) if lm is not None else nullcontext()
        for _ in range(samples):
            with cm:
                baseline_pred = baseline_module(task_input=ex.task_input)
                ex_baseline_scores.append(metric(ex, baseline_pred))

                evolved_pred = optimized_module(task_input=ex.task_input)
                ex_evolved_scores.append(metric(ex, evolved_pred))

        # Use median (sorted middle) for robustness
        ex_baseline_scores.sort()
        ex_evolved_scores.sort()
        baseline_scores.append(ex_baseline_scores[len(ex_baseline_scores) // 2])
        evolved_scores.append(ex_evolved_scores[len(ex_evolved_scores) // 2])
    return baseline_scores, evolved_scores


def write_text_guarded(path, text: str, label: str) -> None:
    """Write text, converting FD exhaustion into an explicit, classified exit.

    ``path`` is any object with ``.write_text`` (Path or a test fake).

    The 'error 1' escalations were bare ``OSError: [Errno 24] Too many open
    files`` crashes on artifact writes (and the holdout loop) that surfaced as
    exit 1 and got mislabeled "LLM provider issue" by cron-evolve.sh. EMFILE
    now exits 2 with a hint; other OS errors warn and continue.
    """
    try:
        path.write_text(text)
    except OSError as e:
        if is_emfile(e):
            console.print(f"[red]✗ Could not write {label} ({path}): {EMFILE_HINT}[/red]")
            sys.exit(EXIT_EMFILE)
        console.print(f"[yellow]⚠ Could not write {label} ({path}): {e}[/yellow]")


def estimate_improvement(
    baseline_module,
    optimized_module,
    valset_examples,
    metric,
    lm=None,
    samples: int = 3,
    max_examples: int = 12,
) -> float:
    """Estimate (evolved − baseline) score delta on the valset.

    Used only when the growth gate trips, to decide whether the extra size
    buys material quality (see config.growth_waiver_min_improvement). Runs
    with the same deterministic setup as the holdout eval. The example count
    is capped (default 12) so the estimate stays inside the run's wall-clock
    budget; calls are usually cache hits from the GEPA run in the recurring
    deterministic case.
    """
    baseline_scores, evolved_scores = run_holdout_evaluation(
        baseline_module,
        optimized_module,
        valset_examples[:max_examples],
        metric,
        lm=lm,
        samples=samples,
    )
    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    return avg_evolved - avg_baseline

# ── Timeout budget (t_316c92c4) ──────────────────────────────────────────
# The cron harness wraps evolution in `gtimeout 480` (cron-evolve.sh) and
# SIGKILLs the process on expiry — any progress is lost. To fit inside that
# budget we (a) bound every LLM call with a per-call timeout + retry cap,
# (b) guard the holdout phase with an in-process wall-clock budget so it
# degrades to a PARTIAL result instead of a kill, and (c) cache holdout
# scores across runs so warm runs skip the 144-call holdout phase entirely.
DEFAULT_MAX_BUDGET_SECONDS = 480  # must match cron-evolve.sh `gtimeout 480`
DEFAULT_LLM_TIMEOUT_SECONDS = 60  # per-attempt; litellm default is 600s
DEFAULT_LLM_RETRIES = 2           # per-call; dspy default is 8


def make_lm(model: str) -> dspy.LM:
    """Build a dspy.LM with a bounded per-call timeout and retry budget.

    dspy/litellm defaults allow a single stalled call to hang for 600s+ with
    8 retries — longer than the entire 480s evolution budget. A stalled call
    must instead raise quickly so the harness can classify it (API error)
    rather than burn the budget and get SIGKILLed with no output.
    """
    return dspy.LM(
        model,
        timeout=DEFAULT_LLM_TIMEOUT_SECONDS,  # passed through to litellm
        num_retries=DEFAULT_LLM_RETRIES,      # dspy-level retry cap
    )


def _holdout_cache_key(model: str, skill_hash: str, program: str,
                       task_input: str, sample_idx: int) -> str:
    """Stable cache key for one holdout score.

    Includes the skill-body hash so a skill-text change invalidates holdout
    scores (same semantics as the ~/.dspy_cache key), and the model name so
    switching eval models never reuses stale scores.
    """
    raw = f"{model}|{skill_hash}|{program}|{task_input}|{sample_idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def evaluate_holdout(
    holdout_examples,
    baseline_module,
    optimized_module,
    *,
    lm: Optional[dspy.LM] = None,
    samples: int = 3,
    metric=skill_fitness_metric,
    max_budget_seconds: Optional[float] = None,
    score_cache: Optional[dict] = None,
    cache_key=None,
) -> dict:
    """Score baseline vs optimized on the holdout set within a wall-clock budget.

    Replaces the previous unbounded holdout loop (24 examples x 3 samples x
    2 programs = 144 LLM calls, the last and largest phase of a run). When
    ``max_budget_seconds`` is set, the loop stops early and returns the scores
    gathered so far with ``budget_exceeded=True`` — a PARTIAL result the
    caller can report instead of the gtimeout wrapper killing the process.

    When ``score_cache`` + ``cache_key`` are provided, scores are memoized per
    (program, example, sample): warm runs with an unchanged (model, skill)
    pair make zero LLM calls in this phase.

    Returns dict with baseline_scores / evolved_scores (median-aggregated per
    example, complete examples only), examples_evaluated / examples_total,
    budget_exceeded, cache_hits, calls_made.
    """
    score_cache = score_cache if score_cache is not None else {}
    results = {"baseline": [], "evolved": []}
    cache_hits = 0
    calls_made = 0
    budget_exceeded = False
    evaluated = 0
    start = time.monotonic()

    def _remaining() -> float:
        if max_budget_seconds is None:
            return float("inf")
        return max_budget_seconds - (time.monotonic() - start)

    for ex in holdout_examples:
        for program, module in (("baseline", baseline_module), ("evolved", optimized_module)):
            sample_scores = []
            for sample_i in range(samples):
                if _remaining() <= 0:
                    budget_exceeded = True
                    break
                key = cache_key(ex, program, sample_i) if cache_key else None
                if key is not None and key in score_cache:
                    sample_scores.append(score_cache[key])
                    cache_hits += 1
                    continue
                if lm is not None:
                    with dspy.context(lm=lm):
                        pred = module(task_input=ex.task_input)
                else:
                    pred = module(task_input=ex.task_input)
                sample_scores.append(metric(ex, pred))
                calls_made += 1
                if key is not None:
                    score_cache[key] = sample_scores[-1]
            if budget_exceeded:
                break
            # Median aggregation (smooths LLM stochasticity; baseline variance
            # was 5x larger than plausible improvements with 1 sample).
            sample_scores.sort()
            results[program].append(sample_scores[len(sample_scores) // 2])
        if budget_exceeded:
            break
        evaluated += 1

    return {
        "baseline_scores": results["baseline"],
        "evolved_scores": results["evolved"],
        "examples_evaluated": evaluated,
        "examples_total": len(holdout_examples),
        "budget_exceeded": budget_exceeded,
        "cache_hits": cache_hits,
        "calls_made": calls_made,
    }


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    seed: Optional[int] = None,
    max_budget: int = DEFAULT_MAX_BUDGET_SECONDS,
    holdout_samples: int = 3,
):
    """Main evolution function — orchestrates the full optimization loop."""
    _seed = seed if seed is not None else 42
    run_start = time.time()

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )
    if seed is not None:
        config.random_seed = seed
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        save_path = Path("datasets") / "skills" / skill_name

        def _generate():
            return builder.generate(
                artifact_text=skill["raw"],
                artifact_type="skill",
                seed=_seed,
            )

        dataset = EvalDataset.load_or_generate(
            path=save_path,
            generator_fn=_generate,
            seed=_seed,
        )
        console.print(f"  Dataset: {len(dataset.all_examples)} examples (source: synthetic, seed={_seed})")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    # Validate the FULL artifact (frontmatter + body), not just the body —
    # the body-only check made skill_structure fail on every baseline
    # (false negative, harmless "proceeding anyway" warning) and was
    # asymmetric with the evolved check below.
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy — bounded per-call timeout/retries so a single stalled
    # API call can't blow the whole 480s budget (litellm default timeout is
    # 600s; dspy default num_retries is 8).
    lm = make_lm(eval_model)
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    # GEPA requires 5-arg metric: (gold, pred, trace, pred_name, pred_trace)
    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return skill_fitness_metric(gold, pred, trace)

    try:
        optimizer = dspy.GEPA(
            metric=gepa_metric,
            reflection_lm=lm,
            max_full_evals=iterations,
        )

        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(
            metric=skill_fitness_metric,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # The optimized module's instructions contain the evolved skill text
    # (May be overridden by the baseline-score guard in step 8b if the
    # optimizer produced params worse than the default baseline)
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    # Validate the full artifact (frontmatter + body), always against the
    # baseline BODY. (The old conditional compared the evolved artifact to
    # itself when frontmatter was missing — a false pass.)
    evolved_constraints = validator.validate_all(
        evolved_full,
        "skill",
        baseline_text=skill["body"],
    )
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    # Growth-only failure → quality waiver check. The growth gate rejects
    # bloated candidates, but a candidate that grows past the soft cap may
    # still be worth deploying when the extra size buys a material valset
    # improvement (config.growth_waiver_min_improvement) and stays under the
    # hard cap. Measure the improvement, then re-validate with it.
    if not all_pass:
        failed = [c for c in evolved_constraints if not c.passed]
        growth_failed = [c for c in failed if c.constraint_name == "growth_limit"]
        if len(failed) == 1 and len(growth_failed) == 1:
            console.print("[yellow]  Growth gate exceeded — measuring valset improvement for a waiver...[/yellow]")
            improvement_est = estimate_improvement(
                baseline_module,
                optimized_module,
                valset,
                skill_fitness_metric,
                lm=lm,
            )
            console.print(
                f"  Valset improvement vs baseline: {improvement_est:+.3f} "
                f"(waiver needs >= {config.growth_waiver_min_improvement:+.3f})"
            )
            evolved_constraints = validator.validate_all(
                evolved_full,
                "skill",
                baseline_text=skill["body"],
                improvement=improvement_est,
            )
            all_pass = True
            for c in evolved_constraints:
                icon = "✓" if c.passed else "✗"
                color = "green" if c.passed else "red"
                console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
                if not c.passed:
                    all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection (FD-exhaustion-safe write: the original
        # 'error 1' crashed here on OSError 24 and got mislabeled as an API
        # error by the cron wrapper).
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_guarded(output_path, evolved_full, "failed variant")
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set (budget-guarded + score-cached) ──────
    holdout_examples = dataset.to_dspy_examples("holdout")
    holdout_budget = max(0.0, max_budget - (time.time() - run_start))
    console.print(f"\n[bold]Evaluating on holdout set ({len(holdout_examples)} examples, {holdout_samples} samples each)[/bold]")

    # Persistent score cache (checkpointing for the 480s budget): keyed by
    # (eval model, skill-body hash, program, example, sample) so warm runs
    # skip the 144-call holdout phase entirely. The skill hash invalidates
    # the cache when the skill text changes (mirrors ~/.dspy_cache semantics).
    cache_path = Path("output") / skill_name / "holdout_scores.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    score_cache: dict = {}
    if cache_path.exists():
        try:
            score_cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            score_cache = {}
    skill_hash = hashlib.md5(skill["body"].encode("utf-8")).hexdigest()[:12]
    cache_key_fn = lambda ex, program, sample_i: _holdout_cache_key(
        eval_model, skill_hash, program,
        getattr(ex, "task_input", "") or "", sample_i,
    )

    holdout_result = None
    try:
        holdout_result = evaluate_holdout(
            holdout_examples,
            baseline_module,
            optimized_module,
            lm=lm,
            samples=holdout_samples,
            metric=skill_fitness_metric,
            max_budget_seconds=holdout_budget,
            score_cache=score_cache,
            cache_key=cache_key_fn,
        )
    except SystemExit:
        raise
    except Exception as e:
        # The 'error 1' EMFILE crash hit this phase. Classify explicitly:
        # FD exhaustion → exit 2 with a hint; anything else → eval error.
        code, msg = classify_error(e)
        console.print(f"[red]✗ Holdout evaluation failed: {msg}[/red]")
        sys.exit(code)

    # Persist the score cache; a failure here must not kill an otherwise
    # successful run (worst case the next run pays the calls again).
    try:
        cache_path.write_text(json.dumps(score_cache))
    except OSError as e:
        console.print(f"[yellow]⚠ Could not persist holdout score cache: {e}[/yellow]")

    baseline_scores = holdout_result["baseline_scores"]
    evolved_scores = holdout_result["evolved_scores"]
    if holdout_result["budget_exceeded"]:
        console.print(
            f"[yellow]⚠ Holdout budget exhausted — returning PARTIAL result "
            f"({holdout_result['examples_evaluated']}/{holdout_result['examples_total']} "
            f"examples; {holdout_result['cache_hits']} cache hits, "
            f"{holdout_result['calls_made']} live calls)[/yellow]"
        )

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 8b. Baseline-score guard — never deploy params worse than baseline ──
    # Root cause: GEPA/MIPROv2 with few iterations (default 5) can converge to
    # a local minimum worse than the default DSPy parameters, causing regression.
    # This guard discards the optimizer's output when it underperforms the baseline,
    # keeping the original skill text as the "evolved" artifact.
    if improvement < 0:
        console.print(f"\n[yellow]⚠ Optimizer regressed ({improvement:+.3f}) — keeping baseline instead of deploying worse params[/yellow]")
        evolved_body = skill["body"]
        evolved_full = skill["raw"]
        # Update the metrics to reflect we kept the baseline
        avg_evolved = avg_baseline
        improvement = 0.0
        console.print("  → Saved original skill text as the deployed artifact (no regression deployed)")
    # else: evolved_body/evolved_full already set in step 6 above

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evolved skill / baseline / metrics (FD-exhaustion-safe writes)
    write_text_guarded(output_dir / "evolved_skill.md", evolved_full, "evolved skill")
    write_text_guarded(output_dir / "baseline_skill.md", skill["raw"], "baseline skill")

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "holdout_complete": not holdout_result["budget_exceeded"],
        "holdout_examples_evaluated": holdout_result["examples_evaluated"],
        "holdout_examples_total": holdout_result["examples_total"],
        "holdout_cache_hits": holdout_result["cache_hits"],
        "holdout_calls_made": holdout_result["calls_made"],
        "max_budget_seconds": max_budget,
    }
    write_text_guarded(output_dir / "metrics.json", json.dumps(metrics, indent=2), "metrics")

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--seed", default=None, type=int, help="Random seed for reproducible dataset splits (default: 42)")
@click.option("--max-budget", default=DEFAULT_MAX_BUDGET_SECONDS, type=int,
              help=f"Hard wall-clock budget in seconds (default {DEFAULT_MAX_BUDGET_SECONDS}; must match the cron gtimeout wrapper)")
@click.option("--holdout-samples", default=3, type=int,
              help="Samples per holdout example, median-aggregated (default 3)")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run, seed, max_budget, holdout_samples):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        seed=seed,
        max_budget=max_budget,
        holdout_samples=holdout_samples,
    )


if __name__ == "__main__":
    main()
