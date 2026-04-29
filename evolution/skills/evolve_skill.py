"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import click
import dspy
import litellm
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, MINIMAX_MODELS, validate_model_string
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external, scrub_secrets
from evolution.core.fitness import skill_fitness_metric, init_fitness_metric, reset_fitness_metric
from evolution.core.constraints import ConstraintValidator
from evolution.core.benchmark_gate import BenchmarkGate
from evolution.monitor.progress import start_run, log_event, complete_run, fail_run
from evolution.skills.skill_module import (
    SkillModule,
    SKILL_BODY_START,
    SKILL_BODY_END,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()

_DATASET_STALE_DAYS = 7


def _warn_stale_datasets(dataset_dir: Path) -> None:
    """Warn if existing JSONL dataset files are older than _DATASET_STALE_DAYS."""
    if not dataset_dir.exists():
        return
    now = datetime.now()
    for jsonl in dataset_dir.glob("*.jsonl"):
        age = now - datetime.fromtimestamp(jsonl.stat().st_mtime)
        if age > timedelta(days=_DATASET_STALE_DAYS):
            console.print(
                f"[yellow]⚠ {jsonl.name} is {age.days} days old — may contain "
                f"stale transcript-derived content. Consider deleting "
                f"{dataset_dir}/ and re-mining.[/yellow]"
            )
            break


# Default per-request timeout for every LLM call DSPy makes. Without an
# explicit value, litellm waits forever on silent connection drops, hanging
# the whole optimization loop. 90s is generous for sonnet/opus reasoning calls
# while still cutting a hung connection. Override with LITELLM_REQUEST_TIMEOUT.
litellm.request_timeout = float(os.environ.get("LITELLM_REQUEST_TIMEOUT", "90"))


# ── Validation helpers ────────────────────────────────────────────────────


def _baseline_validation_text(skill: dict) -> str:
    return skill["raw"]


def _evolved_validation_text(skill: dict, evolved_body: str) -> str:
    return reassemble_skill(skill["frontmatter"], evolved_body)


def validate_skill_constraints(
    validator: ConstraintValidator,
    skill: dict,
    evolved_body: Optional[str] = None,
) -> list:
    """Validate a skill while keeping budget checks on the body only.

    Size, growth, and non-empty checks apply to the mutable markdown body.
    Structural validation applies to the full SKILL.md (frontmatter + body).
    """
    artifact_body = skill["body"] if evolved_body is None else evolved_body
    baseline_body = None if evolved_body is None else skill["body"]
    full_skill_text = (
        _baseline_validation_text(skill)
        if evolved_body is None
        else _evolved_validation_text(skill, evolved_body)
    )

    results = [
        result
        for result in validator.validate_all(artifact_body, "skill", baseline_text=baseline_body)
        if result.constraint_name != "skill_structure"
    ]
    results.append(validator._check_skill_structure(full_skill_text))
    return results


def _is_successful_improvement(baseline_text: str, evolved_text: str, improvement: float) -> bool:
    """Return True iff optimization produced a real artifact change AND a score win."""
    return evolved_text != baseline_text and improvement > 0


# ── Evolved-skill extraction ──────────────────────────────────────────────


class _ExtractionResult:
    __slots__ = ("body", "extraction_failed", "reason")

    def __init__(self, body: str, *, extraction_failed: bool = False, reason: str = ""):
        self.body = body
        self.extraction_failed = extraction_failed
        self.reason = reason


def _extract_evolved_body(optimized_module, baseline_body: str) -> _ExtractionResult:
    """Pull the evolved skill body out of the optimizer's signature.

    Modern DSPy optimizers (GEPA / MIPROv2) mutate the predictor's
    signature.instructions string. The skill body is delimited by HTML
    comment sentinels so it can be recovered even when the surrounding
    wrapper text is rewritten or when the body itself contains markdown
    horizontal rules (`---`).
    """
    instructions = ""
    for _name, predictor in optimized_module.predictor.named_predictors():
        instructions = getattr(predictor.signature, "instructions", "") or ""
        break

    if not instructions:
        sig = getattr(optimized_module.predictor, "signature", None)
        if sig is not None:
            instructions = getattr(sig, "instructions", "") or ""

    if not instructions:
        return _ExtractionResult(
            baseline_body,
            extraction_failed=True,
            reason="optimizer returned no instructions string",
        )

    # Preferred path — sentinel-delimited body.
    start = instructions.find(SKILL_BODY_START)
    end = instructions.find(SKILL_BODY_END)
    if start != -1 and end != -1 and end > start:
        body = instructions[start + len(SKILL_BODY_START):end].strip()
        if body:
            return _ExtractionResult(body)
        return _ExtractionResult(
            baseline_body,
            extraction_failed=True,
            reason="sentinels present but body between them was empty",
        )

    # Legacy fallback — `\n\n---\n` separator. Kept for backward compat with
    # SkillModule(treat_as_untrusted=False) call sites.
    legacy_separator = "\n\n---\n"
    legacy_header = "Follow these skill instructions to complete the task:\n\n"
    if legacy_header in instructions:
        rest = instructions.split(legacy_header, 1)[1]
        body = rest.split(legacy_separator, 1)[0] if legacy_separator in rest else rest
        body = body.strip()
        if body:
            return _ExtractionResult(body)

    # Final fallback — return whole instructions as-is (rare, indicates the
    # optimizer rewrote the wrapper too). Flag as a partial extraction so the
    # caller can warn and not silently treat it as a no-op.
    return _ExtractionResult(
        instructions.strip() or baseline_body,
        extraction_failed=not bool(instructions.strip()),
        reason="sentinels missing from optimizer output — used raw instructions",
    )


# ── Proposal output ───────────────────────────────────────────────────────


def _write_proposal_bundle(
    output_dir: Path,
    skill_name: str,
    skill_path: Path,
    baseline_text: str,
    evolved_text: str,
    metrics: dict,
    constraint_results: list,
    benchmark_results: list,
) -> Path:
    """Write a proposal bundle: baseline, evolved, metrics, decision, diff.

    This is the filesystem-only realisation of `create_pr=True`. A reviewer
    can inspect `proposals/<skill>/<timestamp>/` and decide whether to copy
    `evolved_skill.md` into the hermes-agent skills/ tree by hand, or wire a
    follow-up `gh pr create` step. We deliberately do not run git ops or push
    anything — that requires explicit user authorization for each run.
    """
    proposal_dir = output_dir / "proposals" / skill_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    proposal_dir.mkdir(parents=True, exist_ok=True)

    (proposal_dir / "baseline_skill.md").write_text(baseline_text)
    (proposal_dir / "evolved_skill.md").write_text(evolved_text)
    (proposal_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    decision = {
        "skill_name": skill_name,
        "source_path": str(skill_path),
        "constraint_results": [
            {"name": c.constraint_name, "passed": c.passed, "message": c.message}
            for c in constraint_results
        ],
        "benchmark_results": [
            {
                "name": b.gate_name,
                "skipped": b.skipped,
                "passed": b.passed,
                "regression": b.regression,
                "threshold": b.threshold,
                "message": b.message,
            }
            for b in benchmark_results
        ],
        "improvement": metrics.get("improvement"),
        "successful_improvement": metrics.get("successful_improvement"),
    }
    (proposal_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    # Unified diff for human review.
    import difflib
    diff_lines = difflib.unified_diff(
        baseline_text.splitlines(keepends=True),
        evolved_text.splitlines(keepends=True),
        fromfile=f"a/{skill_path.name}",
        tofile=f"b/{skill_path.name}",
    )
    (proposal_dir / "diff.patch").write_text("".join(diff_lines))

    return proposal_dir


# ── Main entry point ──────────────────────────────────────────────────────


_USE_MINIMAX_DEFAULT = f"minimax/{MINIMAX_MODELS[0]}"


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    use_minimax: bool = False,
    use_llm_judge: bool = False,
    create_pr: bool = False,
    consent_external_ingest: bool = False,
    output_dir: Optional[str] = None,
    source_project: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Main evolution function — orchestrates the full optimization loop."""

    # ── Validate user-supplied model strings ──────────────────────────
    for label, m in [("optimizer", optimizer_model), ("eval", eval_model), ("judge", judge_model)]:
        if m is not None:
            try:
                validate_model_string(m)
            except ValueError as exc:
                console.print(f"[red]✗ Invalid --{label}-model: {exc}[/red]")
                sys.exit(1)

    # ── Resolve model defaults ──────────────────────────────────────────
    # `--use-minimax` only sets defaults for models the user did not specify.
    # This avoids the security-review's H3: silently routing user-supplied
    # OpenAI strings to api.minimax.io (jurisdictional surprise).
    cfg_defaults = EvolutionConfig()
    if use_minimax:
        optimizer_model = optimizer_model or _USE_MINIMAX_DEFAULT
        eval_model = eval_model or _USE_MINIMAX_DEFAULT
        judge_model = judge_model or _USE_MINIMAX_DEFAULT
    else:
        optimizer_model = optimizer_model or cfg_defaults.optimizer_model
        eval_model = eval_model or cfg_defaults.eval_model
        judge_model = judge_model or eval_model

    # ── Build config ────────────────────────────────────────────────────
    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        run_pytest=run_tests,
        create_pr=create_pr,
        api_base=api_base,
        api_key=api_key,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo).expanduser()
    if output_dir:
        config.output_dir = Path(output_dir).expanduser()

    # ── 0. Register run in progress DB ──────────────────────────────────
    run_meta = start_run(skill_name, config)
    run_id = run_meta["run_id"]

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(
            f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]"
        )
        fail_run(run_id, f"Skill '{skill_name}' not found")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")
    log_event(run_id, "loading", f"Loaded {skill_path.name} ({len(skill['raw']):,} chars)")

    # ── 1b. Consent gate for external ingest ───────────────────────────
    # Enforced before --dry-run so users learn about the requirement during
    # setup validation rather than being surprised on a real run.
    if eval_source == "sessiondb" and not consent_external_ingest:
        console.print(
            "[red]✗ --eval-source sessiondb reads chat transcripts from "
            "~/.claude, ~/.copilot, and ~/.hermes.[/red]\n"
            "[red]  This includes ALL projects in those directories, not just the current one.[/red]\n"
            "[red]  Transcripts may contain:[/red]\n"
            "[red]    - Full source code and terminal output[/red]\n"
            "[red]    - Personal data, credentials, and business-confidential content[/red]\n"
            "[red]    - Assistant responses quoting sensitive file contents[/red]\n"
            "[red]  Up to 1000 chars per message will be sent to the configured LLMs[/red]\n"
            f"[red]  for relevance scoring, optimization, and holdout evaluation ({eval_model}).[/red]\n"
        )
        if use_minimax:
            console.print(
                "[red]  ⚠ --use-minimax is active: data will be sent to api.minimax.io[/red]\n"
                "[red]    (MiniMax, a Chinese-jurisdiction provider).[/red]\n"
            )
        console.print(
            "[red]  Re-run with --consent-external-ingest to proceed.[/red]\n"
            "[red]  Use --source-project <name> to limit mining to a specific project.[/red]"
        )
        sys.exit(2)

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints, run benchmarks, write proposal (create_pr={create_pr})")
        complete_run(run_id, {"scoring_method": "dry_run", "constraints_passed": 0})
        return

    # ── 2. Build or load evaluation dataset ────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path).expanduser())
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = (
            Path(dataset_path).expanduser()
            if dataset_path
            else config.output_dir / "datasets" / "skills" / skill_name
        )
        _warn_stale_datasets(save_path)
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
            source_project=source_project,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(artifact_text=skill["raw"], artifact_type="skill")
        save_path = config.output_dir / "datasets" / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path).expanduser())
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(
        f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout"
    )
    log_event(
        run_id,
        "dataset_built",
        f"source={eval_source} train={len(dataset.train)} val={len(dataset.val)} holdout={len(dataset.holdout)}",
    )

    # ── 3. Validate constraints on baseline ────────────────────────────
    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validate_skill_constraints(validator, skill)
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False
    if not all_pass:
        console.print(
            "[yellow]⚠ Baseline skill has constraint violations — proceeding anyway, "
            "but evolved-vs-baseline comparison will be unreliable[/yellow]"
        )

    # ── 4. Set up DSPy + GEPA optimizer ────────────────────────────────
    console.print("\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")
    console.print(
        f"  Scoring: {'LLM-as-judge (LLMJudge)' if use_llm_judge else 'deterministic multi-signal'}"
    )

    # Wire the metric. When `use_llm_judge=True`, an LLMJudge with
    # completeness rubric is used; otherwise the deterministic multi-signal
    # scorer runs. Either way, GEPA receives feedback strings it can use for
    # reflective mutation.
    init_fitness_metric(
        config,
        skill_text=skill["body"],
        use_llm_judge=use_llm_judge,
        max_skill_size=config.max_skill_size,
    )

    lm = config.make_lm(eval_model)
    dspy.configure(lm=lm)

    baseline_module = SkillModule(skill["body"])
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ───────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()

    try:
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_metric_calls=iterations * 10,
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
        optimizer_used = "GEPA"
    except (TypeError, AttributeError, ImportError) as exc:
        # GEPA construction errors only — broad `except` was masking real
        # failures from the fitness metric. Fitness errors should bubble.
        console.print(f"[yellow]GEPA not available ({exc.__class__.__name__}: {exc}); falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(metric=skill_fitness_metric, auto="light")
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
        optimizer_used = "MIPROv2"

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s using {optimizer_used}")
    log_event(run_id, "optimization_complete", f"completed in {elapsed:.1f}s using {optimizer_used}")

    # ── 6. Extract evolved skill text ──────────────────────────────────
    extraction = _extract_evolved_body(optimized_module, baseline_body=skill["body"])
    if extraction.extraction_failed:
        console.print(
            f"[yellow]⚠ Could not cleanly extract evolved body ({extraction.reason}); "
            "treating run as no-op for safety[/yellow]"
        )
    evolved_body = extraction.body or skill["body"]

    # Defence-in-depth: scrub any secret-shaped strings the model may have
    # synthesised into the evolved text. This catches paraphrased leaks the
    # input filter would not have seen.
    scrubbed_body = scrub_secrets(evolved_body)
    if scrubbed_body != evolved_body:
        console.print(
            "[yellow]⚠ Evolved skill contained secret-shaped substrings — "
            "redacted before persisting to disk[/yellow]"
        )
        evolved_body = scrubbed_body

    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ──────────────────────────────────────
    console.print("\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validate_skill_constraints(validator, skill, evolved_body=evolved_body)
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    # ── 7b. Optional pytest gate against the hermes-agent test suite ──
    test_result = None
    if config.run_pytest:
        console.print("\n[bold]Running hermes-agent test suite as gate[/bold]")
        test_result = validator.run_test_suite(config.hermes_agent_path)
        icon = "✓" if test_result.passed else "✗"
        color = "green" if test_result.passed else "red"
        console.print(f"  [{color}]{icon} {test_result.constraint_name}[/{color}]: {test_result.message}")
        if test_result.details:
            console.print(f"    {test_result.details}")
        if not test_result.passed:
            all_pass = False

    # ── 7c. Optional benchmark-regression gate ────────────────────────
    benchmark_results = BenchmarkGate(config).run_all(
        baseline_skill_path=skill_path,
        evolved_skill_text=evolved_full,
    )
    if benchmark_results:
        console.print("\n[bold]Running benchmark gates[/bold]")
        for b in benchmark_results:
            icon = "✓" if b.passed else "✗"
            color = "green" if b.passed else ("yellow" if b.skipped else "red")
            console.print(f"  [{color}]{icon} {b.display_message}[/{color}]")
            if not b.passed and not b.skipped:
                all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED gates — not deploying[/red]")
        failed_dir = config.output_dir / skill_name / "FAILED"
        failed_dir.mkdir(parents=True, exist_ok=True)
        (failed_dir / "evolved_skill.md").write_text(evolved_full)
        console.print(f"  Saved failed variant to {failed_dir}/evolved_skill.md")
        fail_run(run_id, "Evolved skill failed gates")
        return

    # ── 8. Evaluate on holdout set ─────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score_pred = skill_fitness_metric(ex, baseline_pred)
            baseline_scores.append(_score_value(baseline_score_pred))

            evolved_pred = optimized_module(task_input=ex.task_input)
            evolved_score_pred = skill_fitness_metric(ex, evolved_pred)
            evolved_scores.append(_score_value(evolved_score_pred))

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline
    successful_improvement = _is_successful_improvement(skill["raw"], evolved_full, improvement)

    # ── 8b. MAD confidence on per-example deltas ───────────────────────
    # Median Absolute Deviation tells us whether the improvement is
    # statistically meaningful or within the noise of the LLM judge.
    mad_label = ""
    mad_confidence = None
    mad_delta_value = None
    if len(baseline_scores) >= 2:
        from evolution.core.mad_scoring import compute_mad

        deltas = [e - b for e, b in zip(evolved_scores, baseline_scores)]
        mad_delta_value = compute_mad(deltas)
        if mad_delta_value > 0:
            mad_confidence = abs(improvement) / mad_delta_value
        else:
            mad_confidence = float("inf") if abs(improvement) > 0 else 0.0
        if mad_confidence >= 2.0:
            mad_label = "likely real"
        elif mad_confidence >= 1.0:
            mad_label = "marginal"
        else:
            mad_label = "within noise"

    # ── 9. Report results ──────────────────────────────────────────────
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
    if mad_confidence is not None:
        confidence_color = (
            "green" if mad_label == "likely real"
            else "yellow" if mad_label == "marginal"
            else "red"
        )
        confidence_display = (
            f"{mad_confidence:.2f}x" if mad_confidence != float("inf") else "∞"
        )
        table.add_row(
            "MAD Confidence",
            "",
            f"[{confidence_color}]{confidence_display} ({mad_label})[/{confidence_color}]",
            "",
        )

    console.print()
    console.print(table)

    # ── 10. Save output ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / skill_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evolved_skill.md").write_text(evolved_full)
    (run_dir / "baseline_skill.md").write_text(skill["raw"])

    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "judge_model": judge_model,
        "scoring_method": "llm_judge" if use_llm_judge else "deterministic_multi_signal",
        "optimizer_used": optimizer_used,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "successful_improvement": successful_improvement,
        "extraction_failed": extraction.extraction_failed,
        "extraction_reason": extraction.reason,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "test_suite_passed": getattr(test_result, "passed", None) if test_result else None,
        "mad_confidence": mad_confidence if mad_confidence not in (None, float("inf")) else None,
        "mad_delta": mad_delta_value,
        "mad_label": mad_label or None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    console.print(f"\n  Output saved to {run_dir}/")

    # ── 11. Optional proposal bundle ───────────────────────────────────
    if config.create_pr and successful_improvement:
        proposal_dir = _write_proposal_bundle(
            output_dir=config.output_dir,
            skill_name=skill_name,
            skill_path=skill_path,
            baseline_text=skill["raw"],
            evolved_text=evolved_full,
            metrics=metrics,
            constraint_results=evolved_constraints,
            benchmark_results=benchmark_results,
        )
        console.print(
            f"\n[bold green]✓ Proposal bundle written to {proposal_dir}/[/bold green]"
        )
        console.print(
            "  Review baseline_skill.md / evolved_skill.md / diff.patch / decision.json,"
        )
        console.print(
            f"  then copy evolved_skill.md to {skill_path} (or open a PR) if you approve."
        )

    # Always reset global judge state so subsequent calls / tests start clean.
    reset_fitness_metric()

    complete_run(run_id, {
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "constraints_passed": 1 if all_pass else 0,
        "scoring_method": "llm_judge" if use_llm_judge else "deterministic_multi_signal",
    })

    if successful_improvement:
        console.print(
            f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} "
            f"({improvement / max(0.001, avg_baseline) * 100:+.1f}%)[/bold green]"
        )
    elif evolved_full == skill["raw"]:
        console.print(
            f"\n[yellow]⚠ Evolution produced no artifact change "
            f"(score delta: {improvement:+.3f})[/yellow]"
        )
    else:
        console.print(
            f"\n[yellow]⚠ Evolution did not improve skill "
            f"(change: {improvement:+.3f})[/yellow]"
        )


def _score_value(pred) -> float:
    """Coerce a metric return to a plain float, accepting either a Prediction or a number."""
    if isinstance(pred, (int, float)):
        return float(pred)
    val = getattr(pred, "score", None)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option(
    "--eval-source",
    default="synthetic",
    type=click.Choice(["synthetic", "golden", "sessiondb"]),
    help="Source for evaluation dataset",
)
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default=None, help="Model for GEPA reflections")
@click.option("--eval-model", default=None, help="Model for evaluations")
@click.option("--judge-model", default=None, help="Model for LLM-as-judge scoring (defaults to --eval-model)")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo (defaults to ~/.hermes/hermes-agent)")
@click.option("--run-tests", is_flag=True, help="Run hermes-agent pytest suite as a constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--use-minimax", is_flag=True, help="Default optimizer/eval/judge to MiniMax (requires MINIMAX_API_KEY); user-supplied --*-model still wins")
@click.option("--use-llm-judge", is_flag=True, help="Use LLMJudge for scoring instead of the deterministic multi-signal metric")
@click.option("--create-pr", is_flag=True, help="On success, write a proposal bundle (baseline/evolved/diff/decision) for human review")
@click.option(
    "--consent-external-ingest",
    is_flag=True,
    help="Required when --eval-source sessiondb; acknowledges that local chat transcripts will be sent to the eval/judge LLM",
)
@click.option("--output-dir", default=None, help="Override output directory (default: ./output)")
@click.option(
    "--source-project",
    default=None,
    help="Limit Claude Code transcript mining to sessions from this project directory name",
)
@click.option(
    "--api-base",
    default=None,
    help="Custom API base URL for local models (vLLM, Ollama, LiteLLM-compatible endpoints)",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for the custom --api-base endpoint",
)
def main(**kwargs):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(skill_name=kwargs.pop("skill"), **kwargs)


if __name__ == "__main__":
    main()
