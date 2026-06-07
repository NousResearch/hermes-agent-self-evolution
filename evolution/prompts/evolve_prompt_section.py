"""Evolve a named system-prompt section in Hermes ``prompt_builder.py`` via DSPy + GEPA.

Mirrors ``evolution.tools.evolve_tool`` but for prompt sections, with the
splice-and-restore integration model (see ``HermesPromptSectionInstaller``).
The whole evaluation is behavioral: every candidate is spliced into the live
``prompt_builder.py`` and scored by a real ``hermes -z`` subprocess, so the
deploy gate is a ``ClosedLoopValidator`` run rather than a synthetic-judge
holdout.

Usage:
    python -m evolution.prompts.evolve_prompt_section \\
        --section MEMORY_GUIDANCE \\
        --hermes-repo ~/src/NousResearch/hermes-agent \\
        --tasks evolution/validation/suites/memory_guidance.jsonl \\
        --iterations 10
"""

from __future__ import annotations

import fcntl
import json
import logging
import random
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import click
import dspy
from rich.console import Console

from evolution.core.config import EvolutionConfig
from evolution.core.hermes_provider import instantiate_lm, resolve_default_lm
from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    CostCeilingExceeded,
    LMTimingCallback,
    register_litellm_cost_callback,
    register_litellm_failure_callback,
)
from evolution.core.pr_automation import disabled_pr_block
from evolution.core.quality_gate import write_gate_decision
from evolution.core.run_inputs import build_run_inputs
from evolution.core.saturation_check import (
    is_non_interactive,
    interactive_confirm,
    render_saturation_panel,
    saturation_preflight,
)
from evolution.prompts.hermes_prompt_source import HermesPromptSource
from evolution.prompts.prompt_judge import (
    SaveCallJudge,
    ScoreResult,
    judge_save_calls,
    make_memoizing_splice_scorer,
    make_prompt_fitness_metric,
)
from evolution.prompts.prompt_module import PromptModule, _extract_from_sentinels
from evolution.prompts.prompt_proposer import PromptSectionProposer
from evolution.validation.agent_runner import TaskRunContext
from evolution.validation.artifact_installer import (
    HermesPromptSectionInstaller,
    atomic_write_bytes,
)
from evolution.validation.hermes_runner import (
    DEFAULT_TASK_TIMEOUT_SECONDS,
    HermesAgentRunner,
)
from evolution.validation.report import score_task
from evolution.validation.task import Task, TaskSuite
from evolution.validation.validator import ClosedLoopValidator, ValidationInputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()

_GATE_SCHEMA_VERSION = "5"
_BACKUP_SUFFIX = ".cl_backup"
_LOCK_FILENAME = ".cl_validation.lock"


def _split_train_holdout(
    tasks: tuple[Task, ...], *, holdout_ratio: float, seed: int
) -> tuple[list[Task], list[Task]]:
    """Deterministic train/holdout split, stratified only by shuffle+seed.

    Guarantees at least one task on each side when there are >= 2 tasks so
    GEPA has something to train on and the deploy gate has something to
    evaluate.
    """
    ordered = list(tasks)
    random.Random(seed).shuffle(ordered)
    n_holdout = max(1, int(round(len(ordered) * holdout_ratio)))
    n_holdout = min(n_holdout, len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    holdout = ordered[:n_holdout]
    train = ordered[n_holdout:]
    return train, holdout


def _behavioral_examples(tasks: list[Task]) -> list[dspy.Example]:
    """Build GEPA examples whose inputs drive ``PromptModule.forward`` into the
    behavioral branch (task message + closed_loop_task_id)."""
    return [
        dspy.Example(
            task=t.user_message,
            closed_loop_task_id=t.task_id,
        ).with_inputs("task", "closed_loop_task_id")
        for t in tasks
    ]


def _make_layer2_factory(judge: Optional[SaveCallJudge]):
    """Per-task Layer 2 scorer: binds the task's rubric + message into a
    ``score_task``-shaped ``Callable[[list[dict]], float]``. Returns ``None``
    for tasks without an ``expected_save_content`` rubric (no content to
    judge)."""

    def factory(task: Task):
        if task.expected_save_content is None:
            return None

        def judge_fn(memory_calls: list[dict]) -> float:
            return judge_save_calls(
                judge=judge,
                calls=memory_calls,
                expected_content=task.expected_save_content,
                task_text=task.user_message,
            )

        return judge_fn

    return factory


_VAL_SIGNAL_LOW = 0.05
_VAL_SIGNAL_HIGH = 0.95


def val_signal_warning(
    holdout_baseline_rates: dict[str, float],
) -> Optional[dict]:
    """Flag a holdout with no discriminating signal for the deploy gate.

    When every holdout task's baseline pass rate sits at one extreme — all
    ≤ 0.05 (uniform failure) or all ≥ 0.95 (uniform success) — the gate has no
    gradient to measure improvement against, so a "pass" is uninformative.
    Returns a warning dict (offending task ids + their rates) in that case; a
    single mid-range task is enough signal, so any rate strictly between the
    bounds returns ``None``. Empty input → ``None``.
    """
    if not holdout_baseline_rates:
        return None

    rates = holdout_baseline_rates
    all_low = all(r <= _VAL_SIGNAL_LOW for r in rates.values())
    all_high = all(r >= _VAL_SIGNAL_HIGH for r in rates.values())
    if not (all_low or all_high):
        return None

    kind = "uniform_failure" if all_low else "uniform_success"
    return {
        "kind": kind,
        "reason": (
            "All holdout baseline pass rates are at one extreme "
            f"({'≤ %.2f' % _VAL_SIGNAL_LOW if all_low else '≥ %.2f' % _VAL_SIGNAL_HIGH}); "
            "the deploy gate has no discriminating gradient on this suite."
        ),
        "task_ids": sorted(rates.keys()),
        "rates": dict(rates),
    }


def _section_text_from_candidate(candidate: Any, section_name: str) -> str:
    """Extract the section body from a GEPA-built candidate (module or
    component dict), reading the sentinel-delimited region."""
    if isinstance(candidate, dict):
        instructions = candidate.get("passthrough.predict", "")
    else:
        instructions = candidate.passthrough.predict.signature.instructions or ""
    return _extract_from_sentinels(instructions, section_name)


@contextmanager
def _prompt_builder_guard(target_path: Path) -> Iterator[None]:
    """Back up ``prompt_builder.py`` + hold the shared closed-loop flock for the
    duration of the saturation pre-flight + GEPA evolution, then restore the
    original bytes on exit.

    The pre-flight and GEPA inner loop splice candidates directly into the live
    file; this
    guard guarantees the user's checkout is byte-restored afterward and that no
    concurrent harness run (which uses the same lock + backup names) mutates it
    mid-flight. Sequenced before the deploy-gate ``ClosedLoopValidator``, which
    acquires the same lock itself — never nested.
    """
    backup_path = target_path.with_suffix(target_path.suffix + _BACKUP_SUFFIX)
    if backup_path.exists():
        raise RuntimeError(
            f"Stale backup at {backup_path} — a prior run did not clean up. "
            f"Restore {target_path} from it manually, then retry."
        )
    lock_fd = open(target_path.parent / _LOCK_FILENAME, "w")
    try:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another harness run holds {target_path.parent / _LOCK_FILENAME}. "
                f"Wait for it to finish."
            ) from exc
        atomic_write_bytes(backup_path, target_path.read_bytes())
        try:
            yield
        finally:
            atomic_write_bytes(target_path, backup_path.read_bytes())
            backup_path.unlink(missing_ok=True)
    finally:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()


def _run_one_task_score(
    task: Task,
    *,
    runner: HermesAgentRunner,
    layer2_factory,
    layer2_threshold: float,
    reps: int = 1,
    suite_dir: Optional[Path] = None,
) -> ScoreResult:
    """Run a task through the agent ``reps`` times, returning mean pass rate.

    Abstentions are excluded from the denominator; all-abstain scores 0.0.
    reps=1 reproduces the legacy single-run verdict (score ∈ {0.0, 1.0}).
    """
    n_pass = 0
    n_abstain = 0

    # ``skills_src`` is a path relative to the suite file's directory.
    skills_src = (
        (suite_dir / task.skills_src)
        if (task.skills_src and suite_dir is not None)
        else None
    )

    for _ in range(reps):
        with tempfile.TemporaryDirectory(prefix="ps_inner_") as fixture_tmp:
            fixture_dir = Path(fixture_tmp)
            for relative_path, content in task.fixture_setup.items():
                dest = fixture_dir / relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
            ctx = TaskRunContext(
                user_message=task.render_message(fixture_dir),
                fixture_dir=fixture_dir,
                skills_src=skills_src,
            )
            run = runner.run(ctx)
            passed, abstained = score_task(
                expected_tools=task.expected_tools,
                forbidden_tools=task.forbidden_tools,
                run=run,
                test_command=task.test_command,
                fixture_dir=fixture_dir,
                layer2_judge_fn=layer2_factory(task),
                layer2_threshold=layer2_threshold,
                expected_action=task.expected_action,
                target_skill=task.target_skill,
                stale_token=task.stale_token,
            )
            if abstained:
                n_abstain += 1
            elif passed:
                n_pass += 1

    scored = reps - n_abstain
    rate = (n_pass / scored) if scored else 0.0

    observed = (
        f"all {reps} runs abstained" if scored == 0
        else f"passed {n_pass}/{scored}"
        + (f" ({n_abstain} abstained)" if n_abstain else "")
    )
    return ScoreResult(score=rate, feedback=_synth_feedback(task, observed))


def _synth_feedback(task: Task, observed: str) -> str:
    """Outcome-grounded feedback for GEPA's reflection LM. For action tasks it
    states the patch objective + the stale token the skill carries (so the LM
    knows what behavior to instill); for controls it states the do-not-act
    objective. Neutral — describes the eval rubric and observed behavior, never
    any target-prompt wording."""
    if task.expected_action and task.target_skill:
        return (
            f"Objective: while doing the task the agent uses the "
            f"'{task.target_skill}' skill, whose written instructions are STALE "
            f"(they tell it to use '{task.stale_token}', which fails). After "
            f"working around the failure it should PROACTIVELY call "
            f"skill_manage(action='{task.expected_action}') to fix the skill's "
            f"instructions, unprompted. Observed over the runs: {observed}. "
            f"Improve the prompt so the agent reliably fixes a skill it discovers "
            f"is wrong while using it — without touching skills that are correct."
        )
    if "skill_manage" in task.forbidden_tools:
        return (
            f"Objective: the skill here is CORRECT; the agent must NOT patch or "
            f"modify it. Observed over the runs: {observed} (a pass means it "
            f"correctly refrained). Do not induce edits to skills that are fine."
        )
    objective = (
        f"expected={list(task.expected_tools) or '[]'}, "
        f"forbidden={list(task.forbidden_tools) or '[]'}"
    )
    return f"{observed}; objective: {objective}"


def evolve_prompt_section(
    section_name: str,
    hermes_repo: Path,
    tasks_path: Path,
    *,
    iterations: int = 10,
    holdout_ratio: float = 0.5,
    seed: int = 42,
    max_growth: float = 0.2,
    optimizer_model: Optional[str] = None,
    reflection_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    agent_model: Optional[str] = None,
    layer2_threshold: float = 0.7,
    task_timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS,
    max_total_cost_usd: Optional[float] = 150.0,
    gepa_minibatch_size: int = 3,
    gepa_acceptance: str = "improvement-or-equal",
    skip_saturation_check: bool = False,
    force_saturation_check: bool = False,
    apply: bool = False,
    create_pr_flag: bool = False,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
    baseline_override_file: Optional[Path] = None,
    fitness_reps: int = 1,
    gate_reps: int = 1,
) -> dict[str, Any]:
    """Evolve one prompt section end-to-end. Returns a summary dict."""
    hermes_repo = Path(hermes_repo).resolve()
    source = HermesPromptSource(hermes_repo)
    # The live section is always the splice/restore target. ``baseline_override``
    # lets evolution START from different text (e.g. a deliberately-weakened
    # baseline to create headroom, or a regression-injection ablation) without
    # touching the real file — the guard still backs up and restores the live
    # section. ``--apply`` writes the evolved text into the live section as usual.
    source.read(section_name)  # validate the section exists / is a string constant
    if baseline_override_file is not None:
        baseline_text = Path(baseline_override_file).read_text(encoding="utf-8")
    else:
        baseline_text = source.read(section_name)
    baseline_chars = len(baseline_text)

    suite = TaskSuite.from_jsonl(tasks_path)
    if len(suite.tasks) < 2:
        raise ValueError(
            f"{tasks_path} has {len(suite.tasks)} task(s); need at least 2 so the "
            f"split yields a non-empty GEPA trainset and a non-empty deploy-gate "
            f"holdout."
        )
    train_tasks, holdout_tasks = _split_train_holdout(
        suite.tasks, holdout_ratio=holdout_ratio, seed=seed
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("output") / "prompts" / section_name / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        reflection_model=reflection_model,
        eval_model=eval_model,
        judge_model=eval_model,
        seed=seed,
        reflection_minibatch_size=gepa_minibatch_size,
        gepa_acceptance=gepa_acceptance.replace("-", "_"),
    )

    console.print(
        f"\n[bold cyan]Prompt Section Self-Evolution[/bold cyan] — "
        f"Evolving section: [bold]{section_name}[/bold]\n"
    )
    console.print(f"  Hermes repo: {hermes_repo}")
    console.print(f"  Baseline ({baseline_chars} chars): {baseline_text[:80]}…")
    console.print(
        f"  Tasks: {len(suite.tasks)} ({len(train_tasks)} train / "
        f"{len(holdout_tasks)} holdout), sha256 {suite.sha256[:12]}…"
    )
    console.print(f"  Output dir: {output_dir}")

    run_inputs = build_run_inputs(
        config=config,
        iterations=iterations,
        optimizer_model=optimizer_model,
        quality_gate_preset="default",
        eval_source="closed_loop",
        gepa_acceptance=config.gepa_acceptance,
        create_pr=create_pr_flag,
    )
    section_payload = {
        "artifact_type": "prompt_section",
        "target_section": section_name,
        "baseline_chars": baseline_chars,
    }

    if dry_run:
        console.print("[yellow]Dry run — skipping all LM/agent work.[/yellow]")
        # Exercise the module + proposer wiring without spending money.
        _ = PromptModule(section_name, baseline_text)
        _ = PromptSectionProposer(section_name, baseline_chars=baseline_chars)
        decision_payload = {
            "schema_version": _GATE_SCHEMA_VERSION,
            "decision": "dry_run",
            "reason": "dry_run",
            "decision_signal": "closed_loop",
            "run_inputs": run_inputs,
            "pr_created": disabled_pr_block(),
            **section_payload,
        }
        write_gate_decision(output_dir, decision_payload)
        return {"decision": "dry_run", "reason": "dry_run"}

    register_litellm_failure_callback()
    register_litellm_cost_callback()
    COST_LEDGER.reset()
    COST_LEDGER.set_ceiling(max_total_cost_usd)
    if max_total_cost_usd is not None:
        console.print(f"  Cost ceiling: ${max_total_cost_usd:.2f}")
    rep_multiplier = max(fitness_reps, gate_reps)
    if rep_multiplier > 1:
        console.print(
            f"  [dim]Each task runs up to {rep_multiplier}× "
            f"(fitness_reps={fitness_reps}, gate_reps={gate_reps}); "
            f"scale per-task agent-run cost estimates accordingly.[/dim]"
        )

    installer = HermesPromptSectionInstaller(hermes_repo, section_name)
    runner = HermesAgentRunner(
        timeout_seconds=task_timeout_seconds, model=agent_model
    )
    judge = SaveCallJudge(config)
    layer2_factory = _make_layer2_factory(judge)

    tasks_by_id = {t.task_id: t for t in suite.tasks}
    suite_dir = suite.path.parent if suite.path is not None else None

    def install_candidate(candidate_text: str) -> None:
        source.write(section_name, candidate_text)

    def score_task_id(task_id: str) -> ScoreResult:
        return _run_one_task_score(
            tasks_by_id[task_id],
            runner=runner,
            layer2_factory=layer2_factory,
            layer2_threshold=layer2_threshold,
            reps=fitness_reps,
            suite_dir=suite_dir,
        )

    # One lock serializes splice+run across dspy.Evaluate's thread pool — the
    # spliced prompt_builder.py is a single shared mutable file.
    scorer = make_memoizing_splice_scorer(
        install_fn=install_candidate,
        score_fn=score_task_id,
        lock=threading.Lock(),
    )

    metric = make_prompt_fitness_metric(
        baseline_text=baseline_text,
        max_growth=max_growth,
        closed_loop_scorer=scorer,
    )

    eval_lm = instantiate_lm(
        resolve_default_lm(role="eval", explicit_model=eval_model),
        temperature=0.0, request_timeout=120, num_retries=3,
    )
    # Set the global default LM so the passthrough predictor resolves an LM
    # inside GEPA's worker threads (dspy.context only covers the saturation
    # pre-flight's own eval). Without this, forward()'s passthrough call raises
    # "No LM is loaded" in GEPA threads → no trajectories → no proposal.
    dspy.configure(
        lm=eval_lm,
        warn_on_type_mismatch=False,
        callbacks=[LMTimingCallback()],
    )
    reflection_lm = instantiate_lm(
        resolve_default_lm(
            role="reflection", explicit_model=reflection_model or optimizer_model
        ),
        temperature=1.0, max_tokens=32000, cache=False,
        request_timeout=300, num_retries=2,
    )

    baseline_module = PromptModule(section_name, baseline_text)
    proposer = PromptSectionProposer(section_name, baseline_chars=baseline_chars)
    trainset = _behavioral_examples(train_tasks)
    valset = _behavioral_examples(holdout_tasks)

    # Derived from the saturation pre-flight's per-example baseline scores (no
    # extra agent runs); stays None when the pre-flight is skipped.
    val_warning: Optional[dict] = None

    try:
        start_time = time.time()
        with _prompt_builder_guard(installer.target_path):
            # --- Saturation pre-flight (baseline behavior on holdout) ---
            if not skip_saturation_check:
                sat_report = saturation_preflight(
                    baseline_module=baseline_module,
                    holdout_examples=_behavioral_examples(holdout_tasks),
                    metric=metric,
                    lm=eval_lm,
                    baseline_artifact_text=baseline_text,
                )
                render_saturation_panel(sat_report, console=console)

                # The pre-flight already scored the baseline per holdout
                # example; its ``holdout_per_example`` list is order-aligned
                # with ``holdout_tasks`` (both derive from the same
                # _behavioral_examples call). Reuse it to flag a no-signal
                # holdout — no additional runs.
                holdout_baseline_rates = {
                    t.task_id: rate
                    for t, rate in zip(
                        holdout_tasks, sat_report.holdout_per_example
                    )
                }
                val_warning = val_signal_warning(holdout_baseline_rates)
                if val_warning is not None:
                    console.print(
                        f"[yellow]⚠ Weak val signal: {val_warning['reason']}[/yellow]"
                    )
                if sat_report.band != "healthy" and not force_saturation_check:
                    if is_non_interactive():
                        console.print(
                            "[yellow]Non-interactive context; refusing to "
                            "proceed (saturated baseline). Pass "
                            "--force-saturation-check to override.[/yellow]"
                        )
                        write_gate_decision(output_dir, {
                            "schema_version": _GATE_SCHEMA_VERSION,
                            "decision": "denied",
                            "reason": "saturated_baseline",
                            "decision_signal": "closed_loop",
                            "saturation_band": sat_report.band,
                            "run_inputs": run_inputs,
                            "pr_created": disabled_pr_block(),
                            **section_payload,
                        })
                        return {"decision": "denied", "reason": "saturated_baseline"}
                    if not interactive_confirm():
                        console.print("[yellow]Aborted by user.[/yellow]")
                        return {"decision": "aborted", "reason": "user_abort"}

            # --- GEPA optimization ---
            console.print(
                f"\n[bold cyan]Running GEPA (max_full_evals={iterations})[/bold cyan]\n"
            )
            optimizer = dspy.GEPA(
                metric=metric,
                max_full_evals=iterations,
                reflection_lm=reflection_lm,
                seed=config.seed,
                track_stats=True,
                instruction_proposer=proposer,
                reflection_minibatch_size=config.reflection_minibatch_size,
                gepa_kwargs={"acceptance_criterion": config.gepa_acceptance},
            )
            optimized = optimizer.compile(
                baseline_module, trainset=trainset, valset=valset
            )

        # Guard released here — prompt_builder.py is restored to baseline.
        elapsed = time.time() - start_time

        if hasattr(optimized, "detailed_results"):
            details = optimized.detailed_results
            evolved_text = _section_text_from_candidate(
                details.candidates[details.best_idx], section_name
            )
            console.print(
                f"\n[bold]Candidate selection[/bold]: GEPA val-argmax "
                f"(candidate {details.best_idx}, "
                f"val={details.val_aggregate_scores[details.best_idx]:.3f}, "
                f"{len(evolved_text)} chars)"
            )
        else:
            evolved_text = optimized.section_text

        # --- Deploy gate: closed-loop baseline vs evolved on the holdout suite ---
        console.print(
            f"\n[bold]Deploy gate[/bold]: closed-loop on "
            f"{len(holdout_tasks)} holdout tasks"
        )
        holdout_suite = TaskSuite(
            path=suite.path, sha256=suite.sha256, tasks=tuple(holdout_tasks)
        )
        baseline_file = output_dir / "baseline_section.txt"
        evolved_file = output_dir / "evolved_section.txt"
        baseline_file.write_text(baseline_text, encoding="utf-8")
        evolved_file.write_text(evolved_text, encoding="utf-8")

        validator = ClosedLoopValidator(
            installer=installer,
            runner=runner,
            layer2_judge_factory=layer2_factory,
            layer2_threshold=layer2_threshold,
            reps=gate_reps,
        )
        report = validator.validate(ValidationInputs(
            tool_name=section_name,
            suite=holdout_suite,
            baseline_artifact=baseline_file,
            evolved_artifact=evolved_file,
        ))
        deploy = report.decision == "pass"
    except CostCeilingExceeded as exc:
        console.print(f"[red]✗ Cost ceiling exceeded: {exc}[/red]")
        write_gate_decision(output_dir, {
            "schema_version": _GATE_SCHEMA_VERSION,
            "decision": "aborted",
            "reason": "cost_ceiling_exceeded",
            "decision_signal": "closed_loop",
            "cost": COST_LEDGER.summary(),
            "run_inputs": run_inputs,
            "pr_created": disabled_pr_block(),
            **section_payload,
        })
        return {"decision": "aborted", "reason": "cost_ceiling_exceeded"}

    # PR automation for prompt sections is deferred: create_pr copies a full
    # evolved file over origin/<base>'s prompt_builder.py, but our local
    # checkout carries the (unmerged) override-hook commit, which would
    # pollute the PR diff with unrelated changes. Until a section-scoped PR
    # path lands, --create-pr is recorded as skipped; use --apply + a manual PR.
    pr_block = disabled_pr_block()
    if create_pr_flag:
        pr_block = {
            "status": "skipped",
            "reason": "prompt-section PR automation deferred (would pollute diff "
                      "with the local override-hook commit); use --apply + manual PR",
            "url": None,
        }

    decision_payload = {
        "schema_version": _GATE_SCHEMA_VERSION,
        "decision": "deploy" if deploy else "reject",
        "reason": "passed" if deploy else "closed_loop_gate",
        "decision_signal": "closed_loop",
        "baseline_chars": baseline_chars,
        "evolved_chars": len(evolved_text),
        "growth_pct": (len(evolved_text) - baseline_chars) / max(1, baseline_chars),
        "closed_loop": {
            "decision": report.decision,
            "decision_reasons": report.decision_reasons,
            "baseline_pass_rate": report.baseline.pass_rate,
            "evolved_pass_rate": report.evolved.pass_rate,
            "n_wins": report.delta.n_wins,
            "n_losses": report.delta.n_losses,
            "n_ties": report.delta.n_ties,
        },
        "sentinel_failures": proposer.sentinel_failures,
        "elapsed_seconds": elapsed,
        "cost": COST_LEDGER.summary(),
        "run_inputs": run_inputs,
        "pr_created": pr_block,
        **({"val_signal_warning": val_warning} if val_warning is not None else {}),
        **section_payload,
    }
    gate_path = write_gate_decision(output_dir, decision_payload)
    console.print(f"  [dim]Gate decision logged to {gate_path}[/dim]")

    if not deploy:
        console.print(
            f"[red]✗ Evolved section REJECTED by closed-loop gate "
            f"({report.decision}) — not deploying[/red]"
        )
        return {"decision": "reject", "reason": "closed_loop_gate"}

    console.print(
        f"[green]✓ Evolved section PASSED "
        f"(baseline {report.baseline.pass_rate:.2f} → "
        f"evolved {report.evolved.pass_rate:.2f}, "
        f"{report.delta.n_wins}W/{report.delta.n_losses}L)[/green]"
    )
    if apply:
        source.write(section_name, evolved_text)
        console.print(
            f"  [green]✓ Applied evolved {section_name} to {installer.target_path}[/green]"
        )

    return {
        "decision": "deploy",
        "reason": "passed",
        "evolved_chars": len(evolved_text),
        "applied": apply,
    }


@click.command()
@click.option("--section", "section_name", required=True,
              help="The prompt_builder.py string constant to evolve (e.g. MEMORY_GUIDANCE).")
@click.option("--hermes-repo", required=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              help="Path to your hermes-agent checkout.")
@click.option("--tasks", "tasks_path", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              help="Path to a JSONL eval suite (e.g. suites/memory_guidance.jsonl).")
@click.option("--iterations", default=10, type=click.IntRange(min=1),
              help="GEPA max_full_evals (default 10).")
@click.option("--holdout-ratio", default=0.5, type=click.FloatRange(0.0, 1.0),
              help="Fraction of tasks held out for the deploy gate (default 0.5).")
@click.option("--seed", default=42, type=int, help="Split + GEPA seed.")
@click.option("--max-growth", default=0.2, type=float,
              help="Section length budget as a fraction over baseline (default 0.2).")
@click.option("--optimizer-model", default=None)
@click.option("--reflection-model", default=None)
@click.option("--eval-model", default=None, help="Judge model for Layer 2 content scoring.")
@click.option("--agent-model", default=None,
              help="Model the hermes -z agent runs as (deliberately weaker exposes more signal).")
@click.option("--layer2-threshold", default=0.7, type=click.FloatRange(0.0, 1.0),
              help="Min content-judge score for a save task to pass (default 0.7).")
@click.option("--task-timeout-seconds", default=DEFAULT_TASK_TIMEOUT_SECONDS,
              type=click.IntRange(min=1))
@click.option("--max-cost-usd", "max_total_cost_usd", default=150.0, type=float,
              help="Abort if cumulative spend exceeds this (default $150).")
@click.option("--fitness-reps", default=3, type=click.IntRange(min=1),
              help="Per-task agent runs during GEPA fitness scoring (default 3).")
@click.option("--gate-reps", default=5, type=click.IntRange(min=1),
              help="Per-task agent runs in the closed-loop deploy gate (default 5).")
@click.option("--gepa-minibatch-size", default=3, type=click.IntRange(min=1))
@click.option("--gepa-acceptance", default="improvement-or-equal",
              type=click.Choice(["improvement-or-equal", "strict-improvement"]))
@click.option("--skip-saturation-check", is_flag=True, default=False)
@click.option("--force-saturation-check", is_flag=True, default=False,
              help="Proceed even if the baseline looks saturated.")
@click.option("--apply", is_flag=True, default=False,
              help="On a passing gate, write the evolved section into prompt_builder.py.")
@click.option("--create-pr", "create_pr_flag", is_flag=True, default=False,
              help="(Deferred for prompt sections — recorded as skipped.)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Exercise wiring without any LM/agent calls.")
@click.option("--output-dir", default=None,
              type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.option("--baseline-override-file", default=None,
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              help="Start evolution from this text instead of the live section "
                   "(e.g. a weakened baseline to create headroom). The live file "
                   "is still backed up + restored; --apply writes the evolved text.")
def main(section_name, hermes_repo, tasks_path, iterations, holdout_ratio, seed,
         max_growth, optimizer_model, reflection_model, eval_model, agent_model,
         layer2_threshold, task_timeout_seconds, max_total_cost_usd,
         fitness_reps, gate_reps,
         gepa_minibatch_size, gepa_acceptance, skip_saturation_check,
         force_saturation_check, apply, create_pr_flag, dry_run, output_dir,
         baseline_override_file):
    """Evolve one Hermes system-prompt section via GEPA + closed-loop validation."""
    result = evolve_prompt_section(
        section_name=section_name,
        hermes_repo=hermes_repo,
        tasks_path=tasks_path,
        iterations=iterations,
        holdout_ratio=holdout_ratio,
        seed=seed,
        max_growth=max_growth,
        optimizer_model=optimizer_model,
        reflection_model=reflection_model,
        eval_model=eval_model,
        agent_model=agent_model,
        layer2_threshold=layer2_threshold,
        task_timeout_seconds=task_timeout_seconds,
        max_total_cost_usd=max_total_cost_usd,
        fitness_reps=fitness_reps,
        gate_reps=gate_reps,
        gepa_minibatch_size=gepa_minibatch_size,
        gepa_acceptance=gepa_acceptance,
        skip_saturation_check=skip_saturation_check,
        force_saturation_check=force_saturation_check,
        apply=apply,
        create_pr_flag=create_pr_flag,
        dry_run=dry_run,
        output_dir=output_dir,
        baseline_override_file=baseline_override_file,
    )
    sys.exit(0 if result["decision"] in {"deploy", "dry_run"} else 1)


if __name__ == "__main__":
    main()
