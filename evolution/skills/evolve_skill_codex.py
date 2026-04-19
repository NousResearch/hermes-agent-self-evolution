"""Codex-batched self-evolution entrypoint."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import click
from rich.console import Console

from evolution.core.budget import RunBudget
from evolution.core.cached_dataset import load_or_create_dataset
from evolution.core.codex_protocol import build_evaluation_prompt, build_mutation_prompt
from evolution.core.codex_runner import CodexRunner
from evolution.core.codex_schema import EvaluationResult, parse_evaluation_result, parse_mutation_result
from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_io import find_skill, load_skill

console = Console()


def _build_config(iterations: int, hermes_repo: str | None) -> EvolutionConfig:
    config = EvolutionConfig(iterations=iterations)
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)
        config.output_dir = Path(hermes_repo) / "output"
    return config


def _load_dataset(dataset_path: str | None, *, allow_live_generation: bool):
    if not dataset_path:
        raise click.ClickException("cached eval-source requires --dataset-path")
    return load_or_create_dataset(
        Path(dataset_path),
        creator=lambda: (_ for _ in ()).throw(RuntimeError("live dataset generation not implemented yet")),
        allow_live_generation=allow_live_generation,
    )


def _select_holdout_examples(dataset, max_examples: int) -> list[dict]:
    if max_examples < 1:
        raise click.ClickException("HERMES_EVOLUTION_MAX_EXAMPLES must be at least 1")
    return [example.to_dict() for example in dataset.holdout[:max_examples]]


def _ensure_candidate_is_accepted(evaluation_result: EvaluationResult) -> None:
    if evaluation_result.recommendation.winner != "candidate":
        raise click.ClickException(
            "Candidate rejected by evaluation: "
            f"winner={evaluation_result.recommendation.winner}; "
            f"reason={evaluation_result.recommendation.reason}"
        )
    if evaluation_result.improvement <= 0:
        raise click.ClickException(
            "Candidate rejected by evaluation: "
            f"improvement must be > 0, got {evaluation_result.improvement}"
        )


def _save_run_artifacts(config: EvolutionConfig, skill_name: str, baseline_skill: str, candidate_skill: str, metrics: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_skill.md").write_text(baseline_skill)
    (output_dir / "evolved_skill.md").write_text(candidate_skill)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return output_dir


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--eval-source", default="cached", type=click.Choice(["cached"]), help="Dataset source for codex-batched evolution")
@click.option("--dataset-path", default=None, help="Path to cached evaluation dataset")
@click.option("--iterations", default=1, type=int, help="Number of mutation iterations")
@click.option("--dry-run", is_flag=True, help="Validate configuration and guardrails without running Codex")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
def main(skill: str, eval_source: str, dataset_path: str | None, iterations: int, dry_run: bool, hermes_repo: str | None):
    """Run the new codex-batched self-evolution workflow."""
    if iterations != 1:
        raise click.ClickException("codex-batched path currently supports only iterations=1 until the multi-iteration loop exists")

    config = _build_config(iterations=iterations, hermes_repo=hermes_repo)

    dataset, _created = _load_dataset(
        dataset_path,
        allow_live_generation=False,
    )

    if dry_run:
        console.print("[bold green]DRY RUN[/bold green] — codex-batched guardrails validated.")
        console.print(f"skill={skill}")
        console.print(f"eval_source={eval_source}")
        console.print(f"iterations={iterations}")
        return

    hermes_path = config.resolve_hermes_agent_path()
    skill_path = find_skill(skill, hermes_path)
    if not skill_path:
        raise click.ClickException(f"Skill '{skill}' not found in {hermes_path / 'skills'}")
    loaded_skill = load_skill(skill_path)

    budget = RunBudget(
        max_codex_calls=config.max_codex_calls,
        max_run_seconds=config.max_run_seconds,
        phase_timeout_seconds=config.phase_timeout_seconds,
        max_examples=config.max_examples,
        max_iterations=config.iterations,
    )
    budget.start_run()
    runner = CodexRunner(workdir=hermes_path, codex_bin=config.codex_bin)

    budget.register_call("mutation")
    try:
        mutation_payload = runner.run_json_task(
            build_mutation_prompt(skill, loaded_skill["raw"], iterations),
            timeout_seconds=config.phase_timeout_seconds,
        )
        mutation_result = parse_mutation_result(mutation_payload)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    candidate_skill = mutation_result.candidate_skill_markdown

    validator = ConstraintValidator(config)
    results = validator.validate_all(candidate_skill, "skill", baseline_text=loaded_skill["raw"])
    if not all(result.passed for result in results):
        problems = "; ".join(result.message for result in results if not result.passed)
        raise click.ClickException(f"Candidate failed constraints: {problems}")

    holdout_examples = _select_holdout_examples(dataset, config.max_examples)
    budget.register_call("evaluation")
    try:
        evaluation_payload = runner.run_json_task(
            build_evaluation_prompt(skill, loaded_skill["raw"], candidate_skill, holdout_examples),
            timeout_seconds=config.phase_timeout_seconds,
        )
        evaluation_result = parse_evaluation_result(evaluation_payload)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    _ensure_candidate_is_accepted(evaluation_result)

    metrics = {
        "skill_name": skill,
        "iterations": iterations,
        "baseline_score": evaluation_result.baseline_score,
        "candidate_score": evaluation_result.candidate_score,
        "improvement": evaluation_result.improvement,
        "per_example": evaluation_result.per_example,
        "recommendation": evaluation_result.recommendation.to_dict(),
        "prompt_version": "v1",
        "calls_used": budget.calls_used,
        "holdout_examples": len(holdout_examples),
        "budget": {
            "max_codex_calls": budget.max_codex_calls,
            "phase_timeout_seconds": budget.phase_timeout_seconds,
            "max_run_seconds": budget.max_run_seconds,
            "max_examples": budget.max_examples,
            "max_iterations": budget.max_iterations,
        },
    }
    output_dir = _save_run_artifacts(config, skill, loaded_skill["raw"], candidate_skill, metrics)
    console.print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
