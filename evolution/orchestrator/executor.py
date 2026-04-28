"""Execution engine for registered evolution runs.

This module owns the DB-backed run lifecycle: load target/dataset, generate a
candidate via the selected strategy, persist artifacts/evaluations, and write a
manifest. Strategies intentionally share one persistence contract:
- deterministic: offline train/val synthesis for safe tests;
- model-synthesis: single OpenAI-compatible model call;
- dspy-gepa: real DSPy GEPA optimizer adapter with holdout isolation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evolution.artifacts.store import ArtifactStore
from evolution.db.store import EvolutionStore
from evolution.evaluation.rubric import score_candidate_with_rubric
from evolution.models.compare import ModelConfigError, compare_chat_models
from evolution.optimizers.dspy_gepa import DSpyGepaConfig, run_dspy_gepa_skill_optimizer
from evolution.skills.skill_module import load_skill, reassemble_skill


_ALLOWED_STRATEGIES = {"deterministic", "model-synthesis", "dspy-gepa"}


def execute_skill_run(
    store: EvolutionStore,
    root: str | Path,
    run_id: str,
    strategy: str = "deterministic",
    provider: str = "deepseek",
    optimizer_model: str | None = None,
    eval_model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: float = 60.0,
    extra_body: dict[str, Any] | None = None,
    scoring_strategy: str = "deterministic-rubric",
    judge_model: str | None = None,
    dspy_model_prefix: str | None = None,
    gepa_max_full_evals: int | None = None,
    gepa_reflection_minibatch_size: int = 3,
    gepa_log_dir: str | None = None,
    dspy_module: Any | None = None,
    skill_module_factory: Any | None = None,
    client_factory: Any | None = None,
) -> dict[str, Any]:
    """Execute a registered skill-evolution run and persist evidence.

    The M4 executor is deliberately deterministic/offline: it proves the control
    plane can move a run through lifecycle states, write artifacts, persist
    baseline/evolved candidates, and store per-example evaluations without
    needing API keys or a live optimizer.
    """
    if strategy not in _ALLOWED_STRATEGIES:
        raise ValueError(f"Unsupported execution strategy: {strategy}")

    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")
    if run["status"] not in {"pending"}:
        raise ValueError(f"Run {run_id} is not pending; status={run['status']}")

    store.update_run_status(run_id, "running")
    store.add_run_event(run_id, "status", "run started", {"strategy": strategy})

    try:
        result = _execute_skill_run_with_strategy(
            store=store,
            root=Path(root),
            run_id=run_id,
            strategy=strategy,
            provider=provider,
            optimizer_model=optimizer_model,
            eval_model=eval_model,
            base_url=base_url,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
            scoring_strategy=scoring_strategy,
            judge_model=judge_model,
            dspy_model_prefix=dspy_model_prefix,
            gepa_max_full_evals=gepa_max_full_evals,
            gepa_reflection_minibatch_size=gepa_reflection_minibatch_size,
            gepa_log_dir=gepa_log_dir,
            dspy_module=dspy_module,
            skill_module_factory=skill_module_factory,
            client_factory=client_factory,
        )
        completed = store.update_run_status(run_id, "completed", completed=True)
        store.add_run_event(run_id, "status", "run completed", {"strategy": strategy})
        result["run"] = completed
        return result
    except Exception as exc:
        store.update_run_status(run_id, "failed", error=_safe_error(exc), completed=True)
        store.add_run_event(run_id, "status", "run failed", {"error_type": type(exc).__name__})
        raise


def _execute_skill_run_with_strategy(
    store: EvolutionStore,
    root: Path,
    run_id: str,
    strategy: str,
    provider: str,
    optimizer_model: str | None,
    eval_model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body: dict[str, Any] | None,
    scoring_strategy: str,
    judge_model: str | None,
    dspy_model_prefix: str | None,
    gepa_max_full_evals: int | None,
    gepa_reflection_minibatch_size: int,
    gepa_log_dir: str | None,
    dspy_module: Any | None,
    skill_module_factory: Any | None,
    client_factory: Any | None,
) -> dict[str, Any]:
    run = _require(store.get_run(run_id), f"Run not found: {run_id}")
    target = _require(store.get_target(run["target_id"]), f"Target not found: {run['target_id']}")
    dataset_id = _require(run.get("dataset_id"), f"Run {run_id} has no dataset_id")
    dataset = _require(store.get_dataset(dataset_id), f"Dataset not found: {dataset_id}")
    repo = _require(store.get_repository_by_id(target["repository_id"]), f"Repository not found: {target['repository_id']}")

    skill_path = _target_path(repo["local_path"], target["file_path"])
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")

    skill = load_skill(skill_path)
    examples = store.list_eval_examples(dataset_id)
    train_examples = [example for example in examples if example["split"] == "train"]
    val_examples = [example for example in examples if example["split"] == "val"]
    training_examples = [*train_examples, *val_examples]

    baseline_full = skill["raw"]
    evolved_body, optimizer_metadata = _build_candidate_body_for_strategy(
        strategy=strategy,
        baseline_body=skill["body"],
        training_examples=training_examples,
        train_examples=train_examples,
        val_examples=val_examples,
        run_iterations=_run_iterations(run),
        provider=provider,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        base_url=base_url,
        api_key_env=api_key_env,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        extra_body=extra_body,
        dspy_model_prefix=dspy_model_prefix,
        gepa_max_full_evals=gepa_max_full_evals,
        gepa_reflection_minibatch_size=gepa_reflection_minibatch_size,
        gepa_log_dir=gepa_log_dir,
        dspy_module=dspy_module,
        skill_module_factory=skill_module_factory,
        client_factory=client_factory,
    )
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    artifact_store = ArtifactStore(root)
    baseline_candidate = _persist_candidate(
        store=store,
        artifact_store=artifact_store,
        target_id=target["id"],
        run_id=run_id,
        role="baseline",
        text=baseline_full,
        metadata={"source": "target_file", "path": str(skill_path), "strategy": strategy},
    )
    evolved_candidate = _persist_candidate(
        store=store,
        artifact_store=artifact_store,
        target_id=target["id"],
        run_id=run_id,
        role="evolved",
        text=evolved_full,
        parent_candidate_id=baseline_candidate["id"],
        metadata=optimizer_metadata,
    )

    evaluations = []
    resolved_judge_model = judge_model or eval_model or optimizer_model
    for candidate, text in [(baseline_candidate, baseline_full), (evolved_candidate, evolved_full)]:
        for example in examples:
            rubric = score_candidate_with_rubric(
                candidate_text=text,
                example=example,
                candidate_role=candidate["role"],
                strategy=scoring_strategy,
                provider=provider,
                judge_model=resolved_judge_model,
                base_url=base_url,
                api_key_env=api_key_env,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                extra_body=extra_body,
                client_factory=client_factory,
            )
            evaluations.append(
                store.add_evaluation(
                    run_id=run_id,
                    candidate_id=candidate["id"],
                    dataset_id=dataset_id,
                    split=example["split"],
                    example_id=example["id"],
                    metric_name=rubric.metric_name,
                    score=rubric.score,
                    details={
                        **rubric.details,
                        "strategy": strategy,
                    },
                )
            )

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "target": f"{target['target_type']}:{target['name']}",
        "target_path": str(skill_path),
        "engine": run["engine"],
        "execution_strategy": strategy,
        "optimizer": optimizer_metadata,
        "scoring": {
            "strategy": scoring_strategy,
            "judge_model": resolved_judge_model,
        },
        "dataset_id": dataset_id,
        "dataset": {
            "source": dataset["source"],
            "version": dataset["version"],
            "example_count": dataset["example_count"],
            "split_spec": dataset["split_spec_json"],
        },
        "candidates": [baseline_candidate, evolved_candidate],
        "evaluation_summary": _summarize_evaluations(evaluations, [baseline_candidate, evolved_candidate]),
        "holdout_examples_used_for_generation": 0,
    }
    manifest_ref = artifact_store.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        suffix=".json",
        kind="run_manifest",
        mime_type="application/json",
        metadata={"run_id": run_id, "target_id": target["id"], "strategy": strategy},
    )
    manifest_artifact = store.add_artifact(
        kind="run_manifest",
        content_sha256=manifest_ref.content_sha256,
        storage_uri=manifest_ref.storage_uri,
        size_bytes=manifest_ref.size_bytes,
        target_id=target["id"],
        mime_type="application/json",
        metadata=manifest_ref.metadata,
    )
    store.add_run_event(run_id, "artifact", "run manifest written", {"artifact_id": manifest_artifact["id"]})

    return {
        "run": store.get_run(run_id),
        "candidates": [baseline_candidate, evolved_candidate],
        "evaluations": evaluations,
        "manifest_artifact_id": manifest_artifact["id"],
    }


def _persist_candidate(
    store: EvolutionStore,
    artifact_store: ArtifactStore,
    target_id: str,
    run_id: str,
    role: str,
    text: str,
    metadata: dict[str, Any],
    parent_candidate_id: str | None = None,
) -> dict[str, Any]:
    artifact_ref = artifact_store.write_text(
        text,
        suffix=".md",
        kind="skill_candidate",
        mime_type="text/markdown",
        metadata={"run_id": run_id, "role": role, **metadata},
    )
    artifact = store.add_artifact(
        kind="skill_candidate",
        content_sha256=artifact_ref.content_sha256,
        storage_uri=artifact_ref.storage_uri,
        size_bytes=artifact_ref.size_bytes,
        target_id=target_id,
        mime_type="text/markdown",
        metadata=artifact_ref.metadata,
    )
    candidate = store.add_candidate(
        run_id=run_id,
        target_id=target_id,
        role=role,
        artifact_id=artifact["id"],
        content_sha256=artifact["content_sha256"],
        parent_candidate_id=parent_candidate_id,
        metadata=artifact_ref.metadata,
    )
    store.add_run_event(run_id, "candidate", f"{role} candidate persisted", {"candidate_id": candidate["id"]})
    return candidate


def _build_candidate_body_for_strategy(
    *,
    strategy: str,
    baseline_body: str,
    training_examples: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    run_iterations: int,
    provider: str,
    optimizer_model: str | None,
    eval_model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body: dict[str, Any] | None,
    dspy_model_prefix: str | None,
    gepa_max_full_evals: int | None,
    gepa_reflection_minibatch_size: int,
    gepa_log_dir: str | None,
    dspy_module: Any | None,
    skill_module_factory: Any | None,
    client_factory: Any | None,
) -> tuple[str, dict[str, Any]]:
    if strategy == "deterministic":
        return _build_deterministic_candidate_body(baseline_body, training_examples), {
            "source": "deterministic_train_val_synthesis",
            "strategy": strategy,
            "train_val_examples": len(training_examples),
            "holdout_examples_used_for_generation": 0,
        }
    if strategy == "model-synthesis":
        if not optimizer_model:
            raise ValueError("optimizer_model is required for model-synthesis strategy")
        return _build_model_synthesis_candidate_body(
            baseline_body=baseline_body,
            training_examples=training_examples,
            provider=provider,
            optimizer_model=optimizer_model,
            base_url=base_url,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
            client_factory=client_factory,
        )
    if strategy == "dspy-gepa":
        if not optimizer_model:
            raise ValueError("optimizer_model is required for dspy-gepa strategy")
        resolved_eval_model = eval_model or optimizer_model
        try:
            result = run_dspy_gepa_skill_optimizer(
                baseline_body=baseline_body,
                train_examples=train_examples,
                val_examples=val_examples,
                config=DSpyGepaConfig(
                    provider=provider,
                    optimizer_model=optimizer_model,
                    eval_model=resolved_eval_model,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    extra_body=extra_body,
                    max_full_evals=gepa_max_full_evals or max(1, run_iterations),
                    reflection_minibatch_size=gepa_reflection_minibatch_size,
                    dspy_model_prefix=dspy_model_prefix,
                    log_dir=gepa_log_dir,
                ),
                dspy_module=dspy_module,
                module_factory=skill_module_factory,
            )
        except ModelConfigError as exc:
            raise ValueError(str(exc)) from exc
        return result.evolved_body, result.metadata
    raise ValueError(f"Unsupported execution strategy: {strategy}")


def _build_deterministic_candidate_body(baseline_body: str, training_examples: list[dict[str, Any]]) -> str:
    hints = []
    seen = set()
    for example in training_examples:
        expected = " ".join(str(example["expected_behavior"]).split())
        if expected and expected not in seen:
            seen.add(expected)
            hints.append(expected)

    if not hints:
        return baseline_body

    hint_lines = "\n".join(f"- {hint}" for hint in hints)
    return (
        baseline_body.rstrip()
        + "\n\n## Evolution Notes\n"
        + "Deterministic candidate generated from train/val examples only.\n\n"
        + "Expected behavior calibration:\n"
        + hint_lines
    )


def _build_model_synthesis_candidate_body(
    *,
    baseline_body: str,
    training_examples: list[dict[str, Any]],
    provider: str,
    optimizer_model: str,
    base_url: str | None,
    api_key_env: str | None,
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body: dict[str, Any] | None,
    client_factory: Any | None,
) -> tuple[str, dict[str, Any]]:
    prompt = _render_model_synthesis_prompt(baseline_body, training_examples)
    try:
        results = compare_chat_models(
            models=[optimizer_model],
            prompt=prompt,
            provider=provider,
            base_url=base_url,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
            client_factory=client_factory,
        )
    except ModelConfigError as exc:
        raise ValueError(str(exc)) from exc
    result = results[0]
    if not result["ok"]:
        raise RuntimeError(f"model-synthesis failed for {optimizer_model}: {result['error']}")
    evolved_body = _extract_model_synthesis_body(result.get("output_text") or "")
    if not evolved_body:
        raise RuntimeError(f"model-synthesis returned empty candidate for {optimizer_model}")
    metadata = {
        "source": "model_synthesis",
        "strategy": "model-synthesis",
        "provider": provider,
        "optimizer_model": optimizer_model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
        "extra_body": extra_body or {},
        "train_val_examples": len(training_examples),
        "holdout_examples_used_for_generation": 0,
        "model_usage": {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "latency_ms": result.get("latency_ms", 0),
        },
    }
    return evolved_body, metadata


def _render_model_synthesis_prompt(baseline_body: str, training_examples: list[dict[str, Any]]) -> str:
    examples = []
    for index, example in enumerate(training_examples, start=1):
        examples.append(
            f"Example {index} ({example['split']}):\n"
            f"Task input: {example['task_input']}\n"
            f"Expected behavior: {example['expected_behavior']}"
        )
    example_text = "\n\n".join(examples) if examples else "No train/val examples available. Preserve baseline behavior."
    return (
        "You are improving a Hermes Agent SKILL.md body.\n"
        "Return only the improved Markdown body, without YAML frontmatter and without code fences.\n"
        "Use ONLY the train/validation examples below. Do not infer or mention holdout examples.\n"
        "Keep the skill concise, actionable, and compatible with existing Hermes tooling.\n\n"
        "Current skill body:\n"
        f"{baseline_body}\n\n"
        "Train/validation failure pressure:\n"
        f"{example_text}\n"
    )


def _extract_model_synthesis_body(output_text: str) -> str:
    text = output_text.strip()
    fence = re.match(r"^```(?:markdown|md)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    return text.strip()


def _keyword_overlap_score(candidate_text: str, expected_behavior: str) -> tuple[float, dict[str, Any]]:
    expected_terms = _tokens(expected_behavior)
    if not expected_terms:
        return 1.0, {"expected_terms": [], "matched_terms": [], "overlap": 1.0}
    candidate_terms = set(_tokens(candidate_text))
    matched_terms = sorted(set(expected_terms) & candidate_terms)
    score = len(matched_terms) / len(set(expected_terms))
    return score, {
        "expected_terms": sorted(set(expected_terms)),
        "matched_terms": matched_terms,
        "overlap": score,
    }


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text)]


def _summarize_evaluations(evaluations: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    role_by_candidate = {candidate["id"]: candidate["role"] for candidate in candidates}
    buckets: dict[str, dict[str, list[float]]] = {}
    for evaluation in evaluations:
        role = role_by_candidate[evaluation["candidate_id"]]
        buckets.setdefault(role, {}).setdefault(evaluation["split"], []).append(float(evaluation["score"]))

    summary: dict[str, Any] = {}
    for role, splits in buckets.items():
        summary[role] = {
            split: sum(scores) / len(scores)
            for split, scores in splits.items()
            if scores
        }
    return summary


def _target_path(repo_path: str, target_file_path: str) -> Path:
    path = Path(target_file_path)
    if path.is_absolute():
        return path
    return Path(repo_path) / path


def _require(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value


def _safe_error(exc: Exception) -> str:
    # Keep error evidence concise; future secret-redaction can plug in here.
    return str(exc)


def _run_iterations(run: dict[str, Any]) -> int:
    config = run.get("config_json") or {}
    try:
        return max(1, int(config.get("iterations") or 1))
    except (TypeError, ValueError):
        return 1
