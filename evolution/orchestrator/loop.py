"""One-command product loop orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolution.artifacts.store import ArtifactStore
from evolution.datasets.redaction import scan_examples_for_secrets
from evolution.db.store import EvolutionStore
from evolution.orchestrator.executor import execute_skill_run
from evolution.orchestrator.exporter import export_review_bundle
from evolution.orchestrator.gates import evaluate_run_gate
from evolution.orchestrator.run_manager import create_skill_run, parse_target_ref
from evolution.traces.jsonl import failed_traces_to_eval_examples, load_trace_jsonl, scan_traces_for_secrets


def run_loop_once(
    *,
    store: EvolutionStore,
    root: str | Path,
    target_ref: str,
    trace_path: str | Path,
    trace_source: str = "hermes-session",
    dataset_version: str = "loop-v1",
    engine: str = "gepa",
    iterations: int = 5,
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
    min_holdout_improvement: float = 0.0,
    preferred_metric: str = "rubric_score",
    export_out: str | Path | None = None,
) -> dict[str, Any]:
    """Run import→dataset→execute→gate→export once for one target."""
    root = Path(root)
    target_type, target_name = parse_target_ref(target_ref)
    target = store.get_target_by_name(target_type, target_name)
    if not target:
        raise ValueError(f"Target not found: {target_ref}. Run targets scan first.")

    imported = _import_traces(store, target, trace_path, trace_source)
    dataset = _build_trace_dataset(
        store=store,
        root=root,
        target=target,
        target_ref=target_ref,
        target_type=target_type,
        target_name=target_name,
        version=dataset_version,
    )
    run = create_skill_run(
        store=store,
        target_ref=target_ref,
        dataset_id=dataset["id"],
        engine=engine,
        iterations=iterations,
        extra_config={"loop": "once", "trace_path": str(trace_path)},
    )
    execution = execute_skill_run(
        store=store,
        root=root,
        run_id=run["id"],
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
    )
    gate = evaluate_run_gate(
        store=store,
        root=root,
        run_id=run["id"],
        min_holdout_improvement=min_holdout_improvement,
        preferred_metric=preferred_metric,
    )
    export = export_review_bundle(
        store=store,
        root=root,
        run_id=run["id"],
        out_dir=export_out,
        allow_hold=True,
    )
    store.add_run_event(
        run["id"],
        "loop",
        "loop once completed",
        {"dataset_id": dataset["id"], "gate_decision": gate["decision"], "bundle_dir": export["bundle_dir"]},
    )
    return {
        "target": target_ref,
        "imported_traces": len(imported),
        "dataset": dataset,
        "run": execution["run"],
        "gate": gate,
        "export": export,
    }


def _import_traces(store: EvolutionStore, target: dict[str, Any], trace_path: str | Path, trace_source: str) -> list[dict[str, Any]]:
    traces = load_trace_jsonl(Path(trace_path), default_source=trace_source)
    scan_report = scan_traces_for_secrets(traces)
    if scan_report["status"] != "passed":
        raise ValueError(f"secret scan failed: {scan_report['matches']}")
    imported = []
    for trace in traces:
        imported.append(
            store.add_attempt_trace(
                target_id=target["id"],
                source=trace["source"],
                task_input=trace["task_input"],
                observed_output=trace.get("observed_output"),
                expected_behavior=trace.get("expected_behavior"),
                status=trace["status"],
                failure_reason=trace.get("failure_reason"),
                source_ref_hash=trace.get("source_ref_hash"),
                metadata=trace.get("metadata"),
            )
        )
    return imported


def _build_trace_dataset(
    *,
    store: EvolutionStore,
    root: Path,
    target: dict[str, Any],
    target_ref: str,
    target_type: str,
    target_name: str,
    version: str,
) -> dict[str, Any]:
    trace_rows = store.list_attempt_traces(target_id=target["id"], status="failure")
    if not trace_rows:
        raise ValueError(f"No failed traces for {target_ref}")
    normalized = [_trace_row_for_examples(row) for row in trace_rows]
    examples = _assign_trace_splits(failed_traces_to_eval_examples(normalized))
    scan_report = scan_examples_for_secrets(examples)
    if scan_report["status"] != "passed":
        raise ValueError(f"secret scan failed: {scan_report['matches']}")

    split_spec = _split_spec(examples)
    manifest = {
        "schema_version": 1,
        "target": target_ref,
        "source": "loop-traces",
        "version": version,
        "trace_count": len(trace_rows),
        "example_count": len(examples),
        "split_spec": split_spec,
        "secret_scan": scan_report,
        "pii_scan": {"status": "not_implemented"},
    }
    manifest_ref = ArtifactStore(root).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        suffix=".json",
        kind="dataset",
        mime_type="application/json",
        metadata={"target": target_ref, "source": "loop-traces", "version": version},
    )
    artifact = store.add_artifact(
        kind="dataset",
        content_sha256=manifest_ref.content_sha256,
        storage_uri=manifest_ref.storage_uri,
        size_bytes=manifest_ref.size_bytes,
        target_id=target["id"],
        mime_type="application/json",
        metadata=manifest_ref.metadata,
    )
    dataset = store.add_dataset(
        target_id=target["id"],
        source="loop-traces",
        version=version,
        artifact_id=artifact["id"],
        split_spec=split_spec,
        pii_scan_status="not_implemented",
        secret_scan_status=scan_report["status"],
        example_count=len(examples),
    )
    for example in examples:
        store.add_eval_example(
            dataset_id=dataset["id"],
            split=example["split"],
            source=example.get("source"),
            task_input=example["task_input"],
            expected_behavior=example["expected_behavior"],
            source_ref_hash=example.get("source_ref_hash"),
            metadata=example.get("metadata"),
        )
    dataset_dir = root / "datasets" / target_type / target_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / f"{dataset['id']}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return dataset


def _trace_row_for_examples(row: dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    data["metadata"] = row.get("metadata_json") or {}
    return data


def _assign_trace_splits(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_cycle = ["train", "val", "holdout"]
    return [{**example, "split": split_cycle[index % len(split_cycle)]} for index, example in enumerate(examples)]


def _split_spec(examples: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "holdout": 0}
    for example in examples:
        counts[example["split"]] = counts.get(example["split"], 0) + 1
    return {split: count for split, count in counts.items() if count}
