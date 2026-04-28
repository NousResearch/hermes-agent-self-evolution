"""Production CLI foundation for Hermes self-evolution."""

from __future__ import annotations

from pathlib import Path
import json

import click

from evolution.artifacts.store import ArtifactStore
from evolution.config_file import init_evolution_root
from evolution.content.validator import validate_content_package
from evolution.datasets.golden import flatten_splits, load_golden_splits
from evolution.datasets.redaction import scan_examples_for_secrets
from evolution.db.store import EvolutionStore
from evolution.models.compare import ModelConfigError, compare_chat_models
from evolution.orchestrator.executor import execute_skill_run
from evolution.orchestrator.exporter import export_review_bundle
from evolution.orchestrator.gates import evaluate_run_gate
from evolution.orchestrator.loop import run_loop_once
from evolution.orchestrator.promoter import apply_gated_candidate, draft_pr_text
from evolution.orchestrator.run_manager import create_skill_run
from evolution.repos.git import get_git_snapshot
from evolution.repos.targets import scan_skill_targets
from evolution.traces.jsonl import failed_traces_to_eval_examples, load_trace_jsonl, scan_traces_for_secrets


@click.group()
@click.option(
    "--root",
    type=click.Path(path_type=Path),
    default=Path(".evolution"),
    show_default=True,
    help="Evolution state root containing config.yaml, evolution.db, and artifacts.",
)
@click.pass_context
def main(ctx: click.Context, root: Path):
    """Hermes Agent self-evolution control plane."""
    ctx.ensure_object(dict)
    ctx.obj["root"] = root
    ctx.obj["store"] = EvolutionStore(root / "evolution.db")


@main.command("init")
@click.pass_context
def init_cmd(ctx: click.Context):
    """Initialize the evolution state root."""
    root = ctx.obj["root"]
    config_path = init_evolution_root(root)
    click.echo(f"Initialized evolution root: {root}")
    click.echo(f"Config: {config_path}")
    click.echo(f"Database: {root / 'evolution.db'}")


@main.group("content")
def content_group():
    """Validate content/video production packages."""


@content_group.command("validate")
@click.argument("project_path", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--write/--no-write", default=True, show_default=True, help="Write runtime_check.json and qa_verdict.json into the package root.")
@click.option("--allow-hold", is_flag=True, help="Exit zero even when the package verdict is HOLD; useful for audits.")
@click.option("--json", "json_output", is_flag=True, help="Print full JSON validation result.")
def content_validate(project_path: Path, write: bool, allow_hold: bool, json_output: bool):
    """Validate one content package and emit runtime/QA verdict artifacts."""
    try:
        result = validate_content_package(project_path, write=write)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc

    if json_output:
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        click.echo(f"Content package verdict={result['verdict']} project={result['project_path']}")
        click.echo(f"runtime_check={result['runtime_check_path']}")
        click.echo(f"qa_verdict={result['qa_verdict_path']}")
        for hold in result["holds"]:
            click.echo(f"HOLD {hold['id']}: {hold['message']}")

    if result["verdict"] != "pass" and not allow_hold:
        raise click.ClickException(f"content package verdict={result['verdict']}")


@main.group("repo")
def repo_group():
    """Manage target repositories."""


@repo_group.command("add")
@click.argument("name")
@click.option("--path", "repo_path", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--url", default=None, help="Repository URL for metadata.")
@click.option("--default-branch", default="main", show_default=True)
@click.pass_context
def repo_add(ctx: click.Context, name: str, repo_path: Path, url: str | None, default_branch: str):
    """Register or update a repository."""
    store = _store(ctx)
    repo = store.add_repository(name, repo_path.resolve(), url=url, default_branch=default_branch)
    click.echo(f"Registered repository {repo['name']}: {repo['local_path']}")


@repo_group.command("snapshot")
@click.argument("name")
@click.pass_context
def repo_snapshot(ctx: click.Context, name: str):
    """Record current git snapshot for a registered repository."""
    store = _store(ctx)
    repo = _require_repo(store, name)
    snapshot_info = get_git_snapshot(repo["local_path"])
    snapshot = store.add_repo_snapshot(
        repository_id=repo["id"],
        git_sha=snapshot_info.git_sha,
        branch=snapshot_info.branch,
        dirty=snapshot_info.dirty,
        diff_sha256=snapshot_info.diff_sha256,
    )
    click.echo(
        f"Recorded snapshot {snapshot['id']} for {name}: "
        f"{snapshot['git_sha']} {snapshot['branch']} dirty={snapshot['dirty']}"
    )


@main.group("targets")
def targets_group():
    """Discover and list evolvable targets."""


@targets_group.command("scan")
@click.option("--repo", "repo_name", required=True, help="Registered repository name.")
@click.option("--type", "target_type", default="skill", type=click.Choice(["skill"]), show_default=True)
@click.pass_context
def targets_scan(ctx: click.Context, repo_name: str, target_type: str):
    """Scan a repository for evolvable targets and persist them."""
    store = _store(ctx)
    repo = _require_repo(store, repo_name)
    if target_type != "skill":
        raise click.ClickException(f"Unsupported target type: {target_type}")

    specs = scan_skill_targets(repo["local_path"])
    for spec in specs:
        store.upsert_target(
            repository_id=repo["id"],
            target_type=spec.target_type,
            name=spec.name,
            file_path=spec.file_path,
            selector=spec.selector,
            metadata=spec.metadata,
        )

    plural = "target" if len(specs) == 1 else "targets"
    click.echo(f"Scanned {len(specs)} {plural} for {repo_name}")
    for spec in specs:
        click.echo(f"- {spec.target_type}:{spec.name} {spec.file_path}")


@targets_group.command("list")
@click.option("--repo", "repo_name", default=None, help="Filter by registered repository name.")
@click.option("--type", "target_type", default=None, type=click.Choice(["skill"]))
@click.pass_context
def targets_list(ctx: click.Context, repo_name: str | None, target_type: str | None):
    """List persisted targets."""
    store = _store(ctx)
    repository_id = None
    if repo_name:
        repository_id = _require_repo(store, repo_name)["id"]
    targets = store.list_targets(repository_id=repository_id, target_type=target_type)
    if not targets:
        click.echo("No targets")
        return
    for target in targets:
        click.echo(f"{target['id']} {target['target_type']}:{target['name']} {target['file_path']}")


@main.group("dataset")
def dataset_group():
    """Build and inspect evaluation datasets."""


@dataset_group.command("build")
@click.option("--target", "target_ref", required=True, help="Target reference like skill:github-code-review.")
@click.option("--source", required=True, type=click.Choice(["golden"]), help="Dataset source.")
@click.option("--path", "dataset_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--version", default="v1", show_default=True)
@click.pass_context
def dataset_build(ctx: click.Context, target_ref: str, source: str, dataset_path: Path, version: str):
    """Build a dataset from a supported source and persist it."""
    store = _store(ctx)
    root: Path = ctx.obj["root"]
    target_type, target_name = _parse_target_ref(target_ref)
    target = store.get_target_by_name(target_type, target_name)
    if not target:
        raise click.ClickException(f"Target not found: {target_ref}. Run targets scan first.")

    if source != "golden":
        raise click.ClickException(f"Unsupported dataset source: {source}")

    splits = load_golden_splits(dataset_path)
    examples = flatten_splits(splits)
    scan_report = scan_examples_for_secrets(examples)
    if scan_report["status"] != "passed":
        raise click.ClickException(f"secret scan failed: {scan_report['matches']}")

    dataset_dir = root / "datasets" / target_type / target_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_spec = {split: len(rows) for split, rows in splits.items()}
    manifest = {
        "schema_version": 1,
        "target": target_ref,
        "source": source,
        "source_path": str(dataset_path),
        "version": version,
        "split_spec": split_spec,
        "example_count": len(examples),
        "secret_scan": scan_report,
        "pii_scan": {"status": "not_implemented"},
    }
    manifest_ref = ArtifactStore(root).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        suffix=".json",
        kind="dataset",
        mime_type="application/json",
        metadata={"target": target_ref, "source": source, "version": version},
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
        source=source,
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
            source=source,
            task_input=example["task_input"],
            expected_behavior=example["expected_behavior"],
            difficulty=example.get("difficulty"),
            category=example.get("category"),
            metadata={k: v for k, v in example.items() if k not in {"split", "task_input", "expected_behavior", "difficulty", "category"}},
        )

    (dataset_dir / f"{dataset['id']}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    click.echo(f"Built dataset {dataset['id']} for {target_ref}: {len(examples)} examples")
    click.echo(f"Manifest artifact: {manifest_ref.storage_uri}")


@dataset_group.command("list")
@click.option("--target", "target_ref", default=None, help="Optional target reference like skill:github-code-review.")
@click.pass_context
def dataset_list(ctx: click.Context, target_ref: str | None):
    """List persisted datasets."""
    store = _store(ctx)
    target_id = None
    if target_ref:
        target_type, target_name = _parse_target_ref(target_ref)
        target = store.get_target_by_name(target_type, target_name)
        if not target:
            raise click.ClickException(f"Target not found: {target_ref}")
        target_id = target["id"]
    datasets = store.list_datasets(target_id=target_id)
    if not datasets:
        click.echo("No datasets")
        return
    for dataset in datasets:
        click.echo(
            f"{dataset['id']} {dataset['source']} {dataset['version']} "
            f"{dataset['example_count']} examples target={dataset['target_id']}"
        )


@main.group("traces")
def traces_group():
    """Import attempt traces and convert failures into eval datasets."""


@traces_group.command("import")
@click.option("--target", "target_ref", required=True, help="Target reference like skill:github-code-review.")
@click.option("--source", required=True, help="Trace source label, e.g. hermes-session.")
@click.option("--path", "trace_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.pass_context
def traces_import(ctx: click.Context, target_ref: str, source: str, trace_path: Path):
    """Import JSONL attempt traces for a target."""
    store = _store(ctx)
    target_type, target_name = _parse_target_ref(target_ref)
    target = store.get_target_by_name(target_type, target_name)
    if not target:
        raise click.ClickException(f"Target not found: {target_ref}. Run targets scan first.")

    try:
        traces = load_trace_jsonl(trace_path, default_source=source)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    scan_report = scan_traces_for_secrets(traces)
    if scan_report["status"] != "passed":
        raise click.ClickException(f"secret scan failed: {scan_report['matches']}")

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

    click.echo(f"Imported {len(imported)} traces for {target_ref}")


@traces_group.command("list")
@click.option("--target", "target_ref", default=None, help="Optional target reference like skill:github-code-review.")
@click.option("--status", "status", default=None, type=click.Choice(["failure", "success"]))
@click.pass_context
def traces_list(ctx: click.Context, target_ref: str | None, status: str | None):
    """List imported attempt traces."""
    store = _store(ctx)
    target_id = None
    if target_ref:
        target_type, target_name = _parse_target_ref(target_ref)
        target = store.get_target_by_name(target_type, target_name)
        if not target:
            raise click.ClickException(f"Target not found: {target_ref}")
        target_id = target["id"]

    traces = store.list_attempt_traces(target_id=target_id, status=status)
    if not traces:
        click.echo("No traces")
        return
    for trace in traces:
        reason = trace["failure_reason"] or ""
        click.echo(f"{trace['id']} {trace['status']} {trace['source']} target={trace['target_id']} {reason}")


@traces_group.command("dataset")
@click.option("--target", "target_ref", required=True, help="Target reference like skill:github-code-review.")
@click.option("--version", default="trace-v1", show_default=True)
@click.pass_context
def traces_dataset(ctx: click.Context, target_ref: str, version: str):
    """Build an eval dataset from failed attempt traces."""
    store = _store(ctx)
    root: Path = ctx.obj["root"]
    target_type, target_name = _parse_target_ref(target_ref)
    target = store.get_target_by_name(target_type, target_name)
    if not target:
        raise click.ClickException(f"Target not found: {target_ref}")

    trace_rows = store.list_attempt_traces(target_id=target["id"], status="failure")
    if not trace_rows:
        raise click.ClickException(f"No failed traces for {target_ref}")
    normalized_traces = [_trace_row_for_examples(row) for row in trace_rows]
    examples = _assign_trace_splits(failed_traces_to_eval_examples(normalized_traces))
    scan_report = scan_examples_for_secrets(examples)
    if scan_report["status"] != "passed":
        raise click.ClickException(f"secret scan failed: {scan_report['matches']}")

    split_spec = _split_spec(examples)
    manifest = {
        "schema_version": 1,
        "target": target_ref,
        "source": "traces",
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
        metadata={"target": target_ref, "source": "traces", "version": version},
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
        source="traces",
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
    click.echo(f"Built trace dataset {dataset['id']} for {target_ref}: {len(examples)} examples")
    click.echo(f"Manifest artifact: {manifest_ref.storage_uri}")


@main.group("models")
def models_group():
    """Smoke-test and compare OpenAI-compatible model endpoints."""


@models_group.command("compare")
@click.option("--provider", default="deepseek", show_default=True, help="Provider profile name. Use custom/openai-compatible with --base-url and --api-key-env.")
@click.option("--model", "models", multiple=True, required=True, help="Model ID to test. Repeat for comparison.")
@click.option("--base-url", default=None, help="Override provider base URL.")
@click.option("--api-key-env", default=None, help="Environment variable containing the provider API key.")
@click.option("--prompt", default=None, help="Prompt to send to every model.")
@click.option("--prompt-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None, help="Read comparison prompt from a file.")
@click.option("--max-tokens", default=256, show_default=True, type=int)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--timeout", default=60.0, show_default=True, type=float)
@click.option("--extra-body-json", default=None, help="Provider-specific OpenAI SDK extra_body JSON, e.g. '{\"thinking\":{\"type\":\"disabled\"}}'.")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
def models_compare(
    provider: str,
    models: tuple[str, ...],
    base_url: str | None,
    api_key_env: str | None,
    prompt: str | None,
    prompt_file: Path | None,
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body_json: str | None,
    json_output: bool,
):
    """Compare supplied model IDs with the same prompt; no model IDs are implicit."""
    prompt_text = _load_model_compare_prompt(prompt, prompt_file)
    extra_body = _parse_extra_body_json(extra_body_json)
    try:
        results = compare_chat_models(
            models=list(models),
            prompt=prompt_text,
            provider=provider,
            base_url=base_url,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
        )
    except ModelConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    if json_output:
        click.echo(json.dumps({"provider": provider, "models": list(models), "results": results}, indent=2, sort_keys=True))
        return

    for result in results:
        status = "ok" if result["ok"] else "error"
        base = f"{result['model']} {status} latency_ms={result['latency_ms']} tokens={result['total_tokens']}"
        if result["ok"]:
            preview = " ".join(result.get("output_text", "").split())[:240]
            click.echo(f"{base} output={preview}")
        else:
            click.echo(f"{base} error={result.get('error')}")


def _load_model_compare_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt and prompt_file:
        raise click.ClickException("Use either --prompt or --prompt-file, not both")
    if prompt_file:
        loaded = prompt_file.read_text()
    else:
        loaded = prompt or ""
    if not loaded.strip():
        raise click.ClickException("Provide --prompt or --prompt-file")
    return loaded


def _parse_extra_body_json(extra_body_json: str | None) -> dict | None:
    if not extra_body_json:
        return None
    try:
        parsed = json.loads(extra_body_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid --extra-body-json: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise click.ClickException("--extra-body-json must decode to a JSON object")
    return parsed


@main.group("loop")
def loop_group():
    """Run one-shot evolution loops."""


@loop_group.command("once")
@click.option("--target", "target_ref", required=True, help="Target reference like skill:github-code-review.")
@click.option("--trace-path", required=True, type=click.Path(path_type=Path, exists=True, dir_okay=False), help="JSONL traces to import before building the dataset.")
@click.option("--trace-source", default="hermes-session", show_default=True)
@click.option("--dataset-version", default="loop-v1", show_default=True)
@click.option("--engine", default="gepa", type=click.Choice(["gepa", "mipro"]), show_default=True)
@click.option("--iterations", default=5, show_default=True, type=int)
@click.option("--strategy", default="deterministic", type=click.Choice(["deterministic", "model-synthesis", "dspy-gepa"]), show_default=True)
@click.option("--provider", default="deepseek", show_default=True)
@click.option("--optimizer-model", default=None)
@click.option("--eval-model", default=None)
@click.option("--base-url", default=None)
@click.option("--api-key-env", default=None)
@click.option("--max-tokens", default=2048, show_default=True, type=int)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--timeout", default=60.0, show_default=True, type=float)
@click.option("--extra-body-json", default=None, help="Provider-specific OpenAI SDK extra_body JSON.")
@click.option("--scoring-strategy", default="deterministic-rubric", type=click.Choice(["deterministic-rubric", "keyword-overlap", "model-rubric"]), show_default=True)
@click.option("--judge-model", default=None)
@click.option("--dspy-model-prefix", default=None)
@click.option("--gepa-max-full-evals", default=None, type=int)
@click.option("--gepa-reflection-minibatch-size", default=3, show_default=True, type=int)
@click.option("--gepa-log-dir", default=None)
@click.option("--min-holdout-improvement", default=0.0, show_default=True, type=float)
@click.option("--preferred-metric", default="rubric_score", show_default=True)
@click.option("--export-out", default=None, type=click.Path(path_type=Path, file_okay=False))
@click.pass_context
def loop_once(
    ctx: click.Context,
    target_ref: str,
    trace_path: Path,
    trace_source: str,
    dataset_version: str,
    engine: str,
    iterations: int,
    strategy: str,
    provider: str,
    optimizer_model: str | None,
    eval_model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body_json: str | None,
    scoring_strategy: str,
    judge_model: str | None,
    dspy_model_prefix: str | None,
    gepa_max_full_evals: int | None,
    gepa_reflection_minibatch_size: int,
    gepa_log_dir: str | None,
    min_holdout_improvement: float,
    preferred_metric: str,
    export_out: Path | None,
):
    """Run import→dataset→execute→gate→export once."""
    extra_body = _parse_extra_body_json(extra_body_json)
    try:
        result = run_loop_once(
            store=_store(ctx),
            root=ctx.obj["root"],
            target_ref=target_ref,
            trace_path=trace_path,
            trace_source=trace_source,
            dataset_version=dataset_version,
            engine=engine,
            iterations=iterations,
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
            min_holdout_improvement=min_holdout_improvement,
            preferred_metric=preferred_metric,
            export_out=export_out,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Loop once completed target={result['target']} traces={result['imported_traces']} "
        f"dataset={result['dataset']['id']} run={result['run']['id']} "
        f"gate={result['gate']['decision']} bundle_dir={result['export']['bundle_dir']}"
    )


@main.group("run")
def run_group():
    """Create and execute evolution run records."""


@run_group.command("skill")
@click.option("--target", "target_ref", required=True, help="Target reference like skill:github-code-review.")
@click.option("--dataset", "dataset_id", required=True, help="Persisted dataset id.")
@click.option("--engine", default="gepa", type=click.Choice(["gepa", "mipro"]), show_default=True)
@click.option("--iterations", default=10, show_default=True, type=int)
@click.pass_context
def run_skill(ctx: click.Context, target_ref: str, dataset_id: str, engine: str, iterations: int):
    """Create a pending skill evolution run record."""
    try:
        run = create_skill_run(
            store=_store(ctx),
            target_ref=target_ref,
            dataset_id=dataset_id,
            engine=engine,
            iterations=iterations,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Created run {run['id']} for {target_ref} dataset={dataset_id} engine={engine} status={run['status']}")


@run_group.command("execute")
@click.argument("run_id")
@click.option("--strategy", default="deterministic", type=click.Choice(["deterministic", "model-synthesis", "dspy-gepa"]), show_default=True)
@click.option("--provider", default="deepseek", show_default=True, help="Provider profile for model-backed strategies.")
@click.option("--optimizer-model", default=None, help="Model ID for model-backed strategies, e.g. deepseek-v4-pro.")
@click.option("--eval-model", default=None, help="Eval/task model ID for DSPy/GEPA, e.g. deepseek-v4-flash.")
@click.option("--base-url", default=None, help="Override provider base URL.")
@click.option("--api-key-env", default=None, help="Environment variable holding provider API key.")
@click.option("--max-tokens", default=2048, show_default=True, type=int)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--timeout", default=60.0, show_default=True, type=float)
@click.option("--extra-body-json", default=None, help="Provider-specific OpenAI SDK extra_body JSON.")
@click.option("--scoring-strategy", default="deterministic-rubric", type=click.Choice(["deterministic-rubric", "keyword-overlap", "model-rubric"]), show_default=True)
@click.option("--judge-model", default=None, help="Judge model for model-rubric scoring. Defaults to --eval-model or --optimizer-model.")
@click.option("--dspy-model-prefix", default=None, help="DSPy/LiteLLM provider prefix for bare OpenAI-compatible model IDs, e.g. openai.")
@click.option("--gepa-max-full-evals", default=None, type=int, help="GEPA max_full_evals budget. Defaults to run iterations.")
@click.option("--gepa-reflection-minibatch-size", default=3, show_default=True, type=int)
@click.option("--gepa-log-dir", default=None, help="Optional GEPA log directory.")
@click.pass_context
def run_execute(
    ctx: click.Context,
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
    extra_body_json: str | None,
    scoring_strategy: str,
    judge_model: str | None,
    dspy_model_prefix: str | None,
    gepa_max_full_evals: int | None,
    gepa_reflection_minibatch_size: int,
    gepa_log_dir: str | None,
):
    """Execute a pending run and persist candidate/evaluation artifacts."""
    extra_body = _parse_extra_body_json(extra_body_json)
    try:
        result = execute_skill_run(
            store=_store(ctx),
            root=ctx.obj["root"],
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
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Executed run {run_id} status={result['run']['status']} "
        f"candidates={len(result['candidates'])} evaluations={len(result['evaluations'])} "
        f"manifest={result['manifest_artifact_id']}"
    )


@run_group.command("gate")
@click.argument("run_id")
@click.option("--min-holdout-improvement", default=0.0, show_default=True, type=float)
@click.option("--preferred-metric", default="rubric_score", show_default=True, help="Metric to prefer for holdout gate; falls back to available scores.")
@click.pass_context
def run_gate(ctx: click.Context, run_id: str, min_holdout_improvement: float, preferred_metric: str):
    """Evaluate benchmark/constraint gate for a completed run."""
    try:
        result = evaluate_run_gate(
            store=_store(ctx),
            root=ctx.obj["root"],
            run_id=run_id,
            min_holdout_improvement=min_holdout_improvement,
            preferred_metric=preferred_metric,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    reasons = ",".join(result["reasons"]) if result["reasons"] else "none"
    click.echo(
        f"Gate run {run_id} decision={result['decision']} "
        f"candidate={result['candidate_id']} "
        f"metric={result['metrics']['metric_name']} "
        f"holdout_improvement={result['metrics']['holdout_improvement']} "
        f"reasons={reasons}"
    )


@run_group.command("export")
@click.argument("run_id")
@click.option("--out", "out_dir", default=None, type=click.Path(path_type=Path, file_okay=False), help="Directory to write review bundles into.")
@click.option("--allow-hold", is_flag=True, help="Export even when latest gate decision is hold.")
@click.pass_context
def run_export(ctx: click.Context, run_id: str, out_dir: Path | None, allow_hold: bool):
    """Export a human-review bundle for the latest gated candidate."""
    try:
        result = export_review_bundle(
            store=_store(ctx),
            root=ctx.obj["root"],
            run_id=run_id,
            out_dir=out_dir,
            allow_hold=allow_hold,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Exported review bundle for run {run_id} decision={result['gate_decision']} "
        f"candidate={result['candidate_id']} manifest={result['manifest_artifact_id']}"
    )
    click.echo(f"bundle_dir={result['bundle_dir']}")


@run_group.command("apply")
@click.argument("run_id")
@click.option("--apply", "apply_mode", is_flag=True, help="Actually write the evolved candidate locally. Default is dry-run.")
@click.option("--branch", default=None, help="Optional local branch to create/switch before applying.")
@click.option("--commit", is_flag=True, help="Create a local commit after applying. Never pushes.")
@click.option("--message", default=None, help="Commit message when --commit is used.")
@click.option("--allow-hold", is_flag=True, help="Allow applying/exporting a HOLD gate candidate.")
@click.option("--allow-dirty", is_flag=True, help="Allow applying into a dirty git repository.")
@click.pass_context
def run_apply(
    ctx: click.Context,
    run_id: str,
    apply_mode: bool,
    branch: str | None,
    commit: bool,
    message: str | None,
    allow_hold: bool,
    allow_dirty: bool,
):
    """Safely dry-run or locally apply a gated candidate. No push, no merge."""
    try:
        result = apply_gated_candidate(
            store=_store(ctx),
            root=ctx.obj["root"],
            run_id=run_id,
            branch=branch,
            dry_run=not apply_mode,
            commit=commit,
            message=message,
            allow_hold=allow_hold,
            allow_dirty=allow_dirty,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Apply run {run_id} mode={result['mode']} gate={result['gate_decision']} "
        f"target={result['target_file']} branch={result['branch']} "
        f"mutated={result['mutated']} committed={result['committed']} pushed={result['pushed']}"
    )


@run_group.command("pr-draft")
@click.argument("run_id")
@click.option("--branch", default=None, help="Branch name to include in the PR draft.")
@click.option("--allow-hold", is_flag=True, help="Draft PR text even when latest gate is HOLD.")
@click.pass_context
def run_pr_draft(ctx: click.Context, run_id: str, branch: str | None, allow_hold: bool):
    """Draft PR title/body for a gated candidate without mutating anything."""
    try:
        result = draft_pr_text(
            store=_store(ctx),
            root=ctx.obj["root"],
            run_id=run_id,
            branch=branch,
            allow_hold=allow_hold,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"title={result['title']}")
    click.echo(f"branch={result['branch']}")
    click.echo(result["body"])


@main.group("runs")
def runs_group():
    """Inspect evolution runs."""


@runs_group.command("list")
@click.pass_context
def runs_list(ctx: click.Context):
    """List recorded evolution runs."""
    runs = _store(ctx).list_runs()
    if not runs:
        click.echo("No runs")
        return
    for run in runs:
        click.echo(f"{run['id']} {run['status']} {run['engine']} target={run['target_id']}")


@runs_group.command("show")
@click.argument("run_id")
@click.pass_context
def runs_show(ctx: click.Context, run_id: str):
    """Show one evolution run."""
    run = _store(ctx).get_run(run_id)
    if not run:
        raise click.ClickException(f"Run not found: {run_id}")
    for key, value in run.items():
        click.echo(f"{key}: {value}")


def _store(ctx: click.Context) -> EvolutionStore:
    store: EvolutionStore = ctx.obj["store"]
    store.init_schema()
    return store


def _require_repo(store: EvolutionStore, name: str) -> dict:
    repo = store.get_repository(name)
    if not repo:
        raise click.ClickException(f"Repository not found: {name}. Run repo add first.")
    return repo


def _parse_target_ref(target_ref: str) -> tuple[str, str]:
    if ":" not in target_ref:
        raise click.ClickException("Target must be formatted as <type>:<name>, e.g. skill:github-code-review")
    target_type, target_name = target_ref.split(":", 1)
    if not target_type or not target_name:
        raise click.ClickException("Target must be formatted as <type>:<name>, e.g. skill:github-code-review")
    return target_type, target_name


def _trace_row_for_examples(row: dict) -> dict:
    data = dict(row)
    data["metadata"] = row.get("metadata_json") or {}
    return data


def _assign_trace_splits(examples: list[dict]) -> list[dict]:
    split_cycle = ["train", "val", "holdout"]
    assigned = []
    for index, example in enumerate(examples):
        assigned.append({**example, "split": split_cycle[index % len(split_cycle)]})
    return assigned


def _split_spec(examples: list[dict]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "holdout": 0}
    for example in examples:
        counts[example["split"]] = counts.get(example["split"], 0) + 1
    return {split: count for split, count in counts.items() if count}


if __name__ == "__main__":
    main()
