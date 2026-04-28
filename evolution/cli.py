"""Production CLI foundation for Hermes self-evolution."""

from __future__ import annotations

from pathlib import Path
import json

import click

from evolution.artifacts.store import ArtifactStore
from evolution.config_file import init_evolution_root
from evolution.datasets.golden import flatten_splits, load_golden_splits
from evolution.datasets.redaction import scan_examples_for_secrets
from evolution.db.store import EvolutionStore
from evolution.orchestrator.executor import execute_skill_run
from evolution.orchestrator.run_manager import create_skill_run
from evolution.repos.git import get_git_snapshot
from evolution.repos.targets import scan_skill_targets


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
@click.option("--strategy", default="deterministic", type=click.Choice(["deterministic"]), show_default=True)
@click.pass_context
def run_execute(ctx: click.Context, run_id: str, strategy: str):
    """Execute a pending run and persist candidate/evaluation artifacts."""
    try:
        result = execute_skill_run(
            store=_store(ctx),
            root=ctx.obj["root"],
            run_id=run_id,
            strategy=strategy,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Executed run {run_id} status={result['run']['status']} "
        f"candidates={len(result['candidates'])} evaluations={len(result['evaluations'])} "
        f"manifest={result['manifest_artifact_id']}"
    )


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


if __name__ == "__main__":
    main()
