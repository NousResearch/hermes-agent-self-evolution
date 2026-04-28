"""Production CLI foundation for Hermes self-evolution."""

from __future__ import annotations

from pathlib import Path

import click

from evolution.config_file import init_evolution_root
from evolution.db.store import EvolutionStore
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


if __name__ == "__main__":
    main()
