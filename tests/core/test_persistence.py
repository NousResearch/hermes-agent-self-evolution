"""Tests for the SQLite persistence layer."""

from evolution.db.store import EvolutionStore


def test_repository_target_run_roundtrip(tmp_path):
    db_path = tmp_path / ".evolution" / "evolution.db"
    repo_path = tmp_path / "hermes-agent"
    repo_path.mkdir()

    store = EvolutionStore(db_path)
    store.init_schema()

    repo = store.add_repository("hermes-agent", repo_path, url="https://example.invalid/repo.git")
    loaded_repo = store.get_repository("hermes-agent")

    assert loaded_repo["id"] == repo["id"]
    assert loaded_repo["local_path"] == str(repo_path)

    snapshot = store.add_repo_snapshot(
        repository_id=repo["id"],
        git_sha="abc123",
        branch="main",
        dirty=False,
    )
    assert snapshot["git_sha"] == "abc123"

    target = store.upsert_target(
        repository_id=repo["id"],
        target_type="skill",
        name="github-code-review",
        file_path="skills/github/github-code-review/SKILL.md",
        selector=None,
        metadata={"description": "Review PRs"},
    )
    targets = store.list_targets(repository_id=repo["id"], target_type="skill")

    assert len(targets) == 1
    assert targets[0]["id"] == target["id"]
    assert targets[0]["metadata_json"]["description"] == "Review PRs"

    run = store.create_run(
        target_id=target["id"],
        repository_snapshot_id=snapshot["id"],
        engine="gepa",
        config={"iterations": 3},
    )
    runs = store.list_runs()

    assert runs[0]["id"] == run["id"]
    assert runs[0]["status"] == "pending"
    assert runs[0]["config_json"]["iterations"] == 3


def test_init_schema_is_idempotent(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.db")

    store.init_schema()
    store.init_schema()

    assert store.list_repositories() == []
