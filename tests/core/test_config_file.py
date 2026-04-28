"""Tests for evolution root initialization."""

from evolution.config_file import init_evolution_root


def test_init_evolution_root_creates_config_db_and_artifact_dirs(tmp_path):
    root = tmp_path / ".evolution"

    config_path = init_evolution_root(root)

    assert config_path == root / "config.yaml"
    assert config_path.exists()
    assert (root / "evolution.db").exists()
    assert (root / "blobs" / "sha256").exists()
    assert "schema_version: 1" in config_path.read_text()
