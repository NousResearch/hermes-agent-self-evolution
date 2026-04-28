"""Evolution root configuration helpers."""

from __future__ import annotations

from pathlib import Path

from evolution.artifacts.store import ArtifactStore
from evolution.db.store import EvolutionStore


def init_evolution_root(root: str | Path) -> Path:
    """Create .evolution root, config file, DB schema, and blob directories."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    ArtifactStore(root_path).blob_root.mkdir(parents=True, exist_ok=True)
    EvolutionStore(root_path / "evolution.db").init_schema()

    config_path = root_path / "config.yaml"
    if not config_path.exists():
        config_path.write_text(
            "schema_version: 1\n"
            "database: evolution.db\n"
            "artifact_root: .\n"
        )
    return config_path
