"""Tests for artifact persistence metadata."""

from evolution.artifacts.store import ArtifactStore
from evolution.db.store import EvolutionStore


def test_artifact_metadata_roundtrip(tmp_path):
    artifact_store = ArtifactStore(tmp_path / ".evolution")
    db = EvolutionStore(tmp_path / ".evolution" / "evolution.db")
    db.init_schema()

    ref = artifact_store.write_text("baseline", suffix=".md", kind="baseline", metadata={"target": "skill:x"})
    row = db.add_artifact(
        kind=ref.kind,
        content_sha256=ref.content_sha256,
        storage_uri=ref.storage_uri,
        size_bytes=ref.size_bytes,
        mime_type="text/markdown",
        metadata=ref.metadata,
    )
    loaded = db.get_artifact(row["id"])

    assert loaded["content_sha256"] == ref.content_sha256
    assert loaded["metadata_json"]["target"] == "skill:x"
