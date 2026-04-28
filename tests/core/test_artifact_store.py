"""Tests for content-addressed artifact storage."""

import hashlib
from pathlib import Path

from evolution.artifacts.store import ArtifactStore


def test_write_text_stores_content_by_sha256(tmp_path):
    store = ArtifactStore(tmp_path)

    ref = store.write_text("hello", suffix=".md", kind="baseline")

    expected_hash = hashlib.sha256(b"hello").hexdigest()
    assert ref.content_sha256 == expected_hash
    assert ref.size_bytes == 5
    assert ref.kind == "baseline"
    assert ref.storage_uri.endswith(f"/blobs/sha256/{expected_hash[:2]}/{expected_hash}.md")
    assert Path(ref.storage_uri).read_text() == "hello"
    assert store.read_text(ref) == "hello"


def test_write_text_deduplicates_identical_content(tmp_path):
    store = ArtifactStore(tmp_path)

    first = store.write_text("same", suffix=".json", kind="metrics")
    second = store.write_text("same", suffix=".json", kind="metrics")

    assert first.content_sha256 == second.content_sha256
    assert first.storage_uri == second.storage_uri
    assert len(list((tmp_path / "blobs" / "sha256" / first.content_sha256[:2]).iterdir())) == 1


def test_artifact_ref_serializes_for_manifests(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.write_text("{}", suffix=".json", kind="manifest", metadata={"run_id": "run_1"})

    data = ref.to_dict()

    assert data["content_sha256"] == ref.content_sha256
    assert data["storage_uri"] == ref.storage_uri
    assert data["size_bytes"] == 2
    assert data["kind"] == "manifest"
    assert data["metadata"]["run_id"] == "run_1"
