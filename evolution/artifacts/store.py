"""Content-addressed artifact storage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an immutable artifact written by ArtifactStore."""

    content_sha256: str
    storage_uri: str
    size_bytes: int
    kind: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "storage_uri": self.storage_uri,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
        }


class ArtifactStore:
    """Stores immutable artifacts under blobs/sha256/<prefix>/<hash><suffix>."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.blob_root = self.root / "blobs" / "sha256"

    def write_text(
        self,
        content: str,
        suffix: str = ".txt",
        kind: str | None = None,
        mime_type: str | None = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        return self.write_bytes(
            content.encode("utf-8"),
            suffix=suffix,
            kind=kind,
            mime_type=mime_type,
            metadata=metadata,
        )

    def write_bytes(
        self,
        content: bytes,
        suffix: str = "",
        kind: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        digest = hashlib.sha256(content).hexdigest()
        artifact_path = self._path_for(digest, suffix)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if not artifact_path.exists():
            artifact_path.write_bytes(content)
        return ArtifactRef(
            content_sha256=digest,
            storage_uri=str(artifact_path),
            size_bytes=len(content),
            kind=kind,
            mime_type=mime_type,
            metadata=metadata or {},
        )

    def read_text(self, ref: ArtifactRef | str) -> str:
        return self._resolve_path(ref).read_text()

    def read_bytes(self, ref: ArtifactRef | str) -> bytes:
        return self._resolve_path(ref).read_bytes()

    def _path_for(self, digest: str, suffix: str = "") -> Path:
        if suffix and not suffix.startswith("."):
            suffix = f".{suffix}"
        return self.blob_root / digest[:2] / f"{digest}{suffix}"

    def _resolve_path(self, ref: ArtifactRef | str) -> Path:
        if isinstance(ref, ArtifactRef):
            return Path(ref.storage_uri)
        candidate = Path(ref)
        if candidate.exists():
            return candidate
        if len(ref) >= 64:
            matches = list((self.blob_root / ref[:2]).glob(f"{ref}*"))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Artifact not found: {ref}")
