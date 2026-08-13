"""Small local artifact store used by the first scientific vertical slice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from agent_evals.scientific.artifacts.models import ScientificArtifact


class ArtifactStore(Protocol):
    """Persistence contract operations depend on."""

    root: Path

    def save_adata(self, artifact_id: str, adata: Any, *, metadata: dict[str, Any] | None = None) -> ScientificArtifact:
        """Persist an AnnData object."""

    def save_json(self, artifact_id: str, value: Any, *, metadata: dict[str, Any] | None = None) -> ScientificArtifact:
        """Persist JSON-compatible data."""

    def save_table(self, artifact_id: str, table: Any, *, metadata: dict[str, Any] | None = None) -> ScientificArtifact:
        """Persist a tabular object."""

    def save_text(
        self,
        artifact_id: str,
        text: str,
        *,
        file_format: str = "txt",
        metadata: dict[str, Any] | None = None,
    ) -> ScientificArtifact:
        """Persist a rendered document, taking its extension from the format."""


class LocalArtifactStore:
    """Filesystem-backed store with checksums for every materialized file."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_adata(self, artifact_id: str, adata: Any, *, metadata: dict[str, Any] | None = None) -> ScientificArtifact:
        path = self.root / f"{artifact_id}.h5ad"
        adata.write_h5ad(path)
        return self._artifact(artifact_id, path, "anndata", "h5ad", metadata)

    def save_json(self, artifact_id: str, value: Any, *, metadata: dict[str, Any] | None = None) -> ScientificArtifact:
        path = self.root / f"{artifact_id}.json"
        path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        return self._artifact(artifact_id, path, "json", "json", metadata)

    def save_table(self, artifact_id: str, table: Any, *, metadata: dict[str, Any] | None = None) -> ScientificArtifact:
        path = self.root / f"{artifact_id}.csv"
        table.to_csv(path, index=False)
        return self._artifact(artifact_id, path, "table", "csv", metadata)

    def save_text(
        self,
        artifact_id: str,
        text: str,
        *,
        file_format: str = "txt",
        metadata: dict[str, Any] | None = None,
    ) -> ScientificArtifact:
        # The extension comes from the caller's declared format rather than being
        # fixed, because Stage 3's validator dispatches on the *file* and not on
        # the benchmark's declaration -- a report written as ``.txt`` while the
        # benchmark declares ``html`` would leave its rules unevaluable against a
        # file sitting right there.
        suffix = file_format.strip().lstrip(".") or "txt"
        path = self.root / f"{artifact_id}.{suffix}"
        path.write_text(text, encoding="utf-8")
        return self._artifact(artifact_id, path, "report", suffix, metadata)

    def _artifact(
        self,
        artifact_id: str,
        path: Path,
        kind: str,
        file_format: str,
        metadata: dict[str, Any] | None,
    ) -> ScientificArtifact:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ScientificArtifact(
            artifact_id=artifact_id,
            path=path,
            kind=kind,
            format=file_format,
            checksum=digest,
            metadata=metadata or {},
        )


def finite_array(value: Any) -> bool:
    """Return whether a dense or sparse array contains only finite values."""
    if hasattr(value, "toarray"):
        value = value.toarray()
    return bool(np.isfinite(value).all())


__all__ = ["ArtifactStore", "LocalArtifactStore"]
