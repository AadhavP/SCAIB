"""Local workspace lifecycle abstraction for agent runs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkspaceStatus(StrEnum):
    """Lifecycle status of a workspace."""

    CREATED = "created"
    READY = "ready"
    CLOSED = "closed"
    FAILED = "failed"


class WorkspaceManifest(BaseModel):
    """Serializable workspace locations and metadata."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    root: Path
    input_dir: Path
    artifact_dir: Path
    log_dir: Path
    status: WorkspaceStatus = WorkspaceStatus.CREATED
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocalWorkspace:
    """Create a local process workspace without imposing a compute backend.

    The directory layout is intentionally small and portable. Docker, Slurm,
    cloud, and distributed implementations can later satisfy the same manifest
    contract without changing the harness or trajectory format.
    """

    def __init__(self, root: Path, *, workspace_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.manifest = WorkspaceManifest(
            workspace_id=workspace_id,
            root=root,
            input_dir=root / "inputs",
            artifact_dir=root / "artifacts",
            log_dir=root / "logs",
            metadata=metadata or {},
        )

    async def initialize(self) -> WorkspaceManifest:
        """Create the local directory layout and mark the workspace ready."""
        try:
            for path in (
                self.manifest.root,
                self.manifest.input_dir,
                self.manifest.artifact_dir,
                self.manifest.log_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
            self.manifest.status = WorkspaceStatus.READY
        except OSError:
            self.manifest.status = WorkspaceStatus.FAILED
            raise
        return self.manifest.model_copy(deep=True)

    async def close(self) -> WorkspaceManifest:
        """Mark the workspace closed without deleting user artifacts."""
        if self.manifest.status == WorkspaceStatus.READY:
            self.manifest.status = WorkspaceStatus.CLOSED
        return self.manifest.model_copy(deep=True)


__all__ = ["LocalWorkspace", "WorkspaceManifest", "WorkspaceStatus"]
