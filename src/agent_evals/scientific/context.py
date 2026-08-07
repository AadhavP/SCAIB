"""Mutable runtime state for a scientific pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_evals.scientific.artifacts.models import OperationRecord, ScientificArtifact
from agent_evals.scientific.artifacts.storage import ArtifactStore


def utc_now() -> datetime:
    """Return an aware timestamp suitable for persisted provenance."""
    return datetime.now(UTC)


@dataclass
class OperationOutput:
    """Result returned by one concrete scientific operation."""

    adata: Any | None = None
    artifacts: list[ScientificArtifact] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScientificContext:
    """Execution state shared by operations without coupling them to a backend."""

    adata: Any
    dataset_metadata: dict[str, Any]
    artifact_store: ArtifactStore
    workspace: Path
    artifacts: dict[str, ScientificArtifact] = field(default_factory=dict)
    operations: list[OperationRecord] = field(default_factory=list)
    started_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_operation(
        self,
        operation: str,
        parameters: dict[str, Any],
        *,
        status: str,
        artifact_ids: list[str] | None = None,
        error: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> OperationRecord:
        """Append an auditable operation record and return it."""
        record = OperationRecord(
            operation=operation,
            parameters=parameters,
            status=status,
            artifact_ids=artifact_ids or [],
            error=error,
            started_at=started_at or utc_now(),
            completed_at=completed_at or utc_now(),
        )
        self.operations.append(record)
        return record

    def add_artifact(self, artifact: ScientificArtifact) -> None:
        """Register a materialized artifact by its stable ID."""
        self.artifacts[artifact.artifact_id] = artifact

