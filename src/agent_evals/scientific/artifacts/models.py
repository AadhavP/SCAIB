"""Persisted scientific artifact and operation provenance models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.environment.models import ArtifactRecord


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


class ScientificArtifact(BaseModel):
    """A concrete file produced by a scientific operation."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    path: Path
    kind: str = Field(min_length=1)
    format: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_artifact_record(self) -> ArtifactRecord:
        """Adapt this model to the framework's backend-neutral artifact type."""
        return ArtifactRecord(
            artifact_id=self.artifact_id,
            kind=self.kind,
            format=self.format,
            uri=str(self.path),
            checksum=self.checksum,
            # Existence is not validation -- an empty file exists. The bit is set
            # by the environment's artifact validator, which reads the file and
            # checks it against the rules the benchmark declared. Leaving it False
            # here means "not yet checked", which is what is true at this point.
            validated=False,
            metadata=self.metadata,
        )


class OperationRecord(BaseModel):
    """One operation's parameters, outcome, and produced artifacts."""

    model_config = ConfigDict(extra="forbid")

    operation: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(min_length=1)
    artifact_ids: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)
    error: str | None = None
