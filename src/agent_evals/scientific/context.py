"""Mutable runtime state for a scientific pipeline execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_evals.core.reference_columns import (
    AGENT_CLUSTER_COLUMNS,
    AGENT_PREDICTION_COLUMNS,
    RESERVED_REFERENCE_COLUMNS,
)
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
    #: Observation columns written by agent actions during this run. Scoring may
    #: only treat a column as a prediction when the agent actually produced it;
    #: otherwise pre-existing dataset columns would leak into the score.
    agent_produced_columns: set[str] = field(default_factory=set)

    def record_produced_columns(self, columns: Iterable[str]) -> None:
        """Attribute observation columns to the agent that just acted."""
        for column in columns:
            name = str(column)
            if name in RESERVED_REFERENCE_COLUMNS:
                raise ValueError(
                    f"operation attempted to write reserved reference column '{name}'"
                )
            self.agent_produced_columns.add(name)

    def agent_prediction_column(self) -> str | None:
        """Return the agent's prediction column, or None when it produced none."""
        return self._agent_column(AGENT_PREDICTION_COLUMNS)

    def agent_cluster_column(self) -> str | None:
        """Return the agent's grouping column, or None when it produced none.

        Resolved here rather than at the call site so the evaluator cannot look
        for a name the operations never write. That mismatch is not hypothetical:
        it silently disabled ``clustering.ari`` for the entire life of the typed
        path before this method existed.
        """
        return self._agent_column(AGENT_CLUSTER_COLUMNS)

    def _agent_column(self, candidates: Iterable[str]) -> str | None:
        """Return the first candidate this run's agent both wrote and left behind.

        Both halves are required. ``agent_produced_columns`` alone would accept a
        column an agent wrote and a later step dropped, and membership in ``obs``
        alone would accept a column the dataset shipped -- which for a grouping
        column means scoring reference biology as the agent's own work.
        """
        return next(
            (
                column
                for column in candidates
                if column in self.agent_produced_columns and column in self.adata.obs
            ),
            None,
        )

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

