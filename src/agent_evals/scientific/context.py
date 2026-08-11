"""Mutable runtime state for a scientific pipeline execution."""

from __future__ import annotations

from collections.abc import Iterable
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


#: Observation columns that carry reference biology. These are evaluator inputs
#: and must never be read back as though the agent had predicted them.
RESERVED_REFERENCE_COLUMNS = frozenset(
    {
        "bulk_labels",
        "cell_type",
        "cell_type_ref",
        "known_labels",
        "reference_labels",
    }
)

#: Observation columns an agent may write to record its own predictions.
AGENT_PREDICTION_COLUMNS = ("predicted_labels", "predicted_cell_type")


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
        return next(
            (
                column
                for column in AGENT_PREDICTION_COLUMNS
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

