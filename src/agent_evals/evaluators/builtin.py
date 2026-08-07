"""Small deterministic metrics that operate on recorded environment outputs."""

from __future__ import annotations

from typing import Any, TypeGuard

from agent_evals.benchmarks.schema import Direction
from agent_evals.environment.models import ActionStatus, ArtifactRecord
from agent_evals.evaluators.models import EvaluationLevel
from agent_evals.evaluators.registry import (
    MetricComputation,
    MetricContext,
    metric_registry,
)


def _artifacts(context: MetricContext) -> list[ArtifactRecord]:
    return list(context.snapshot.state.artifacts.values())


def _artifact(context: MetricContext, artifact_id: str | None = None) -> ArtifactRecord | None:
    candidates = _artifacts(context)
    if artifact_id is not None:
        return next((item for item in candidates if item.artifact_id == artifact_id), None)
    return candidates[-1] if candidates else None


def _metadata_number(artifact: ArtifactRecord, key: str) -> float | None:
    value = artifact.metadata.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _metadata_bool(artifact: ArtifactRecord, key: str) -> bool | None:
    value = artifact.metadata.get(key)
    return value if isinstance(value, bool) else None


@metric_registry.register(
    "execution-success",
    name="Execution success",
    description="Fraction of submitted actions that executed successfully.",
    level=EvaluationLevel.EXECUTION,
    direction=Direction.HIGHER_IS_BETTER,
)
def execution_success(context: MetricContext) -> MetricComputation:
    actions = context.snapshot.state.actions
    succeeded = sum(record.result.status == ActionStatus.SUCCEEDED for record in actions)
    total = len(actions)
    return MetricComputation(
        raw_value=(succeeded / total if total else 0.0),
        evidence=(f"{succeeded} of {total} submitted actions succeeded",),
        metadata={"succeeded": succeeded, "total": total},
    )


@metric_registry.register(
    "artifact-validity",
    name="Artifact validity",
    description="Fraction of expected artifacts present and marked validated.",
    level=EvaluationLevel.ARTIFACT,
    direction=Direction.HIGHER_IS_BETTER,
)
def artifact_validity(context: MetricContext) -> MetricComputation:
    expected = context.task_instance.expected_artifacts
    records = {artifact.artifact_id: artifact for artifact in _artifacts(context)}
    valid_ids = [
        artifact.id
        for artifact in expected
        if artifact.id in records and records[artifact.id].validated
    ]
    total = len(expected)
    return MetricComputation(
        raw_value=(len(valid_ids) / total if total else 0.0),
        evidence=(f"validated artifacts: {', '.join(valid_ids) or 'none'}",),
        artifact_ids=tuple(valid_ids),
        metadata={"valid": len(valid_ids), "expected": total},
    )


@metric_registry.register(
    "cell-retention",
    name="Cell retention",
    description="Ratio of cells retained by the recorded filtering operation.",
    level=EvaluationLevel.ARTIFACT,
    direction=Direction.HIGHER_IS_BETTER,
)
def cell_retention(context: MetricContext) -> MetricComputation:
    for artifact in reversed(_artifacts(context)):
        before = _metadata_number(artifact, "cells_before")
        after = _metadata_number(artifact, "cells_after")
        if before is not None and after is not None and before > 0:
            return MetricComputation(
                raw_value=after / before,
                evidence=(f"{artifact.artifact_id}: {after:g} of {before:g} cells retained",),
                artifact_ids=(artifact.artifact_id,),
            )
    return MetricComputation(raw_value=None, evidence=("no cell counts were recorded",))


@metric_registry.register(
    "embedding-validity",
    name="Embedding validity",
    description="Validates finite embedding coordinates and dimensions.",
    level=EvaluationLevel.ARTIFACT,
    direction=Direction.HIGHER_IS_BETTER,
)
def embedding_validity(context: MetricContext) -> MetricComputation:
    embeddings = [artifact for artifact in _artifacts(context) if artifact.kind == "embedding"]
    if not embeddings:
        return MetricComputation(raw_value=None, evidence=("no embedding artifact was recorded",))
    valid = True
    evidence: list[str] = []
    for artifact in embeddings:
        observations = _metadata_number(artifact, "n_observations")
        dimensions = _metadata_number(artifact, "n_dimensions")
        finite = _metadata_bool(artifact, "finite")
        artifact_valid = (
            artifact.validated
            and observations is not None
            and observations > 0
            and dimensions is not None
            and dimensions > 0
            and finite is True
        )
        valid = valid and artifact_valid
        evidence.append(f"{artifact.artifact_id}: {'valid' if artifact_valid else 'invalid'}")
    return MetricComputation(
        raw_value=1.0 if valid else 0.0,
        evidence=tuple(evidence),
        artifact_ids=tuple(artifact.artifact_id for artifact in embeddings),
    )


@metric_registry.register(
    "batch-mixing",
    name="Batch mixing",
    description="One minus the mean same-batch neighbor fraction in an embedding.",
    level=EvaluationLevel.METHOD,
    direction=Direction.HIGHER_IS_BETTER,
    required_artifacts=("corrected-embedding",),
)
def batch_mixing(context: MetricContext) -> MetricComputation:
    artifact = _artifact(context, "corrected-embedding")
    if artifact is None:
        return MetricComputation(raw_value=None, evidence=("corrected embedding is unavailable",))
    labels = artifact.metadata.get("batch_labels")
    neighbors = artifact.metadata.get("neighbors")
    if not _is_label_sequence(labels) or not _is_neighbor_sequence(neighbors):
        return MetricComputation(
            raw_value=None,
            evidence=("batch_labels and neighbors are required in embedding metadata",),
            artifact_ids=(artifact.artifact_id,),
        )
    fractions: list[float] = []
    for index, row in enumerate(neighbors):
        if index >= len(labels) or not row:
            continue
        comparable = [neighbor for neighbor in row if 0 <= neighbor < len(labels)]
        if comparable:
            fractions.append(sum(labels[neighbor] == labels[index] for neighbor in comparable) / len(comparable))
    if not fractions:
        return MetricComputation(raw_value=None, evidence=("no valid neighbors were recorded",))
    value = 1.0 - (sum(fractions) / len(fractions))
    return MetricComputation(
        raw_value=value,
        evidence=(f"computed from {len(fractions)} neighborhood rows",),
        artifact_ids=(artifact.artifact_id,),
    )


@metric_registry.register(
    "biology-conservation",
    name="Biological conservation",
    description="Uses an executor-reported conservation diagnostic when available.",
    level=EvaluationLevel.METHOD,
    direction=Direction.HIGHER_IS_BETTER,
)
def biology_conservation(context: MetricContext) -> MetricComputation:
    for artifact in reversed(_artifacts(context)):
        value = _metadata_number(artifact, "biology_conservation")
        if value is not None:
            return MetricComputation(
                raw_value=value,
                evidence=(f"reported by {artifact.artifact_id}",),
                artifact_ids=(artifact.artifact_id,),
            )
    return MetricComputation(raw_value=None, evidence=("no conservation diagnostic was recorded",))


@metric_registry.register(
    "runtime",
    name="Wall-clock runtime",
    description="Elapsed wall-clock duration for the agent run.",
    level=EvaluationLevel.EXECUTION,
    direction=Direction.LOWER_IS_BETTER,
)
def runtime(context: MetricContext) -> MetricComputation:
    return MetricComputation(
        raw_value=context.run.wall_clock_seconds,
        evidence=(f"run wall-clock time: {context.run.wall_clock_seconds:.6f}s",),
    )


def _is_label_sequence(value: Any) -> TypeGuard[list[str | int]]:
    return isinstance(value, list) and all(isinstance(item, (str, int)) for item in value)


def _is_neighbor_sequence(value: Any) -> TypeGuard[list[list[int]]]:
    return isinstance(value, list) and all(
        isinstance(row, list) and all(isinstance(item, int) for item in row) for row in value
    )


__all__ = [
    "artifact_validity",
    "batch_mixing",
    "biology_conservation",
    "cell_retention",
    "embedding_validity",
    "execution_success",
    "runtime",
]
