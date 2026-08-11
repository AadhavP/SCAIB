"""Scanpy-backed scientific executor."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, ClassVar

from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ActionStatus,
    ResourceUsage,
)
from agent_evals.scientific.context import OperationOutput, ScientificContext
from agent_evals.scientific.executor.base import ScientificExecutor
from agent_evals.scientific.executor.registry import scientific_executor_registry
from agent_evals.scientific.operations import (
    annotate,
    batch_correct,
    cluster,
    differential_expression,
    normalize,
    pca,
    qc_filter,
    select_hvg,
)

Operation = Callable[[ScientificContext, dict[str, Any]], OperationOutput]


@scientific_executor_registry.register("scanpy")
class ScanpyExecutor(ScientificExecutor):
    """Dispatch declared operations to real Scanpy implementations."""

    _operations: ClassVar[dict[str, Operation]] = {
        "qc": qc_filter,
        "qc_filter": qc_filter,
        "normalize": normalize,
        "select_hvg": select_hvg,
        "hvg": select_hvg,
        "pca": pca,
        "batch_correct": batch_correct,
        "harmony": batch_correct,
        "cluster": cluster,
        "leiden": cluster,
        "clustering": cluster,
        "neighborhood-graph": cluster,
        "annotate": annotate,
        "annotation": annotate,
        "differential_expression": differential_expression,
        "differential-expression": differential_expression,
        "marker-genes": differential_expression,
    }

    def execute(self, action: ActionIntent, context: ScientificContext) -> ActionExecutionResult:
        """Execute one Scanpy operation and capture timing, artifacts, and errors."""
        started = datetime.now(UTC)
        timer = perf_counter()
        operation = self._operations.get(action.action_id)
        if operation is None:
            error = f"unsupported Scanpy operation '{action.action_id}'"
            context.record_operation(action.action_id, action.parameters, status="failed", error=error, started_at=started)
            return ActionExecutionResult(
                intent_id=action.intent_id, action_id=action.action_id, status=ActionStatus.FAILED,
                error=error, started_at=started, completed_at=datetime.now(UTC),
            )
        columns_before = {str(column) for column in context.adata.obs.columns}
        try:
            output = operation(context, action.parameters)
            if output.adata is not None:
                context.adata = output.adata
            # Attribute newly written observation columns to the agent so scoring
            # can distinguish agent predictions from pre-existing reference labels.
            new_columns = {
                str(column) for column in context.adata.obs.columns
            } - columns_before
            context.record_produced_columns(new_columns)
            for artifact in output.artifacts:
                context.add_artifact(artifact)
            artifact_records = [artifact.to_artifact_record() for artifact in output.artifacts]
            context.record_operation(
                action.action_id, action.parameters, status="succeeded",
                artifact_ids=[artifact.artifact_id for artifact in output.artifacts], started_at=started,
            )
            return ActionExecutionResult(
                intent_id=action.intent_id, action_id=action.action_id, status=ActionStatus.SUCCEEDED,
                outputs=output.outputs, artifacts=artifact_records,
                resource_usage=ResourceUsage(wall_time_seconds=perf_counter() - timer),
                started_at=started, completed_at=datetime.now(UTC),
            )
        except Exception as error:  # operation errors must be represented in the trajectory
            message = f"{type(error).__name__}: {error}"
            context.record_operation(action.action_id, action.parameters, status="failed", error=message, started_at=started)
            return ActionExecutionResult(
                intent_id=action.intent_id, action_id=action.action_id, status=ActionStatus.FAILED,
                error=message, resource_usage=ResourceUsage(wall_time_seconds=perf_counter() - timer),
                started_at=started, completed_at=datetime.now(UTC),
            )


__all__ = ["ScanpyExecutor"]
