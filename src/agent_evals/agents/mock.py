"""Deterministic agent and in-memory ports used for harness tests and demos."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import uuid4

from agent_evals.agents.harness import build_agent_run
from agent_evals.agents.trajectory import (
    AgentConfiguration,
    AgentFailure,
    AgentRun,
    FailureKind,
    RawTraceEvent,
    RunTerminationStatus,
)
from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ActionStatus,
    ArtifactRecord,
    EpisodeStatus,
    Observation,
)
from agent_evals.environment.ports import ExecutionContext
from agent_evals.environment.runtime import ScientificEnvironment


def _raw_event(sequence: int, event_type: str, payload: dict[str, Any]) -> RawTraceEvent:
    """Create a deterministic mock raw event with an observable timestamp."""
    return RawTraceEvent(
        event_id=str(uuid4()),
        source="mock",
        sequence=sequence,
        timestamp=datetime.now(UTC),
        event_type=event_type,
        payload=payload,
    )


class MockObservationBuilder:
    """Expose deterministic dataset observations without biological execution."""

    async def build(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: Any,
    ) -> list[Observation]:
        """Return a stable placeholder observation for deterministic tests."""
        values: dict[str, Any] = {
            "current-anndata": {"cells": 4, "genes": 3, "source": "mock"},
            "batch-labels": ["batch-a", "batch-a", "batch-b", "batch-b"],
            "biological-labels": ["t", "t", "b", "b"],
            "available-tools": ["mock-executor"],
        }
        return [
            Observation(
                observation_id=observation_id,
                value=values.get(observation_id, {"source": "mock"}),
                source="mock-dataset",
            )
            for observation_id in task.observations
        ]


class MockActionExecutor:
    """Produce deterministic, inspectable artifacts for the evaluation demo."""

    _OUTPUTS: ClassVar[dict[str, tuple[str, str, str]]] = {
        "qc": ("qc-table", "table", "parquet"),
        "normalize": ("normalized-anndata", "anndata", "h5ad"),
        "pca": ("uncorrected-embedding", "embedding", "parquet"),
        "harmony": ("corrected-embedding", "embedding", "parquet"),
        "neighborhood-graph": ("corrected-embedding-figure", "figure", "png"),
        "marker-genes": ("marker-table", "table", "parquet"),
        "annotate": ("annotated-anndata", "anndata", "h5ad"),
    }

    async def execute(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> ActionExecutionResult:
        """Return one deterministic artifact for known demo actions."""
        output = self._OUTPUTS.get(intent.action_id)
        artifacts = []
        if output is not None:
            artifact_id, kind, file_format = output
            metadata: dict[str, Any] = {"schema_valid": True}
            if intent.action_id == "qc":
                metadata.update({"cells_before": 4, "cells_after": 3, "columns": ["total_counts", "n_genes_by_counts", "pct_counts_mt"]})
            elif intent.action_id == "normalize":
                metadata.update({"cells_before": 3, "cells_after": 3})
            elif intent.action_id == "pca":
                metadata.update({"n_observations": 3, "n_dimensions": intent.parameters.get("n_components", 50), "finite": True})
            elif intent.action_id == "harmony":
                metadata.update(
                    {
                        "n_observations": 3,
                        "n_dimensions": 50,
                        "finite": True,
                        "batch_labels": ["batch-a", "batch-a", "batch-b"],
                        "neighbors": [[1, 2], [0, 2], [0, 1]],
                        "biology_conservation": 0.9,
                    }
                )
            elif intent.action_id == "annotate":
                metadata.update({"annotation_accuracy": 0.8, "annotation_macro_f1": 0.75})
            artifacts.append(
                ArtifactRecord(
                    artifact_id=artifact_id,
                    kind=kind,
                    format=file_format,
                    validated=True,
                    metadata=metadata,
                )
            )
        return ActionExecutionResult(
            intent_id=intent.intent_id,
            action_id=intent.action_id,
            status=ActionStatus.SUCCEEDED,
            artifacts=artifacts,
        )


class MockAgentAdapter:
    """Deterministic adapter with single, good, and bad workflow policies."""

    adapter_name = "mock"
    adapter_version = "1.0.0"

    def __init__(
        self,
        *,
        executor: MockActionExecutor | None = None,
        observation_builder: MockObservationBuilder | None = None,
    ) -> None:
        self.executor = executor or MockActionExecutor()
        self.observation_builder = observation_builder or MockObservationBuilder()

    async def run(
        self,
        task: TaskSpecification,
        environment: ScientificEnvironment,
        configuration: AgentConfiguration,
    ) -> AgentRun:
        """Run observable typed decisions so evaluation can compare policies."""
        started_at = datetime.now(UTC)
        raw_events: list[RawTraceEvent] = []
        failures: list[AgentFailure] = []
        initial = await environment.reset(
            seed=configuration.seed,
            dataset_id=configuration.metadata.get("dataset_id")
            or (task.datasets[0] if task.datasets else None),
        )
        raw_events.append(
            _raw_event(
                0,
                "observation",
                {"observation_ids": list(initial.state.observations)},
            )
        )

        policy = str(configuration.metadata.get("mock_policy", "single"))
        if policy == "good":
            selected_actions = list(task.allowed_actions)
        elif policy == "bad":
            selected_actions = list(task.allowed_actions)
        else:
            selected_actions = task.allowed_actions[:1]
        if configuration.max_steps is not None:
            selected_actions = selected_actions[: configuration.max_steps]

        action_specs = {action.id: action for action in environment.specification.actions}
        for action_id in selected_actions:
            action_spec = action_specs[action_id]
            parameters = _mock_parameters(action_id, action_spec.parameters, policy)
            intent = ActionIntent(
                action_id=action_id,
                parameters=parameters,
                rationale="Inspect the dataset and apply the first declared quality-control step.",
                metadata={
                    "decision_type": "step_selection",
                    "method": action_id,
                    "method_id": action_id,
                    "implementation": "mock-deterministic",
                    "alternatives_considered": list(task.allowed_actions),
                    "expected_outputs": list(action_spec.expected_outputs),
                },
            )
            raw_events.append(
                _raw_event(
                    1,
                    "action_proposed",
                    {"action_id": action_id, "parameters": intent.parameters, "method": action_id},
                )
            )
            result = await environment.step(intent)
            raw_events.append(
                _raw_event(
                    2,
                    "tool_result",
                    {
                        "action_id": action_id,
                        "status": result.execution.status.value if result.execution else "rejected",
                    },
                )
            )
            if not result.accepted or result.execution is None or result.execution.status != ActionStatus.SUCCEEDED:
                error_message = "; ".join(result.validation.errors)
                if result.accepted and result.execution is not None:
                    error_message = result.execution.error or "mock execution failed"
                failures.append(
                    AgentFailure(
                        kind=FailureKind.INVALID_ACTION if not result.accepted else FailureKind.TOOL_ERROR,
                        message=error_message,
                    )
                )

        status = RunTerminationStatus.FAILED if failures else RunTerminationStatus.COMPLETED
        environment.terminate(
            status=EpisodeStatus.FAILED if failures else EpisodeStatus.COMPLETED,
            reason="mock trajectory complete" if not failures else failures[0].message,
        )
        final = environment.episode.snapshot() if environment.episode is not None else initial
        return build_agent_run(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            configuration=configuration,
            task=task,
            snapshot=final,
            raw_events=raw_events,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            termination_status=status,
            termination_reason="mock trajectory complete" if not failures else failures[0].message,
            failures=failures,
        )


MockAgent = MockAgentAdapter
"""Short alias used by examples and tests."""


__all__ = [
    "MockActionExecutor",
    "MockAgent",
    "MockAgentAdapter",
    "MockObservationBuilder",
]


def _mock_parameters(action_id: str, parameters: list[Any], policy: str) -> dict[str, Any]:
    """Choose explicit observable parameters; the bad policy violates bounds."""
    defaults: dict[str, dict[str, Any]] = {
        "qc": {"min_genes": 200, "max_mito_fraction": 0.2},
        "normalize": {"target_sum": 10000},
        "pca": {"n_components": 50},
        "harmony": {"batch_key": "batch"},
        "neighborhood-graph": {"n_neighbors": 15},
        "marker-genes": {"group_key": "cell_type"},
        "annotate": {"label_vocabulary": ["T", "B", "NK"]},
    }
    selected = dict(defaults.get(action_id, {}))
    if policy == "bad":
        for parameter in parameters:
            if parameter.type in {"integer", "number"} and parameter.minimum is not None:
                selected[parameter.name] = parameter.minimum - 1
                break
    return {name: value for name, value in selected.items() if any(parameter.name == name for parameter in parameters)}
