"""Structured models for scientific task evaluation reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.trajectory import (
    AgentFailure,
    DecisionCascade,
)
from agent_evals.benchmarks.schema import (
    ActionSpecification,
    ArtifactSpecification,
    ConstraintSpecification,
    DatasetSpecification,
    Direction,
    EvaluationConfiguration,
    MetricSpecification,
    WorkflowStage,
)
from agent_evals.environment.models import (
    ActionStatus,
    ArtifactRecord,
    EpisodeSnapshot,
    ResourceUsage,
)


class EvaluationRuntimeModel(BaseModel):
    """Strict base model for persisted evaluation artifacts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvaluationLevel(StrEnum):
    """Granularity at which an evaluation result is reported."""

    DECISION = "decision"
    METHOD = "method"
    PARAMETER = "parameter"
    EXECUTION = "execution"
    ARTIFACT = "artifact"


class MetricStatus(StrEnum):
    """Lifecycle status of one independently computed metric."""

    SUCCEEDED = "succeeded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class TaskInstance(EvaluationRuntimeModel):
    """Resolved executable evaluation problem derived from a benchmark spec."""

    task_id: str
    benchmark_id: str
    benchmark_version: str
    dataset: DatasetSpecification | None = None
    initial_environment_state: EpisodeSnapshot | None = None
    scientific_objective: str
    allowed_actions: list[ActionSpecification] = Field(default_factory=list)
    workflow: list[WorkflowStage] = Field(default_factory=list)
    evaluation: EvaluationConfiguration
    evaluation_metrics: list[MetricSpecification] = Field(default_factory=list)
    expected_artifacts: list[ArtifactSpecification] = Field(default_factory=list)
    resource_constraints: ConstraintSpecification


class DecisionEvaluation(EvaluationRuntimeModel):
    """Deterministic evaluation of one observable decision selection."""

    decision_id: str
    step_id: str
    decision_type: str
    level: EvaluationLevel
    valid: bool
    score: float = Field(ge=0, le=1)
    selected_value: Any | None = None
    evidence: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class ExecutionEvaluation(EvaluationRuntimeModel):
    """Evaluation of action execution independent of scientific result quality."""

    step: int
    action_id: str
    status: ActionStatus
    succeeded: bool
    artifact_ids: list[str] = Field(default_factory=list)
    resource_usage: ResourceUsage
    error: str | None = None


class MetricResult(EvaluationRuntimeModel):
    """Structured result retaining raw values, evidence, and normalization."""

    metric_id: str
    metric_name: str
    level: EvaluationLevel
    direction: Direction
    raw_value: Any | None = None
    value: Any | None = None
    normalized_score: float | None = Field(default=None, ge=0, le=1)
    status: MetricStatus
    evidence: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(EvaluationRuntimeModel):
    """Non-aggregated report summary grouped by evaluation level."""

    metric_counts: dict[str, int] = Field(default_factory=dict)
    successful_actions: int = 0
    total_actions: int = 0
    valid_artifacts: int = 0
    total_expected_artifacts: int = 0
    by_level: dict[str, list[str]] = Field(default_factory=dict)


class EvaluationReport(EvaluationRuntimeModel):
    """Complete structured report for one trajectory and task instance."""

    report_version: str = "1.0.0"
    benchmark_id: str
    benchmark_version: str
    task_id: str
    agent_id: str
    run_id: str
    task_instance: TaskInstance
    decisions: DecisionCascade
    decision_evaluations: list[DecisionEvaluation] = Field(default_factory=list)
    execution_results: list[ExecutionEvaluation] = Field(default_factory=list)
    metric_results: list[MetricResult] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    failures: list[AgentFailure] = Field(default_factory=list)
    summary: EvaluationSummary
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the report as canonical JSON."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, payload: str) -> EvaluationReport:
        """Restore a report from canonical JSON."""
        return cls.model_validate_json(payload)


__all__ = [
    "DecisionEvaluation",
    "EvaluationLevel",
    "EvaluationReport",
    "EvaluationSummary",
    "ExecutionEvaluation",
    "MetricResult",
    "MetricStatus",
    "TaskInstance",
]
