"""Deterministic evaluator for trajectories, executions, artifacts, and metrics."""

from __future__ import annotations

from typing import Any

from agent_evals.agents.trajectory import AgentRun, ScientificDecision
from agent_evals.benchmarks.schema import (
    BenchmarkSpecification,
    Direction,
    MetricSpecification,
)
from agent_evals.core.exceptions import RegistryError
from agent_evals.environment.models import ActionStatus, EpisodeSnapshot
from agent_evals.evaluators.models import (
    DecisionEvaluation,
    EvaluationLevel,
    EvaluationReport,
    EvaluationSummary,
    ExecutionEvaluation,
    MetricResult,
    MetricStatus,
    TaskInstance,
)
from agent_evals.evaluators.registry import (
    MetricContext,
    MetricRegistry,
    metric_registry,
)
from agent_evals.evaluators.task import build_task_instance


class EvaluationEngine:
    """Evaluate one completed or partial run without calculating global reward."""

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or metric_registry

    def evaluate(
        self,
        specification: BenchmarkSpecification,
        run: AgentRun,
        *,
        task_id: str | None = None,
    ) -> EvaluationReport:
        """Produce a structured report from one run and its final environment state."""
        resolved_task_id = task_id or run.task_id
        task_instance = build_task_instance(
            specification,
            task_id=resolved_task_id,
            initial_environment_state=None,
        )
        snapshot = run.final_environment_state
        decision_evaluations = self._evaluate_decisions(task_instance, run.trajectory.decisions.decisions)
        execution_results = [
            ExecutionEvaluation(
                step=record.step,
                action_id=record.intent.action_id,
                status=record.result.status,
                succeeded=record.result.status == ActionStatus.SUCCEEDED,
                artifact_ids=[artifact.artifact_id for artifact in record.result.artifacts],
                resource_usage=record.result.resource_usage,
                error=record.result.error,
            )
            for record in snapshot.state.actions
        ]
        metric_results = [
            self._evaluate_metric(
                metric_id=metric.id,
                metric_spec=metric,
                task_instance=task_instance,
                run=run,
            )
            for metric in task_instance.evaluation_metrics
        ]
        summary = _summary(
            decision_evaluations,
            execution_results,
            metric_results,
            task_instance,
            snapshot,
        )
        return EvaluationReport(
            benchmark_id=specification.metadata.id,
            benchmark_version=specification.metadata.version,
            task_id=resolved_task_id,
            agent_id=run.agent_id,
            run_id=run.run_id,
            task_instance=task_instance,
            decisions=run.trajectory.decisions,
            decision_evaluations=decision_evaluations,
            execution_results=execution_results,
            metric_results=metric_results,
            artifacts=list(snapshot.state.artifacts.values()),
            failures=run.failures,
            summary=summary,
            metadata={"evaluation_levels": task_instance.evaluation.levels},
        )

    def _evaluate_metric(
        self,
        *,
        metric_id: str,
        metric_spec: MetricSpecification,
        task_instance: TaskInstance,
        run: AgentRun,
    ) -> MetricResult:
        """Compute one metric while isolating missing dependencies and failures."""
        try:
            registered = self.registry.get(metric_id)
        except RegistryError as error:
            return MetricResult(
                metric_id=metric_id,
                metric_name=metric_spec.name,
                level=_level_for_metric(metric_id),
                direction=metric_spec.direction,
                status=MetricStatus.UNAVAILABLE,
                error=str(error),
            )
        available_artifacts = set(run.final_environment_state.state.artifacts)
        missing = sorted(set(registered.required_artifacts) - available_artifacts)
        if missing:
            return MetricResult(
                metric_id=metric_id,
                metric_name=metric_spec.name,
                level=registered.level,
                direction=metric_spec.direction,
                status=MetricStatus.UNAVAILABLE,
                evidence=[f"missing artifact dependency: {', '.join(missing)}"],
                error="missing_artifact",
            )
        try:
            computation = registered.compute(
                MetricContext(
                    task_instance=task_instance,
                    run=run,
                    snapshot=run.final_environment_state,
                    specification=metric_spec,
                )
            )
        except Exception as error:  # pragma: no cover - defensive plugin boundary
            return MetricResult(
                metric_id=metric_id,
                metric_name=metric_spec.name,
                level=registered.level,
                direction=metric_spec.direction,
                status=MetricStatus.ERROR,
                error=str(error),
            )
        normalized = _normalize_score(computation.raw_value, metric_spec)
        status = MetricStatus.SUCCEEDED if computation.raw_value is not None else MetricStatus.UNAVAILABLE
        return MetricResult(
            metric_id=metric_id,
            metric_name=metric_spec.name,
            level=registered.level,
            direction=metric_spec.direction,
            raw_value=computation.raw_value,
            value=computation.raw_value,
            normalized_score=normalized,
            status=status,
            evidence=list(computation.evidence),
            artifact_ids=list(computation.artifact_ids),
            error=None if status == MetricStatus.SUCCEEDED else "metric_value_unavailable",
            metadata=computation.metadata or {},
        )

    @staticmethod
    def _evaluate_decisions(
        task_instance: TaskInstance,
        decisions: list[ScientificDecision],
    ) -> list[DecisionEvaluation]:
        """Evaluate observable action, method, and parameter selections."""
        allowed_actions = {action.id: action for action in task_instance.allowed_actions}
        results: list[DecisionEvaluation] = []
        for decision in decisions:
            level = _level_for_decision(decision)
            evidence: list[str] = []
            valid = decision.action_category in allowed_actions
            if valid:
                evidence.append(f"action '{decision.action_category}' is allowed")
            else:
                evidence.append(f"action '{decision.action_category}' is not allowed")
            if decision.parameter_choice is not None and valid:
                action = allowed_actions[decision.action_category]
                parameter = next(
                    (
                        candidate
                        for candidate in action.parameters
                        if candidate.name == decision.parameter_choice.name
                    ),
                    None,
                )
                parameter_valid = parameter is not None
                if parameter is not None and isinstance(decision.parameter_choice.value, (int, float)):
                    parameter_valid = (
                        (parameter.minimum is None or decision.parameter_choice.value >= parameter.minimum)
                        and (parameter.maximum is None or decision.parameter_choice.value <= parameter.maximum)
                        and (not parameter.choices or decision.parameter_choice.value in parameter.choices)
                    )
                valid = valid and parameter_valid
                evidence.append(
                    f"parameter '{decision.parameter_choice.name}' is "
                    f"{'valid' if parameter_valid else 'invalid'}"
                )
            score = 1.0 if valid else 0.0
            results.append(
                DecisionEvaluation(
                    decision_id=decision.decision_id,
                    step_id=decision.step_id,
                    decision_type=decision.decision_type,
                    level=level,
                    valid=valid,
                    score=score,
                    selected_value=decision.selected_value,
                    evidence=evidence,
                    artifact_ids=decision.output_artifacts,
                )
            )
        return results


def _level_for_decision(decision: ScientificDecision) -> EvaluationLevel:
    if decision.decision_type == "method_selection":
        return EvaluationLevel.METHOD
    if decision.decision_type == "parameter_selection":
        return EvaluationLevel.PARAMETER
    if decision.decision_type == "execution_configuration":
        return EvaluationLevel.EXECUTION
    return EvaluationLevel.DECISION


def _level_for_metric(metric_id: str) -> EvaluationLevel:
    if metric_id in {"execution-success", "runtime"}:
        return EvaluationLevel.EXECUTION
    if metric_id in {"artifact-validity", "cell-retention", "embedding-validity"}:
        return EvaluationLevel.ARTIFACT
    return EvaluationLevel.METHOD


def _normalize_score(value: Any, specification: MetricSpecification) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    expected = specification.expected_range
    if expected is not None and expected.minimum is not None and expected.maximum is not None:
        span = expected.maximum - expected.minimum
        if span == 0:
            return 1.0 if numeric == expected.minimum else 0.0
        ratio = (numeric - expected.minimum) / span
        return max(0.0, min(1.0, 1.0 - ratio if specification.direction == Direction.LOWER_IS_BETTER else ratio))
    if specification.direction == Direction.LOWER_IS_BETTER:
        return 1.0 / (1.0 + max(0.0, numeric))
    return max(0.0, min(1.0, numeric))


def _summary(
    decisions: list[DecisionEvaluation],
    executions: list[ExecutionEvaluation],
    metrics: list[MetricResult],
        task_instance: TaskInstance,
        snapshot: EpisodeSnapshot,
    ) -> EvaluationSummary:
    counts: dict[str, int] = {}
    by_level: dict[str, list[str]] = {}
    for result in metrics:
        counts[result.status.value] = counts.get(result.status.value, 0) + 1
        by_level.setdefault(result.level.value, []).append(result.metric_id)
    for decision in decisions:
        by_level.setdefault(decision.level.value, []).append(decision.decision_id)
    expected_ids = {artifact.id for artifact in task_instance.expected_artifacts}
    valid_artifacts = sum(
        artifact.validated
        for artifact_id, artifact in snapshot.state.artifacts.items()
        if artifact_id in expected_ids
    )
    return EvaluationSummary(
        metric_counts=counts,
        successful_actions=sum(item.succeeded for item in executions),
        total_actions=len(executions),
        valid_artifacts=valid_artifacts,
        total_expected_artifacts=len(task_instance.expected_artifacts),
        by_level=by_level,
    )


__all__ = ["EvaluationEngine"]
