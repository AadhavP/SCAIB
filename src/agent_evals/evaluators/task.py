"""Resolution of declarative benchmark tasks into executable instances."""

from __future__ import annotations

from agent_evals.benchmarks.schema import BenchmarkSpecification
from agent_evals.environment.models import EpisodeSnapshot
from agent_evals.evaluators.models import TaskInstance


def build_task_instance(
    specification: BenchmarkSpecification,
    *,
    task_id: str,
    initial_environment_state: EpisodeSnapshot | None = None,
) -> TaskInstance:
    """Resolve all task references without executing a scientific operation."""
    task = next((candidate for candidate in specification.tasks if candidate.id == task_id), None)
    if task is None:
        raise ValueError(f"unknown task '{task_id}'")
    dataset_id = task.datasets[0] if task.datasets else None
    dataset = next(
        (candidate for candidate in specification.datasets if candidate.id == dataset_id),
        None,
    )
    actions = [
        action for action in specification.actions if action.id in task.allowed_actions
    ]
    metrics_ids = task.evaluation.metrics or task.metrics
    metrics = [metric for metric in specification.metrics if metric.id in metrics_ids]
    artifacts = [
        artifact for artifact in specification.artifacts if artifact.id in task.artifacts
    ]
    return TaskInstance(
        task_id=task.id,
        benchmark_id=specification.metadata.id,
        benchmark_version=specification.metadata.version,
        dataset=dataset,
        initial_environment_state=initial_environment_state,
        scientific_objective=task.objective,
        allowed_actions=actions,
        workflow=task.workflow,
        evaluation=task.evaluation,
        evaluation_metrics=metrics,
        expected_artifacts=artifacts,
        resource_constraints=task.constraints or specification.constraints,
    )


__all__ = ["build_task_instance"]
