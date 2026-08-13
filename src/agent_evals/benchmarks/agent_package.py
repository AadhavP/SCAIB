"""Build the public task package supplied to a scientific agent.

The YAML specification is the benchmark's contract, but it is not a good agent
brief on its own.  In particular, handing an agent only ``allowed_actions``
forces it to guess parameter names, inputs, outputs, and what counts as a
successful hand-off.  This module creates one explicit, serializable package
from the same validated specification used by the executor.

Only public task information is included.  Dataset metadata is intentionally
limited to provenance and declared shape; arbitrary dataset metadata and
reference observations never cross this boundary.
"""

from __future__ import annotations

from typing import Any

from agent_evals.benchmarks.schema import (
    ActionKind,
    BenchmarkSpecification,
    DatasetSpecification,
    TaskSpecification,
)


def build_agent_task_package(
    specification: BenchmarkSpecification,
    task: TaskSpecification,
) -> dict[str, Any]:
    """Return the public scientific brief for one task.

    The package is deliberately made of plain JSON-compatible values so it can
    be passed unchanged to provider runtimes, persisted in trajectories, or
    rendered in a UI.
    """
    datasets = {dataset.id: dataset for dataset in specification.datasets}
    observations = {observation.id: observation for observation in specification.observations}
    actions = {action.id: action for action in specification.actions}
    metrics = {metric.id: metric for metric in specification.metrics}
    environments = {environment.id: environment for environment in specification.environments}
    artifacts = {artifact.id: artifact for artifact in specification.artifacts}

    selected_datasets = [
        _dataset_contract(datasets[dataset_id])
        for dataset_id in task.datasets
        if dataset_id in datasets
    ]
    selected_observations = [
        _observation_contract(observations[observation_id])
        for observation_id in task.observations
        if observation_id in observations
    ]
    selected_actions = [
        _action_contract(actions[action_id])
        for action_id in task.allowed_actions
        if action_id in actions
    ]

    package: dict[str, Any] = {
        "benchmark": {
            "id": specification.metadata.id,
            "title": specification.metadata.title,
            "version": specification.metadata.version,
            "description": specification.metadata.description,
            "domains": list(specification.metadata.domains),
            "tags": list(specification.metadata.tags),
        },
        "task": {
            "id": task.id,
            "name": task.name,
            "objective": task.objective,
            "end_goal": _end_goal(task),
            "description": task.description,
            "datasets": list(task.datasets),
            "observations": list(task.observations),
            "allowed_actions": list(task.allowed_actions),
            "artifacts": list(task.artifacts),
            "required_artifacts": sorted(specification.required_task_artifacts(task)),
            "metrics": list(task.metrics),
        },
        "datasets": selected_datasets,
        "observations": selected_observations,
        "actions": selected_actions,
        "artifacts": [
            _artifact_contract(artifacts[artifact_id])
            for artifact_id in task.artifacts
            if artifact_id in artifacts
        ],
        "workflow": [
            {
                "id": stage.id,
                "name": stage.name,
                "description": stage.description,
                "allowed_actions": list(stage.allowed_actions),
                "depends_on": list(stage.depends_on),
                "required": stage.required,
                "metrics": list(stage.metrics),
            }
            for stage in task.workflow
        ],
        "success_criteria": [
            {
                "name": condition.name,
                "description": condition.description,
                "condition": condition.condition,
            }
            for condition in task.termination
        ],
        "stopping_criteria": [
            {
                "name": condition.name,
                "description": condition.description,
                "condition": condition.condition,
                "terminal": condition.terminal,
            }
            for condition in task.termination
        ],
        "evaluation": {
            "levels": list(task.evaluation.levels),
            "metrics": [_metric_contract(metrics[metric_id]) for metric_id in task.evaluation.metrics if metric_id in metrics],
            "decision_contracts": {
                category: profile.model_dump(mode="json")
                for category, profile in specification.decision_evaluation.items()
                if category in _task_decision_categories(task)
            },
        },
        "constraints": _constraints(task, specification),
        "environment": (
            environments[task.environment].model_dump(mode="json")
            if task.environment in environments
            else None
        ),
        "interaction_protocol": {
            "turn": "The agent receives the current state and returns one action per turn.",
            "feedback": "After every accepted action, inspect execution status, outputs, state changes, and pipeline history before choosing the next action.",
            "failure_recovery": "A failed action is retained in history; diagnose the reported error and retry with a changed method or parameter when appropriate.",
            "replanning": "The agent may revise its plan as new evidence arrives; do not treat the initial plan as binding.",
            "termination": "Return finish, done, terminate, or final_submission only after the end goal is met and the required artifact contract is satisfied.",
            "evidence_boundary": "Reference labels and reference-derived scores are evaluator-only. Use visible data summaries and execution evidence to make decisions.",
            "scientific_judgment": "Diagnose unexpected results, weak evidence, and confounding factors from the observations. The benchmark does not provide an exhaustive failure-mode checklist; revise the workflow when the evidence warrants it.",
        },
    }
    return package


def _end_goal(task: TaskSpecification) -> str:
    """State the deliverable without prescribing a scientific method."""
    return task.end_goal or (
        f"Complete '{task.name}' and produce the strongest defensible result "
        "supported by the available observations and required artifacts."
    )


def _dataset_contract(dataset: DatasetSpecification) -> dict[str, Any]:
    """Expose reproducibility metadata without arbitrary hidden annotations."""
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "source": dataset.source,
        "source_url": dataset.source_url,
        "organism": dataset.organism,
        "modality": dataset.modality,
        "format": dataset.format,
        "expected_observations": dict(dataset.expected_observations),
        "citation": list(dataset.citation),
        "license": dataset.license,
    }


def _observation_contract(observation: Any) -> dict[str, Any]:
    return {
        "id": observation.id,
        "name": observation.name,
        "description": observation.description,
        "type": observation.type,
        "source": observation.source,
        "required": observation.required,
        "schema": dict(observation.schema_definition),
    }


def _action_contract(action: Any) -> dict[str, Any]:
    return {
        "id": action.id,
        "name": action.name,
        "purpose": action.purpose,
        "kind": action.kind.value if isinstance(action.kind, ActionKind) else action.kind,
        "parameters": [
            {
                "name": parameter.name,
                "description": parameter.description,
                "type": parameter.type,
                "required": parameter.required,
                "default": parameter.default,
                "choices": list(parameter.choices),
                "minimum": parameter.minimum,
                "maximum": parameter.maximum,
                "constraints": dict(parameter.constraints),
            }
            for parameter in action.parameters
        ],
        "required_inputs": list(action.required_inputs),
        "expected_outputs": list(action.expected_outputs),
        "estimated_cost": (
            action.estimated_cost.model_dump(mode="json")
            if action.estimated_cost is not None
            else None
        ),
    }


def _metric_contract(metric: Any) -> dict[str, Any]:
    return {
        "id": metric.id,
        "name": metric.name,
        "description": metric.description,
        "direction": metric.direction.value,
        "expected_range": (
            metric.expected_range.model_dump(mode="json")
            if metric.expected_range is not None
            else None
        ),
        "unit": metric.unit,
        "category": metric.category,
    }


def _artifact_contract(artifact: Any) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "name": artifact.name,
        "description": artifact.description,
        "kind": artifact.kind.value,
        "format": artifact.format,
        "required": artifact.required,
        "produced_by": list(artifact.produced_by),
        "validation": [
            {
                "name": rule.name,
                "description": rule.description,
                "rule": rule.rule,
            }
            for rule in artifact.validation
        ],
    }


def _constraints(task: TaskSpecification, specification: BenchmarkSpecification) -> dict[str, Any]:
    constraints = task.constraints or specification.constraints
    return constraints.model_dump(mode="json")


def _task_decision_categories(task: TaskSpecification) -> set[str]:
    """Name decision categories relevant to the actions in this task."""
    categories: set[str] = set()
    for action_id in task.allowed_actions:
        if action_id in {"qc", "qc_filter"}:
            categories.add("qc_strategy")
        elif action_id == "normalize":
            categories.add("normalization")
        elif action_id in {"harmony", "batch_correct"}:
            categories.add("integration")
        elif action_id in {"cluster", "clustering", "leiden", "louvain"}:
            categories.add("clustering")
        elif action_id in {"annotate", "annotation"}:
            categories.add("annotation")
        elif action_id in {"marker-genes", "differential-expression", "differential_expression"}:
            categories.add("differential_expression")
    return categories


__all__ = ["build_agent_task_package"]
