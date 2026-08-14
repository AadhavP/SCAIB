"""Tests for the declarative benchmark specification language."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_evals.benchmarks.agent_package import build_agent_task_package
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.registry import BenchmarkSpecificationRegistry
from agent_evals.benchmarks.schema import (
    BenchmarkMetadata,
    BenchmarkSpecification,
    TaskSpecification,
)
from agent_evals.environment.models import ActionIntent
from agent_evals.environment.ports import DeclarativeActionValidator

EXAMPLES = Path(__file__).parents[1] / "examples" / "benchmarks"


def test_realistic_examples_load_and_registry_search() -> None:
    """All published examples should be valid and discoverable by metadata."""
    registry = BenchmarkSpecificationRegistry()
    registry.discover(EXAMPLES)

    assert registry.list_ids() == [
        "pbmc-batch-correction",
        "pbmc-cell-annotation",
        "pbmc-cell-annotation-free",
        "pbmc-differential-expression",
    ]
    assert registry.search(tags={"single-cell"})
    assert registry.get("pbmc-cell-annotation").tasks[0].metrics
    assert registry.validate() == []


def test_unknown_reference_is_rejected() -> None:
    """A task cannot silently refer to an undeclared dataset."""
    with pytest.raises(ValidationError, match="unknown dataset"):
        BenchmarkSpecification(
            metadata=BenchmarkMetadata(
                id="invalid",
                title="Invalid benchmark",
                description="Used to test validation.",
                license="MIT",
            ),
            tasks=[
                TaskSpecification(
                    id="task",
                    name="Task",
                    objective="Test references.",
                    description="This task has a missing dataset.",
                    datasets=["missing-dataset"],
                )
            ],
        )


def test_circular_task_dependencies_are_rejected() -> None:
    """Task graphs must remain acyclic for deterministic orchestration."""
    with pytest.raises(ValidationError, match="circular task dependency"):
        BenchmarkSpecification(
            metadata=BenchmarkMetadata(
                id="cyclic",
                title="Cyclic benchmark",
                description="Used to test task graph validation.",
                license="MIT",
            ),
            tasks=[
                TaskSpecification(
                    id="a",
                    name="A",
                    objective="First step.",
                    description="Depends on B.",
                    depends_on=["b"],
                ),
                TaskSpecification(
                    id="b",
                    name="B",
                    objective="Second step.",
                    description="Depends on A.",
                    depends_on=["a"],
                ),
            ],
        )


def test_json_semantics_round_trip(tmp_path: Path) -> None:
    """Serialization should preserve model semantics across file formats."""
    from agent_evals.benchmarks.io import dump_benchmark

    specification = load_benchmark(EXAMPLES / "pbmc-cell-annotation.yaml")
    output = tmp_path / "annotation.json"
    dump_benchmark(specification, output)
    assert load_benchmark(output).model_dump() == specification.model_dump()


def test_agent_task_package_contains_executable_scientific_contract() -> None:
    """The opening brief must remove contract guessing from the agent loop."""
    specification = load_benchmark(EXAMPLES / "pbmc-cell-annotation.yaml")
    package = build_agent_task_package(specification, specification.tasks[0])

    assert package["task"]["objective"]
    qc = next(action for action in package["actions"] if action["id"] == "qc")
    assert {parameter["name"] for parameter in qc["parameters"]} >= {
        "method",
        "min_genes",
        "max_mito_fraction",
        "min_cells",
    }
    assert "adaptive_quantile" in next(
        parameter["choices"]
        for parameter in qc["parameters"]
        if parameter["name"] == "method"
    )
    assert package["artifacts"]
    assert qc["execution_owner"] == "scaib_environment"
    assert qc["verification"] == "typed_result_and_artifacts"
    assert package["interaction_protocol"]["failure_recovery"]


def test_declared_defaults_are_materialized_at_the_environment_boundary() -> None:
    specification = load_benchmark(EXAMPLES / "pbmc-cell-annotation.yaml")
    intent = DeclarativeActionValidator.apply_defaults(
        ActionIntent(action_id="qc"),
        specification,
    )

    assert intent.parameters["method"] == "fixed_threshold"
    assert intent.parameters["min_genes"] == 200
    assert intent.parameters["max_mito_fraction"] == 0.2


def test_parameter_validation_rejects_provider_coercion_for_scalar_types() -> None:
    """A stringified threshold must not silently become a scientific number."""
    specification = load_benchmark(EXAMPLES / "pbmc-cell-annotation.yaml")
    action = next(item for item in specification.actions if item.id == "qc")

    errors = DeclarativeActionValidator._validate_parameters(
        action,
        ActionIntent(
            action_id="qc",
            parameters={
                "method": "fixed_threshold",
                "min_genes": "200",
                "max_mito_fraction": 0.2,
                "min_cells": 1,
            },
        ),
    )

    assert any("min_genes" in error and "integer" in error for error in errors)


def test_parameter_validation_checks_nested_collection_types() -> None:
    """List-shaped scientific parameters must not accept a scalar by accident."""
    specification = load_benchmark(EXAMPLES / "pbmc-differential-expression.yaml")
    action = next(item for item in specification.actions if item.id == "differential-expression")

    errors = DeclarativeActionValidator._validate_parameters(
        action,
        ActionIntent(
            action_id="differential-expression",
            parameters={
                "method": "wilcoxon",
                "group_key": "clusters",
                "covariates": "batch",
            },
        ),
    )

    assert any("covariates" in error and "list[string]" in error for error in errors)

