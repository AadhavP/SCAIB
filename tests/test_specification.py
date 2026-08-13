"""Tests for the declarative benchmark specification language."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.registry import BenchmarkSpecificationRegistry
from agent_evals.benchmarks.schema import (
    BenchmarkMetadata,
    BenchmarkSpecification,
    EnvironmentSpecification,
    TaskSpecification,
)

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


def test_environment_spec_publishes_the_agent_visible_runtime_contract() -> None:
    """A publishable free-execution benchmark must document more than a backend."""
    specification = load_benchmark(EXAMPLES / "pbmc-cell-annotation-free.yaml")
    environment = specification.environments[0]

    assert environment.runtime.python == "3.12"
    assert "scanpy" in environment.packages.python
    assert environment.working_directory == "/workspace"
    assert "/workspace/results" in environment.writable_paths
    assert environment.data[0].id == "pbmc_input"
    assert environment.data[0].read_only is True
    assert environment.deliverables[0].artifact_id == "cell-labels"
    assert environment.hidden_reference_boundaries == [
        "reference cell-type labels",
        "reference marker rankings",
    ]
    assert any("Do not prescribe" in note for note in environment.agent_instructions)


def test_publishable_mode_rejects_incomplete_benchmark_release_contract() -> None:
    """Release validation should catch specs that load but are not publishable."""
    incomplete = BenchmarkSpecification(
        metadata=BenchmarkMetadata(
            id="paper-target",
            title="Paper target",
            description="A syntactically valid but under-specified benchmark.",
            license="MIT",
        ),
        tasks=[
            TaskSpecification(
                id="task",
                name="Task",
                objective="Produce an artifact.",
                description="No hard cutoff or score profile was declared.",
            )
        ],
    )

    problems = incomplete.validate_publishable()

    assert "declares no metric_groups" in problems
    assert "cutoff.max_steps is required" in problems
    assert "cutoff.max_wall_time_seconds is required" in problems


def test_publishable_mode_accepts_current_pbmc_specs() -> None:
    """The shipped PBMC benchmarks should satisfy the paper-artifact contract."""
    for path in sorted(EXAMPLES.glob("pbmc-*.yaml")):
        assert load_benchmark(path).validate_publishable() == [], path.name


def test_environment_validation_rejects_delivery_paths_outside_writable_roots() -> None:
    with pytest.raises(ValidationError, match="deliverable 'labels' path"):
        EnvironmentSpecification(
            id="local-python",
            name="Local Python",
            description="A local runtime.",
            packages={"python": ["scanpy"]},
            writable_paths=["/workspace/results"],
            deliverables=[
                {
                    "artifact_id": "labels",
                    "path": "/tmp/labels.csv",
                    "artifact_type": "table",
                    "format": "csv",
                }
            ],
        )

