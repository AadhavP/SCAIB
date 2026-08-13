"""Tests for the declarative benchmark specification language."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.registry import BenchmarkSpecificationRegistry
from agent_evals.benchmarks.schema import (
    BenchmarkMetadata,
    BenchmarkSpecification,
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

