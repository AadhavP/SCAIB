"""Registration helpers for the concrete PBMC scientific benchmark suite."""

from __future__ import annotations

from pathlib import Path

from agent_evals.benchmarks.registry import benchmark_spec_registry
from agent_evals.benchmarks.schema import BenchmarkSpecification

SCIENTIFIC_BENCHMARK_IDS = (
    "pbmc-cell-annotation",
    "pbmc-batch-correction",
    "pbmc-differential-expression",
)


def register_scientific_benchmarks(
    root: Path | str = Path("examples/benchmarks"),
) -> list[BenchmarkSpecification]:
    """Load and register the three PBMC benchmark specifications."""
    root_path = Path(root)
    return [
        benchmark_spec_registry.register_file(root_path / f"{benchmark_id}.yaml", replace=True)
        for benchmark_id in SCIENTIFIC_BENCHMARK_IDS
    ]


__all__ = ["SCIENTIFIC_BENCHMARK_IDS", "register_scientific_benchmarks"]
