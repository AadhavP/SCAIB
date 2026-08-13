"""How a benchmark named on the command line becomes a file on disk.

One function rather than a copy per command module: ``run``, ``evaluate``,
``validate-benchmark``, and ``env inspect`` all accept the same ``--benchmark``
argument, and two of them resolving it differently would mean the command that
validates a benchmark could validate a different file from the one the command
that runs it loads.
"""

from __future__ import annotations

from pathlib import Path

#: Where a bare benchmark id is looked up when it is not a path that exists.
EXAMPLE_BENCHMARK_DIR = Path("examples/benchmarks")


def resolve_benchmark_path(reference: str) -> Path:
    """Return the YAML path for a benchmark path or bare registered id."""
    path = Path(reference)
    if path.exists():
        return path
    return EXAMPLE_BENCHMARK_DIR / f"{reference}.yaml"


__all__ = ["EXAMPLE_BENCHMARK_DIR", "resolve_benchmark_path"]
