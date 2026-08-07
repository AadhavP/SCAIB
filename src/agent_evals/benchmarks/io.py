"""Serialization and loading helpers for benchmark specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agent_evals.benchmarks.migrations import schema_migrations
from agent_evals.benchmarks.schema import BenchmarkSpecification


def benchmark_from_dict(
    data: dict[str, Any],
    *,
    migrate: bool = False,
) -> BenchmarkSpecification:
    """Validate a Python dictionary as a benchmark specification.

    When ``migrate`` is true, registered representation-only migrations run
    before Pydantic validation.  Migrations are opt-in so reproducible callers
    can see and control every schema transition.
    """
    payload = schema_migrations.migrate(data) if migrate else data
    return BenchmarkSpecification.model_validate(payload)


def load_benchmark(
    path: Path | str,
    *,
    migrate: bool = False,
) -> BenchmarkSpecification:
    """Load and validate a YAML or JSON benchmark definition from disk.

    The file format is selected by extension.  YAML is also accepted for
    extensionless files so registry discovery can use explicit paths.
    """
    benchmark_path = Path(path)
    with benchmark_path.open(encoding="utf-8") as handle:
        if benchmark_path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"benchmark file '{benchmark_path}' must contain a mapping")
    return benchmark_from_dict(data, migrate=migrate)


def benchmark_to_dict(specification: BenchmarkSpecification) -> dict[str, Any]:
    """Serialize a benchmark to a JSON/YAML-compatible dictionary."""
    return specification.model_dump_serializable()


def dump_benchmark(
    specification: BenchmarkSpecification,
    path: Path | str,
    *,
    format: str | None = None,
) -> None:
    """Write a specification as formatted YAML or JSON.

    ``format`` may be ``"yaml"`` or ``"json"``.  If omitted, the file
    extension determines the format, with YAML as the default.
    """
    output_path = Path(path)
    output_format = (format or output_path.suffix.lstrip(".") or "yaml").lower()
    if output_format in {"yml", "yaml"}:
        content = yaml.safe_dump(
            benchmark_to_dict(specification),
            sort_keys=False,
            default_flow_style=False,
        )
    elif output_format == "json":
        content = json.dumps(benchmark_to_dict(specification), indent=2) + "\n"
    else:
        raise ValueError("benchmark format must be 'yaml' or 'json'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


class BenchmarkSpecificationLoader:
    """Small injectable loader used by registries, CLIs, and API adapters."""

    def load(
        self,
        path: Path | str,
        *,
        migrate: bool = False,
    ) -> BenchmarkSpecification:
        """Load one benchmark definition from a YAML or JSON file."""
        return load_benchmark(path, migrate=migrate)
