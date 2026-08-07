"""Registries for executable adapters and declarative benchmark definitions."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from pathlib import Path

from agent_evals.benchmarks.base import BaseBenchmark
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.schema import BenchmarkSpecification
from agent_evals.core.exceptions import RegistryError


class BenchmarkRegistry:
    """Registry to register and resolve benchmark classes by ID."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseBenchmark]] = {}

    def register(
        self, benchmark_id: str
    ) -> Callable[[type[BaseBenchmark]], type[BaseBenchmark]]:
        """Decorator to register a benchmark class under a unique ID."""

        def decorator(cls: type[BaseBenchmark]) -> type[BaseBenchmark]:
            if benchmark_id in self._registry:
                raise RegistryError(
                    f"Benchmark with ID '{benchmark_id}' is already registered."
                )
            self._registry[benchmark_id] = cls
            return cls

        return decorator

    def get(self, benchmark_id: str) -> type[BaseBenchmark]:
        """Retrieve registered benchmark class by ID."""
        if benchmark_id not in self._registry:
            raise RegistryError(
                f"Benchmark '{benchmark_id}' not found in registry. "
                f"Available: {self.list_ids()}"
            )
        return self._registry[benchmark_id]

    def list_ids(self) -> list[str]:
        """Return list of all registered benchmark IDs."""
        return sorted(self._registry.keys())


# Global benchmark registry singleton instance
benchmark_registry = BenchmarkRegistry()


class BenchmarkSpecificationRegistry:
    """Discover, validate, and resolve declarative benchmark specifications.

    Multiple versions of the same benchmark ID may coexist.  ``get`` resolves
    the highest registered version when no version is requested, while search
    and listing support explicit version filters for reproducible consumers.
    This registry stores validated models only; it never executes a benchmark.
    """

    def __init__(self) -> None:
        self._specifications: dict[tuple[str, str], BenchmarkSpecification] = {}

    def register(
        self,
        specification: BenchmarkSpecification,
        *,
        replace: bool = False,
    ) -> BenchmarkSpecification:
        """Register one validated specification and reject duplicate keys."""
        key = (specification.metadata.id, specification.metadata.version)
        if key in self._specifications and not replace:
            raise RegistryError(
                f"Benchmark specification '{key[0]}@{key[1]}' is already registered."
            )
        self._specifications[key] = specification
        return specification

    def register_file(
        self,
        path: Path | str,
        *,
        replace: bool = False,
    ) -> BenchmarkSpecification:
        """Load, validate, and register a YAML or JSON specification file."""
        return self.register(load_benchmark(path), replace=replace)

    def discover(
        self,
        directory: Path | str,
        *,
        recursive: bool = True,
        replace: bool = False,
    ) -> builtins.list[BenchmarkSpecification]:
        """Register all YAML and JSON definitions under a directory."""
        root = Path(directory)
        paths = sorted(
            path
            for pattern in ("*.yaml", "*.yml", "*.json")
            for path in (root.rglob(pattern) if recursive else root.glob(pattern))
        )
        return [self.register_file(path, replace=replace) for path in paths]

    def get(
        self,
        benchmark_id: str,
        *,
        version: str | None = None,
    ) -> BenchmarkSpecification:
        """Resolve a benchmark by ID and optionally exact version."""
        matches = [
            specification
            for (candidate_id, candidate_version), specification in self._specifications.items()
            if candidate_id == benchmark_id
            and (version is None or candidate_version == version)
        ]
        if not matches:
            available = self.list_ids()
            suffix = f"@{version}" if version else ""
            raise RegistryError(
                f"Benchmark specification '{benchmark_id}{suffix}' not found. "
                f"Available: {available}"
            )
        return max(matches, key=lambda item: self._version_key(item.metadata.version))

    def list(
        self,
        *,
        version: str | None = None,
    ) -> list[BenchmarkSpecification]:
        """List registered specifications in stable ID/version order."""
        values = [
            specification
            for (_, candidate_version), specification in self._specifications.items()
            if version is None or candidate_version == version
        ]
        return sorted(
            values,
            key=lambda item: (item.metadata.id, self._version_key(item.metadata.version)),
        )

    def list_ids(self, *, version: str | None = None) -> builtins.list[str]:
        """List unique registered benchmark IDs."""
        return sorted({item.metadata.id for item in self.list(version=version)})

    def search(
        self,
        query: str | None = None,
        *,
        tags: set[str] | builtins.list[str] | None = None,
        version: str | None = None,
    ) -> builtins.list[BenchmarkSpecification]:
        """Search metadata by text, required tags, and exact version."""
        normalized_query = query.lower() if query else None
        required_tags = set(tags or [])
        matches: list[BenchmarkSpecification] = []
        for specification in self.list(version=version):
            metadata = specification.metadata
            searchable = " ".join(
                [metadata.id, metadata.title, metadata.description, *metadata.tags, *metadata.domains]
            ).lower()
            if normalized_query and normalized_query not in searchable:
                continue
            if not required_tags.issubset(set(metadata.tags)):
                continue
            matches.append(specification)
        return matches

    def validate(self) -> builtins.list[str]:
        """Return registry-level validation errors; an empty list means valid."""
        errors: list[str] = []
        for specification in self._specifications.values():
            try:
                BenchmarkSpecification.model_validate(specification.model_dump())
            except ValueError as error:
                errors.append(f"{specification.metadata.id}: {error}")
        return errors

    def clear(self) -> None:
        """Remove registered specifications, primarily for isolated test suites."""
        self._specifications.clear()

    @staticmethod
    def _version_key(version: str) -> tuple[int, int, int]:
        """Convert the supported SemVer subset into a sortable tuple."""
        major, minor, patch = (int(part) for part in version.split("."))
        return major, minor, patch


SpecificationRegistry = BenchmarkSpecificationRegistry
"""Short public alias for :class:`BenchmarkSpecificationRegistry`."""


benchmark_spec_registry = BenchmarkSpecificationRegistry()
"""Global registry for declarative benchmark specifications."""
