"""Version-aware metric registry and computation contracts."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_evals.core.exceptions import RegistryError
from agent_evals.metrics.models import MetricDefinition


@dataclass(frozen=True)
class MetricComputation:
    """Raw metric output returned by a backend adapter."""

    raw_value: Any
    metadata: dict[str, Any] | None = None
    evidence: tuple[str, ...] = ()
    #: Set when no implementation of this metric exists in this deployment, as
    #: opposed to an implementation that ran and read no number. The difference
    #: decides who is accountable: a missing backend is a gap in SCAIB, and
    #: charging the agent a zero for it would score the agent on the harness's
    #: unfinished work. Kept separate from ``raw_value is None`` because both
    #: look identical at the call site and only one of them is the agent's doing.
    unavailable: bool = False


MetricComputer = Callable[[Any], MetricComputation]


class MetricRegistry:
    """Register, resolve, search, and validate versioned metrics."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], MetricDefinition] = {}
        self._computers: dict[tuple[str, str], MetricComputer] = {}

    def register(
        self,
        definition: MetricDefinition,
        compute: MetricComputer | None = None,
        *,
        replace: bool = False,
    ) -> MetricDefinition:
        """Register one definition and optional computation backend."""
        key = (definition.metric_id, definition.version)
        if key in self._definitions and not replace:
            raise RegistryError(
                f"Metric '{definition.metric_id}@{definition.version}' is already registered."
            )
        self._definitions[key] = definition
        if compute is not None:
            self._computers[key] = compute
        return definition

    def get(self, metric_id: str, version: str | None = None) -> MetricDefinition:
        """Resolve an exact version or the highest registered version."""
        matches = [
            definition
            for (candidate_id, candidate_version), definition in self._definitions.items()
            if candidate_id == metric_id
            and (version is None or candidate_version == version)
        ]
        if not matches:
            raise RegistryError(f"Metric '{metric_id}@{version}' is not registered.")
        return max(matches, key=lambda item: self._version_key(item.version))

    def get_computer(self, metric_id: str, version: str | None = None) -> MetricComputer:
        """Resolve a registered computation function."""
        definition = self.get(metric_id, version)
        try:
            return self._computers[(definition.metric_id, definition.version)]
        except KeyError as error:
            raise RegistryError(
                f"Metric '{definition.metric_id}@{definition.version}' has no computation backend."
            ) from error

    def list(
        self,
        *,
        category: str | None = None,
        role: str | None = None,
    ) -> builtins.list[MetricDefinition]:
        """List definitions in stable ID/version order."""
        values = list(self._definitions.values())
        if category is not None:
            values = [item for item in values if item.category.value == category]
        if role is not None:
            values = [item for item in values if item.role.value == role]
        return sorted(values, key=lambda item: (item.metric_id, self._version_key(item.version)))

    def search(self, query: str) -> builtins.list[MetricDefinition]:
        """Search IDs, names, descriptions, and backend names."""
        normalized = query.lower()
        return [
            definition
            for definition in self.list()
            if normalized
            in " ".join(
                [
                    definition.metric_id,
                    definition.name,
                    definition.description,
                    definition.computation_backend,
                ]
            ).lower()
        ]

    def validate(self) -> builtins.list[str]:
        """Return registry validation errors without mutating the registry."""
        errors: builtins.list[str] = []
        for definition in self._definitions.values():
            try:
                MetricDefinition.model_validate(definition.model_dump())
            except ValueError as error:
                errors.append(f"{definition.metric_id}@{definition.version}: {error}")
        return errors

    def clear(self) -> None:
        """Clear registrations for isolated test suites."""
        self._definitions.clear()
        self._computers.clear()

    @staticmethod
    def _version_key(version: str) -> tuple[int, ...]:
        """Sort practical numeric metric versions such as 1.0 or 1.0.1."""
        try:
            return tuple(int(part) for part in version.split("."))
        except ValueError as error:
            raise RegistryError(f"invalid metric version '{version}'") from error


metric_registry = MetricRegistry()

__all__ = ["MetricComputation", "MetricComputer", "MetricRegistry", "metric_registry"]
