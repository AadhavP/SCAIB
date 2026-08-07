"""Registry and contracts for deterministic scientific metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_evals.benchmarks.schema import Direction
from agent_evals.core.exceptions import RegistryError
from agent_evals.evaluators.models import EvaluationLevel

if TYPE_CHECKING:
    from agent_evals.agents.trajectory import AgentRun
    from agent_evals.benchmarks.schema import MetricSpecification
    from agent_evals.environment.models import EpisodeSnapshot
    from agent_evals.evaluators.models import TaskInstance


@dataclass(frozen=True)
class MetricComputation:
    """Raw output and evidence returned by a registered metric."""

    raw_value: Any
    evidence: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MetricContext:
    """Read-only inputs made available to a metric implementation."""

    task_instance: TaskInstance
    run: AgentRun
    snapshot: EpisodeSnapshot
    specification: MetricSpecification | None = None


MetricComputer = Callable[[MetricContext], MetricComputation]


@dataclass(frozen=True)
class RegisteredMetric:
    """Executable metric definition held by the registry."""

    metric_id: str
    name: str
    description: str
    level: EvaluationLevel
    direction: Direction
    required_artifacts: tuple[str, ...]
    compute: MetricComputer


class MetricRegistry:
    """Register and resolve deterministic metric implementations."""

    def __init__(self) -> None:
        self._metrics: dict[str, RegisteredMetric] = {}

    def register(
        self,
        metric_id: str,
        *,
        name: str,
        description: str,
        level: EvaluationLevel,
        direction: Direction,
        required_artifacts: tuple[str, ...] = (),
    ) -> Callable[[MetricComputer], MetricComputer]:
        """Decorator registering a metric under a stable benchmark ID."""

        def decorator(compute: MetricComputer) -> MetricComputer:
            if metric_id in self._metrics:
                raise RegistryError(f"Metric '{metric_id}' is already registered.")
            self._metrics[metric_id] = RegisteredMetric(
                metric_id=metric_id,
                name=name,
                description=description,
                level=level,
                direction=direction,
                required_artifacts=required_artifacts,
                compute=compute,
            )
            return compute

        return decorator

    def get(self, metric_id: str) -> RegisteredMetric:
        """Return one metric or raise a useful registry error."""
        if metric_id not in self._metrics:
            raise RegistryError(
                f"Metric '{metric_id}' not found in registry. Available: {self.list_metrics()}"
            )
        return self._metrics[metric_id]

    def list_metrics(self) -> list[str]:
        """Return registered metric IDs in stable order."""
        return sorted(self._metrics)

    def clear(self) -> None:
        """Clear registrations for isolated tests."""
        self._metrics.clear()


metric_registry = MetricRegistry()


__all__ = [
    "MetricComputation",
    "MetricContext",
    "MetricRegistry",
    "RegisteredMetric",
    "metric_registry",
]
