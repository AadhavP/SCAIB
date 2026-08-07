"""Registry for generic scientific metric adapters."""

from __future__ import annotations

import builtins
from collections.abc import Callable

from agent_evals.evaluation.metrics.base import ScientificMetric


class MetricRegistry:
    """Register concrete metric adapters by stable name and category."""

    def __init__(self) -> None:
        self._metrics: dict[str, ScientificMetric] = {}

    def register(
        self,
        metric: ScientificMetric | type[ScientificMetric] | None = None,
    ) -> ScientificMetric | Callable[[type[ScientificMetric]], type[ScientificMetric]]:
        """Register an instance or return a class decorator."""
        if metric is None:
            return self._decorator
        if isinstance(metric, type):
            instance = metric()
        else:
            instance = metric
        if instance.name in self._metrics:
            raise ValueError(f"metric '{instance.name}' is already registered")
        self._metrics[instance.name] = instance
        return instance

    def _decorator(self, metric_type: type[ScientificMetric]) -> type[ScientificMetric]:
        """Register a metric class and preserve decorator semantics."""
        self.register(metric_type)
        return metric_type

    def get(self, name: str) -> ScientificMetric:
        """Resolve one metric by stable name."""
        try:
            return self._metrics[name]
        except KeyError as error:
            raise KeyError(f"metric '{name}' is not registered") from error

    def list(self, category: str | None = None) -> builtins.list[ScientificMetric]:
        """List metrics in stable name order, optionally by category."""
        values = list(self._metrics.values())
        if category is not None:
            values = [metric for metric in values if metric.category == category]
        return sorted(values, key=lambda metric: metric.name)

    def search(self, query: str = "", category: str | None = None) -> builtins.list[ScientificMetric]:
        """Search metric names/categories/roles."""
        normalized = query.lower()
        return [
            metric
            for metric in self.list(category)
            if normalized in f"{metric.name} {metric.category} {metric.role.value}".lower()
        ]

    def clear(self) -> None:
        """Clear the registry for isolated tests."""
        self._metrics.clear()


metric_registry = MetricRegistry()

__all__ = ["MetricRegistry", "metric_registry"]
