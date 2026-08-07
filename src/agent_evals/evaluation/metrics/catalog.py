"""Automatic catalog of adapter-backed scientific metrics."""

from __future__ import annotations

from agent_evals.evaluation.metrics.adapters import LegacyMetricAdapter
from agent_evals.evaluation.metrics.registry import MetricRegistry
from agent_evals.metrics.registry import metric_registry as legacy_registry


def register_legacy_catalog(registry: MetricRegistry) -> None:
    """Register all existing versioned metrics without duplicating algorithms."""
    for definition in legacy_registry.list():
        if definition.metric_id in {metric.name for metric in registry.list()}:
            continue
        registry.register(LegacyMetricAdapter(definition))


__all__ = ["register_legacy_catalog"]
