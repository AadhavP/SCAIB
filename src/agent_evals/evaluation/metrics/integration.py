"""Generic integration metric catalog with optional scIB-metrics backends."""

from agent_evals.evaluation.metrics.catalog import register_legacy_catalog
from agent_evals.evaluation.metrics.registry import MetricRegistry, metric_registry


def register_integration_metrics(registry: MetricRegistry = metric_registry) -> None:
    """Register iLISI, kBET, graph, PCR, and preservation adapters."""
    register_legacy_catalog(registry)


__all__ = ["register_integration_metrics"]
