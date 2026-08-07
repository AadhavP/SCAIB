"""Generic annotation metric catalog backed by the shared registry."""

from agent_evals.evaluation.metrics.catalog import register_legacy_catalog
from agent_evals.evaluation.metrics.registry import MetricRegistry, metric_registry


def register_annotation_metrics(registry: MetricRegistry = metric_registry) -> None:
    """Register annotation adapters from the canonical metric catalog."""
    register_legacy_catalog(registry)


__all__ = ["register_annotation_metrics"]
