"""Generic clustering metric catalog backed by the shared registry."""

from agent_evals.evaluation.metrics.annotation import register_annotation_metrics
from agent_evals.evaluation.metrics.registry import MetricRegistry, metric_registry


def register_clustering_metrics(registry: MetricRegistry = metric_registry) -> None:
    """Register clustering adapters."""
    register_annotation_metrics(registry)


__all__ = ["register_clustering_metrics"]
