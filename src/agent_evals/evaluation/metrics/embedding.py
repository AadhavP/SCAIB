"""Generic embedding metric catalog."""

from agent_evals.evaluation.metrics.catalog import register_legacy_catalog
from agent_evals.evaluation.metrics.registry import MetricRegistry, metric_registry


def register_embedding_metrics(registry: MetricRegistry = metric_registry) -> None:
    """Register trustworthiness, continuity, kNN, and stress adapters."""
    register_legacy_catalog(registry)


__all__ = ["register_embedding_metrics"]
