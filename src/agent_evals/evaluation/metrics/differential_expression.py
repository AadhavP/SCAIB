"""Generic differential-expression metric catalog."""

from agent_evals.evaluation.metrics.catalog import register_legacy_catalog
from agent_evals.evaluation.metrics.registry import MetricRegistry, metric_registry


def register_differential_expression_metrics(registry: MetricRegistry = metric_registry) -> None:
    """Register DE ranking, correlation, and direction adapters."""
    register_legacy_catalog(registry)


__all__ = ["register_differential_expression_metrics"]
