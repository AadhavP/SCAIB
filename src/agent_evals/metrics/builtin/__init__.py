"""Versioned high-value PBMC metric definitions and computers."""

from agent_evals.metrics.builtin.annotation import annotation_definitions
from agent_evals.metrics.builtin.clustering import clustering_definitions
from agent_evals.metrics.builtin.differential_expression import de_definitions
from agent_evals.metrics.builtin.embedding import embedding_definitions
from agent_evals.metrics.builtin.integration import integration_definitions
from agent_evals.metrics.registry import MetricRegistry, metric_registry


def register_builtin_metrics(registry: MetricRegistry = metric_registry) -> None:
    """Register the v1 objective metric catalog idempotently."""
    for definition, computer in [
        *annotation_definitions(),
        *clustering_definitions(),
        *integration_definitions(),
        *embedding_definitions(),
        *de_definitions(),
    ]:
        registry.register(definition, computer, replace=True)


register_builtin_metrics()

__all__ = ["register_builtin_metrics"]
