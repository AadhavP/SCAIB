"""Versioned scientific metric definitions, computation, and aggregation."""

from agent_evals.metrics.aggregation import (
    AggregationResult,
    MetricGroup,
    MetricWeight,
    aggregate_group,
)
from agent_evals.metrics.applicability import (
    ApplicabilityContext,
    ApplicabilityResult,
    MetricApplicabilityEngine,
)
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.models import (
    MetricApplicability,
    MetricCategory,
    MetricDefinition,
    MetricDirection,
    MetricRole,
    NormalizationSpec,
)
from agent_evals.metrics.normalization import NormalizationEngine
from agent_evals.metrics.registry import (
    MetricComputation,
    MetricRegistry,
    metric_registry,
)
from agent_evals.metrics.results import MetricResult, MetricStatus

__all__ = [
    "AggregationResult",
    "ApplicabilityContext",
    "ApplicabilityResult",
    "MetricApplicability",
    "MetricApplicabilityEngine",
    "MetricCategory",
    "MetricComputation",
    "MetricDefinition",
    "MetricDirection",
    "MetricGroup",
    "MetricRegistry",
    "MetricResult",
    "MetricRole",
    "MetricStatus",
    "MetricWeight",
    "NormalizationEngine",
    "NormalizationSpec",
    "ScientificMetricContext",
    "aggregate_group",
    "metric_registry",
]

from agent_evals.metrics.builtin import register_builtin_metrics

register_builtin_metrics()
