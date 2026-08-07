"""Generic scientific metric engine and adapter-backed metric catalog."""

from agent_evals.evaluation.metrics.base import (
    ArtifactBundle,
    EvaluationContext,
    MetricApplicability,
    MetricRole,
    MetricStatus,
    RawMetricResult,
    ReferenceBundle,
    ScientificMetric,
    ScientificMetricResult,
    ScoreAnchors,
)
from agent_evals.evaluation.metrics.catalog import register_legacy_catalog
from agent_evals.evaluation.metrics.engine import ScientificMetricEngine
from agent_evals.evaluation.metrics.registry import MetricRegistry, metric_registry

register_legacy_catalog(metric_registry)

__all__ = [
    "ArtifactBundle",
    "EvaluationContext",
    "MetricApplicability",
    "MetricRegistry",
    "MetricRole",
    "MetricStatus",
    "RawMetricResult",
    "ReferenceBundle",
    "ScientificMetric",
    "ScientificMetricEngine",
    "ScientificMetricResult",
    "ScoreAnchors",
    "metric_registry",
]
