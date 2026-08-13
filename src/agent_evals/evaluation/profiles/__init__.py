"""Declarative benchmark metric profiles."""

from agent_evals.evaluation.profiles.base import (
    BenchmarkMetricProfile,
    MetricGroupProfile,
    MetricProfileEntry,
    load_metric_profile,
)
from agent_evals.evaluation.profiles.pbmc_annotation import pbmc_annotation_profile
from agent_evals.evaluation.profiles.pbmc_de import pbmc_de_profile
from agent_evals.evaluation.profiles.pbmc_integration import pbmc_integration_profile
from agent_evals.evaluation.profiles.resolution import (
    BUILTIN_PROFILES,
    profile_external_scores,
    profile_metric_ids,
    profiled_benchmark_ids,
    resolve_metric_profile,
    unregistered_profile_metrics,
)

__all__ = [
    "BUILTIN_PROFILES",
    "BenchmarkMetricProfile",
    "MetricGroupProfile",
    "MetricProfileEntry",
    "load_metric_profile",
    "pbmc_annotation_profile",
    "pbmc_de_profile",
    "pbmc_integration_profile",
    "profile_external_scores",
    "profile_metric_ids",
    "profiled_benchmark_ids",
    "resolve_metric_profile",
    "unregistered_profile_metrics",
]
