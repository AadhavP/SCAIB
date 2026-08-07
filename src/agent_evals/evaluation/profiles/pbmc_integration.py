"""Frozen profile for PBMC integration-focused evaluations."""

from agent_evals.evaluation.profiles.base import (
    BenchmarkMetricProfile,
    MetricGroupProfile,
    MetricProfileEntry,
)


def pbmc_integration_profile() -> BenchmarkMetricProfile:
    return BenchmarkMetricProfile(
        benchmark="pbmc_integration",
        metric_groups={
            "technical": MetricGroupProfile(
                weight=1.0,
                metrics={
                    "batch_integration.iLISI": MetricProfileEntry(weight=0.6),
                    "batch_integration.graph_connectivity": MetricProfileEntry(weight=0.4),
                },
            )
        },
    )


__all__ = ["pbmc_integration_profile"]
