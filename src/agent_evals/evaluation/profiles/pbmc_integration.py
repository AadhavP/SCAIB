"""Frozen profile for PBMC integration-focused evaluations.

The axes follow scIB's central distinction: removing batch structure is not useful
if it also erases biology. Each metric is computed by the versioned metric
registry, with scib-metrics-backed metrics carrying their package/version metadata
in the result.
"""

from agent_evals.evaluation.profiles.base import (
    BenchmarkMetricProfile,
    MetricGroupProfile,
    MetricProfileEntry,
)


def pbmc_integration_profile() -> BenchmarkMetricProfile:
    """Return the benchmark-owned integration score profile."""
    return BenchmarkMetricProfile(
        benchmark="pbmc_integration",
        metric_groups={
            "technical_mixing": MetricGroupProfile(
                weight=0.4,
                metrics={
                    "batch_integration.iLISI": MetricProfileEntry(weight=0.35),
                    "batch_integration.kBET": MetricProfileEntry(weight=0.35),
                    "batch_integration.BRAS": MetricProfileEntry(weight=0.30),
                },
            ),
            "biological_conservation": MetricGroupProfile(
                weight=0.6,
                metrics={
                    "biological_conservation.cell_type_asw": MetricProfileEntry(
                        weight=0.5
                    ),
                    "biological_conservation.graph_connectivity": MetricProfileEntry(
                        weight=0.5
                    ),
                },
            ),
        },
        metadata={
            "scientific_reference": "scIB-style integration evaluation",
            "backend": "scib-metrics",
        },
    )


__all__ = ["pbmc_integration_profile"]
