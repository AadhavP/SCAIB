"""Canonical PBMC annotation metric profile."""

from agent_evals.evaluation.profiles.base import (
    BenchmarkMetricProfile,
    MetricGroupProfile,
    MetricProfileEntry,
)


def pbmc_annotation_profile() -> BenchmarkMetricProfile:
    """Return the frozen PBMC biology/technical/robustness profile."""
    return BenchmarkMetricProfile(
        benchmark="pbmc_annotation",
        metric_groups={
            "biology": MetricGroupProfile(
                weight=0.6,
                metrics={
                    "clustering.ari": MetricProfileEntry(weight=0.4),
                    "cell_annotation.macro_f1": MetricProfileEntry(weight=0.4),
                    "cell_annotation.rare_recall": MetricProfileEntry(weight=0.2),
                },
            ),
            "technical": MetricGroupProfile(
                weight=0.2,
                metrics={
                    "batch_integration.iLISI": MetricProfileEntry(weight=0.6, required=False),
                    "biological_conservation.graph_connectivity": MetricProfileEntry(weight=0.4, required=False),
                },
            ),
            "robustness": MetricGroupProfile(
                weight=0.2,
                metrics={},
                external_score="robustness.seed_stability",
            ),
        },
    )


__all__ = ["pbmc_annotation_profile"]
