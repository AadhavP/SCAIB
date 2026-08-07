"""Frozen profile for PBMC differential-expression evaluations."""

from agent_evals.evaluation.profiles.base import (
    BenchmarkMetricProfile,
    MetricGroupProfile,
    MetricProfileEntry,
)


def pbmc_de_profile() -> BenchmarkMetricProfile:
    return BenchmarkMetricProfile(
        benchmark="pbmc_de",
        metric_groups={
            "biology": MetricGroupProfile(
                weight=1.0,
                metrics={"differential_expression.pseudobulk_recall": MetricProfileEntry(weight=1.0)},
            )
        },
    )


__all__ = ["pbmc_de_profile"]
