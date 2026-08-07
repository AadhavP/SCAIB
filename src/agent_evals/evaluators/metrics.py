"""Metric computation primitives for single-cell biology benchmarks."""

from collections.abc import Sequence
from typing import Any

from agent_evals.core.types import MetricScore


def compute_accuracy(
    predictions: Sequence[Any], ground_truth: Sequence[Any]
) -> MetricScore:
    """Compute simple classification accuracy score."""
    if not predictions or len(predictions) != len(ground_truth):
        return MetricScore(name="accuracy", value=0.0)

    correct = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == g)
    accuracy = correct / len(predictions)
    return MetricScore(name="accuracy", value=accuracy)


def compute_execution_time(time_seconds: float) -> MetricScore:
    """Wrap execution latency as a metric score."""
    return MetricScore(
        name="execution_time_seconds", value=time_seconds, unit="seconds"
    )
