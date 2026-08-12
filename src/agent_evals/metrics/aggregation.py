"""Declarative metric-group aggregation."""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evals.metrics.models import MetricRole
from agent_evals.metrics.results import MetricResult, MetricStatus


class MetricWeight(BaseModel):
    """One frozen metric contribution inside a group."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    weight: float = Field(gt=0)
    role: MetricRole | None = None


class MetricGroup(BaseModel):
    """Weighted group with explicit aggregation and minimum evidence."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    metrics: list[MetricWeight] = Field(min_length=1)
    aggregation: str = "weighted_mean"
    minimum_required: int = 1
    contributes_to_primary: bool = True

    @model_validator(mode="after")
    def validate_group(self) -> MetricGroup:
        """Reject duplicate metrics and unsupported aggregation names."""
        ids = [item.metric_id for item in self.metrics]
        if len(ids) != len(set(ids)):
            raise ValueError(f"metric group '{self.group_id}' contains duplicates")
        if self.aggregation not in {"weighted_mean", "weighted_geometric_mean"}:
            raise ValueError(f"unsupported metric aggregation '{self.aggregation}'")
        if self.minimum_required < 1 or self.minimum_required > len(self.metrics):
            raise ValueError("minimum_required must be between 1 and metric count")
        return self


class AggregationResult(BaseModel):
    """Aggregated group score plus transparent inclusion evidence."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    value: float | None
    included_metric_ids: list[str] = Field(default_factory=list)
    excluded_metric_ids: list[str] = Field(default_factory=list)
    missing_required_count: int = 0
    formula: str


def aggregate_group(
    group: MetricGroup,
    results: Iterable[MetricResult],
) -> AggregationResult:
    """Aggregate structurally eligible results without candidate renormalization."""
    by_id = {result.metric_id: result for result in results}
    included: list[tuple[float, float]] = []
    excluded: list[str] = []
    for metric in group.metrics:
        result = by_id.get(metric.metric_id)
        if result is None or result.status == MetricStatus.INELIGIBLE:
            excluded.append(metric.metric_id)
            continue
        score = result.normalized_value
        if score is None:
            score = 0.0
        included.append((score, metric.weight))
    missing_required = len(group.metrics) - len(excluded)
    if missing_required < group.minimum_required:
        value = None
    elif group.aggregation == "weighted_mean":
        denominator = sum(weight for _, weight in included)
        value = sum(score * weight for score, weight in included) / denominator if denominator else 0.0
    else:
        denominator = sum(weight for _, weight in included)
        value = (
            math.exp(sum(weight * math.log(max(score, 1e-12)) for score, weight in included) / denominator)
            if denominator
            else 0.0
        )
    formula = (
        f"{group.aggregation}("
        + ", ".join(f"{metric.metric_id}*{metric.weight:g}" for metric in group.metrics if metric.metric_id not in excluded)
        + ")"
    )
    return AggregationResult(
        group_id=group.group_id,
        value=value,
        included_metric_ids=[metric.metric_id for metric in group.metrics if metric.metric_id not in excluded],
        excluded_metric_ids=excluded,
        missing_required_count=missing_required,
        formula=formula,
    )


__all__ = ["AggregationResult", "MetricGroup", "MetricWeight", "aggregate_group"]
