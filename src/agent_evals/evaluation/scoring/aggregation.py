"""Frozen-weight geometric aggregation with applicability evidence."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.evaluation.profiles.base import MetricGroupProfile, MetricProfileEntry


class MetricScoreInput(BaseModel):
    """Normalized metric value supplied to a profile scorer."""

    model_config = ConfigDict(extra="allow")

    name: str
    value: float | None = Field(default=None, ge=0, le=1)
    applicable: bool = True
    structurally_ineligible: bool = False
    status: str = "computed"


class DomainScore(BaseModel):
    """Domain aggregate with included/excluded/failure breakdown."""

    domain: str
    value: float | None = Field(default=None, ge=0, le=1)
    weight: float = Field(gt=0)
    included_metrics: list[str] = Field(default_factory=list)
    excluded_metrics: list[str] = Field(default_factory=list)
    failed_metrics: list[str] = Field(default_factory=list)
    formula: str


class WeightedGeometricAggregator:
    """Aggregate using frozen weights without rewarding metric avoidance."""

    def aggregate(self, domain: str, profile: MetricGroupProfile, results: list[MetricScoreInput]) -> DomainScore:
        """Compute one weighted geometric domain score."""
        by_name = {result.name: result for result in results}
        included: list[tuple[str, float, float]] = []
        excluded: list[str] = []
        failed: list[str] = []
        entries = dict(profile.metrics)
        if profile.external_score is not None and profile.external_score not in entries:
            entries[profile.external_score] = MetricProfileEntry(weight=1.0, required=True)
        for name, entry in entries.items():
            result = by_name.get(name)
            if result is not None and result.structurally_ineligible:
                excluded.append(name)
                continue
            if result is None and not entry.required:
                excluded.append(name)
                continue
            value = 0.0 if result is None or result.value is None else result.value
            if result is None or result.status != "computed":
                failed.append(name)
            included.append((name, entry.weight, max(0.0, min(1.0, value))))
        if entries and any(entry.required and name in excluded for name, entry in entries.items()):
            aggregate: float | None = None
        elif not included:
            aggregate = None
        elif any(value <= 0 for _, _, value in included):
            aggregate = 0.0
        else:
            denominator = sum(weight for _, weight, _ in included)
            aggregate = math.exp(
                sum(weight * math.log(max(value, 1e-12)) for _, weight, value in included)
                / denominator
            )
        formula = "geometric_mean(" + ", ".join(f"{name}^{weight:g}" for name, weight, _ in included) + ")"
        return DomainScore(
            domain=domain,
            value=aggregate,
            weight=profile.weight,
            included_metrics=[name for name, _, _ in included],
            excluded_metrics=excluded,
            failed_metrics=failed,
            formula=formula,
        )


__all__ = ["DomainScore", "MetricScoreInput", "WeightedGeometricAggregator"]
