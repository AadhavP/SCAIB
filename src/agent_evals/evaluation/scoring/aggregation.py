"""Frozen-weight geometric aggregation with applicability evidence."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.evaluation.profiles.base import MetricGroupProfile, MetricProfileEntry
from agent_evals.metrics.results import MetricResult, MetricStatus


class MetricScoreInput(BaseModel):
    """Normalized metric value supplied to a profile scorer."""

    model_config = ConfigDict(extra="allow")

    name: str
    value: float | None = Field(default=None, ge=0, le=1)
    applicable: bool = True
    structurally_ineligible: bool = False
    #: Typed rather than a bare ``str``. This field held the literal
    #: ``"computed"`` and was compared against the same literal below, while the
    #: caller filled it from ``MetricStatus``. Renaming a status value would then
    #: have moved every metric into ``failed_metrics`` with nothing raising --
    #: the report would have claimed a total metric failure on a healthy run.
    status: MetricStatus = MetricStatus.SCORED

    @classmethod
    def from_metric_result(cls, result: MetricResult) -> MetricScoreInput:
        """Translate one computed metric into an aggregator input.

        A classmethod rather than a literal at the call site because the mark
        below decides whether a metric is dropped or charged as a failure, and
        getting it wrong fails *silently* -- the score simply comes out lower with
        no error anywhere. ``metric_id`` is the dotted registry id the profiles
        key on; ``metric_name`` is a human-readable title, and feeding that here
        would make every profile lookup miss.
        """
        return cls(
            name=result.metric_id,
            value=result.normalized_value,
            applicable=result.eligible,
            # Asks the vocabulary which statuses leave the aggregate rather than
            # naming one here. Spelling ``is INELIGIBLE`` meant a second excluding
            # status was included at 0.0 and reported as a metric the agent failed.
            structurally_ineligible=result.status.excluded_from_scoring,
            status=result.status,
        )

    @classmethod
    def from_external_score(cls, name: str, value: float | None) -> MetricScoreInput:
        """Translate a score computed outside the metric registry.

        Built even when ``value`` is ``None``, and marked excluded in that case.
        Omitting it would not leave it out: an ``external_score`` is injected as a
        *required* entry and a required entry with no result is scored ``0.0``, so
        the domain would report a zero for something nobody tried to measure --
        the opposite error to the free ``1.0`` this replaced.
        """
        measured = value is not None
        return cls(
            name=name,
            value=value,
            applicable=measured,
            structurally_ineligible=not measured,
            status=MetricStatus.SCORED if measured else MetricStatus.MISSING,
        )


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
            if result is None or result.status is not MetricStatus.SCORED:
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
