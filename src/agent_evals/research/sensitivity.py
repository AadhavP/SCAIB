"""Sensitivity and correlation checks for a scientific metric profile.

Metric stacks can look rigorous while being dominated by one parameter choice or
by several nearly duplicate measurements. These summaries do not choose weights;
they expose how much the reported measurements move under declared perturbations
and which metric pairs are redundant.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetricSensitivityObservation(BaseModel):
    """One metric value under one declared configuration."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    scenario_id: str = Field(default="default", min_length=1)
    value: float = Field(ge=0, le=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        """Reject non-finite values that could corrupt range calculations."""
        if not math.isfinite(value):
            raise ValueError("sensitivity value must be finite")
        return value


class MetricSensitivityResult(BaseModel):
    """Summary of one metric's response to configuration changes."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    configurations: int = Field(ge=0)
    scenarios: int = Field(default=0, ge=0)
    minimum: float | None = Field(default=None, ge=0, le=1)
    maximum: float | None = Field(default=None, ge=0, le=1)
    range: float | None = Field(default=None, ge=0, le=1)
    mean_absolute_change: float | None = Field(default=None, ge=0, le=1)
    rank_stability: float | None = Field(default=None, ge=-1, le=1)
    limitations: list[str] = Field(default_factory=list)


class MetricCorrelationVector(BaseModel):
    """Metric values indexed by the same replicate/configuration identity."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(min_length=1)
    values: dict[str, float] = Field(default_factory=dict)

    @field_validator("values")
    @classmethod
    def finite_values(cls, values: dict[str, float]) -> dict[str, float]:
        """Reject non-finite vectors before correlation is calculated."""
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("correlation vector contains a non-finite value")
        return values


class MetricCorrelationResult(BaseModel):
    """Pairwise correlation and overlap evidence for two metrics."""

    model_config = ConfigDict(extra="forbid")

    left_metric_id: str
    right_metric_id: str
    n_shared: int = Field(ge=0)
    pearson: float | None = Field(default=None, ge=-1, le=1)
    high_correlation: bool = False
    limitations: list[str] = Field(default_factory=list)


class MetricValidationStudy(BaseModel):
    """Combined sensitivity/correlation report attached to the metrics gate."""

    model_config = ConfigDict(extra="forbid")

    study_version: str = "1.0.0"
    sensitivity: list[MetricSensitivityResult] = Field(default_factory=list)
    correlations: list[MetricCorrelationResult] = Field(default_factory=list)
    high_correlation_threshold: float = Field(default=0.9, ge=0, le=1)
    minimum_shared_observations: int = Field(default=3, ge=2)
    minimum_sensitivity_scenarios: int = Field(default=2, ge=1)
    limitations: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the study ran without missing measurement evidence."""
        return (
            not self.limitations
            and all(not item.limitations for item in self.sensitivity)
            and all(not item.limitations for item in self.correlations)
        )


def build_metric_sensitivity_study(
    observations: Sequence[MetricSensitivityObservation],
    *,
    high_range_threshold: float = 0.1,
) -> list[MetricSensitivityResult]:
    """Summarize configuration ranges and rank stability for each metric."""
    grouped: dict[str, list[MetricSensitivityObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.metric_id].append(observation)
    results: list[MetricSensitivityResult] = []
    for metric_id, values in sorted(grouped.items()):
        keys = [(item.configuration_id, item.scenario_id) for item in values]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"metric '{metric_id}' has duplicate configuration/scenario IDs"
            )
        numbers = [item.value for item in values]
        minimum = min(numbers) if numbers else None
        maximum = max(numbers) if numbers else None
        span = maximum - minimum if minimum is not None and maximum is not None else None
        by_scenario: dict[str, dict[str, float]] = defaultdict(dict)
        for item in values:
            by_scenario[item.scenario_id][item.configuration_id] = item.value
        configurations = sorted({item.configuration_id for item in values})
        changes = [
            abs(left - right)
            for scenario_values in by_scenario.values()
            for left, right in itertools.combinations(
                [scenario_values[configuration] for configuration in configurations if configuration in scenario_values],
                2,
            )
        ]
        limitations: list[str] = []
        if len(configurations) < 2:
            limitations.append("only one configuration was measured")
        if len(by_scenario) < 2:
            limitations.append("rank stability was unmeasured: only one scenario")
        if span is not None and span > high_range_threshold:
            limitations.append(
                f"metric range {span:.4g} exceeds sensitivity threshold {high_range_threshold:.4g}"
            )
        rank_stability = _rank_stability(by_scenario, configurations)
        results.append(
            MetricSensitivityResult(
                metric_id=metric_id,
                configurations=len(configurations),
                scenarios=len(by_scenario),
                minimum=minimum,
                maximum=maximum,
                range=span,
                mean_absolute_change=sum(changes) / len(changes) if changes else None,
                rank_stability=rank_stability,
                limitations=limitations,
            )
        )
    return results


def build_metric_correlation_study(
    vectors: Sequence[MetricCorrelationVector],
    *,
    high_correlation_threshold: float = 0.9,
    minimum_shared_observations: int = 3,
) -> list[MetricCorrelationResult]:
    """Calculate pairwise Pearson correlations on shared replicate identities."""
    by_metric = {vector.metric_id: vector for vector in vectors}
    if len(by_metric) != len(vectors):
        raise ValueError("metric correlation vector IDs must be unique")
    results: list[MetricCorrelationResult] = []
    for left_id, right_id in itertools.combinations(sorted(by_metric), 2):
        left = by_metric[left_id].values
        right = by_metric[right_id].values
        shared = sorted(set(left) & set(right))
        limitations: list[str] = []
        correlation = _pearson([left[key] for key in shared], [right[key] for key in shared])
        if len(shared) < minimum_shared_observations:
            limitations.append(
                f"only {len(shared)} shared replicate/configuration identities; "
                f"requires {minimum_shared_observations}"
            )
        results.append(
            MetricCorrelationResult(
                left_metric_id=left_id,
                right_metric_id=right_id,
                n_shared=len(shared),
                pearson=correlation,
                high_correlation=(
                    correlation is not None
                    and abs(correlation) >= high_correlation_threshold
                ),
                limitations=limitations,
            )
        )
    return results


def build_metric_validation_study(
    observations: Sequence[MetricSensitivityObservation],
    vectors: Sequence[MetricCorrelationVector],
    *,
    high_correlation_threshold: float = 0.9,
    high_range_threshold: float = 0.1,
    minimum_shared_observations: int = 3,
    minimum_sensitivity_scenarios: int = 2,
) -> MetricValidationStudy:
    """Build the combined report used as metrics-gate evidence."""
    sensitivity = build_metric_sensitivity_study(
        observations,
        high_range_threshold=high_range_threshold,
    )
    correlations = build_metric_correlation_study(
        vectors,
        high_correlation_threshold=high_correlation_threshold,
        minimum_shared_observations=minimum_shared_observations,
    )
    limitations: list[str] = []
    if not observations:
        limitations.append("no metric sensitivity observations were supplied")
    if len(vectors) < 2:
        limitations.append("fewer than two metric vectors were supplied for correlation")
    if any(
        result.scenarios < minimum_sensitivity_scenarios
        for result in sensitivity
    ):
        limitations.append(
            "one or more metrics lack the minimum number of sensitivity scenarios"
        )
    return MetricValidationStudy(
        sensitivity=sensitivity,
        correlations=correlations,
        high_correlation_threshold=high_correlation_threshold,
        minimum_shared_observations=minimum_shared_observations,
        minimum_sensitivity_scenarios=minimum_sensitivity_scenarios,
        limitations=list(dict.fromkeys(limitations)),
    )


def _rank_stability(
    by_scenario: dict[str, dict[str, float]],
    configurations: Sequence[str],
) -> float | None:
    """Measure agreement of configuration rankings across scenarios."""
    complete = [
        values
        for values in by_scenario.values()
        if all(configuration in values for configuration in configurations)
    ]
    if len(complete) < 2 or len(configurations) < 2:
        return None
    ranks = [_ranks([values[configuration] for configuration in configurations]) for values in complete]
    correlations = [
        correlation
        for left, right in itertools.combinations(ranks, 2)
        if (correlation := _pearson(left, right)) is not None
    ]
    return sum(correlations) / len(correlations) if correlations else None


def _ranks(values: Sequence[float]) -> list[float]:
    """Return average ranks with deterministic tie handling."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Calculate Pearson correlation without an optional scientific dependency."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


__all__ = [
    "MetricCorrelationResult",
    "MetricCorrelationVector",
    "MetricSensitivityObservation",
    "MetricSensitivityResult",
    "MetricValidationStudy",
    "build_metric_correlation_study",
    "build_metric_sensitivity_study",
    "build_metric_validation_study",
]
